#!/usr/bin/env python3
"""mobench 本地与 CI 共用的单一验证入口。

串行执行编译、版本一致、Schema/YAML 校验、仓库卫生、AST 守门、
纯核心依赖隔离冒烟、ruff 与 pytest 分支覆盖率检查。
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_step(name: str, fn) -> bool:
    print(f"==> [{name}]")
    ok, detail = fn()
    status = "PASS" if ok else "FAIL"
    print(f"    {status}: {detail}")
    return ok


def check_compileall() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "mobench", "scripts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip() or f"rc={proc.returncode}"
    return True, "mobench 与 scripts 字节码编译通过"


def check_version_sync() -> tuple[bool, str]:
    versions: dict[str, str] = {}

    init_text = (REPO_ROOT / "mobench" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.M)
    if not m:
        return False, "mobench/__init__.py 缺少 __version__"
    versions["mobench/__init__.py"] = m.group(1)

    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    proj_sec = re.search(r"^\[project\]\s*(.*?)(?=^\[|\Z)", pyproject_text, re.M | re.S)
    m = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', proj_sec.group(1) if proj_sec else "", re.M)
    if not m:
        return False, "pyproject.toml [project] 缺少 version"
    versions["pyproject.toml"] = m.group(1)

    pkg = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    versions["package.json"] = str(pkg.get("version", ""))

    plugin = json.loads((REPO_ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    versions[".claude-plugin/plugin.json"] = str(plugin.get("version", ""))

    market = json.loads((REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    plugins = market.get("plugins") or []
    versions[".claude-plugin/marketplace.json"] = str(plugins[0].get("version", "")) if plugins else ""

    unique = set(versions.values())
    if len(unique) != 1 or "" in unique:
        return False, f"版本号不一致: {versions}"
    return True, f"5 处版本号一致 ({next(iter(unique))})"


def _tracked_files(*patterns: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    out = []
    for pat in patterns:
        out.extend(glob.glob(str(REPO_ROOT / pat), recursive=True))
    return sorted({os.path.relpath(p, REPO_ROOT) for p in out})


def check_schema_and_yaml() -> tuple[bool, str]:
    try:
        import jsonschema
        import yaml
    except ImportError as exc:
        return False, f"缺少 PyYAML 或 jsonschema: {exc}"

    json_files = _tracked_files("*.json", "**/*.json")
    yaml_files = _tracked_files("*.yml", "**/*.yml", "*.yaml", "**/*.yaml")
    for rel in json_files:
        path = REPO_ROOT / rel
        if path.is_file():
            json.loads(path.read_text(encoding="utf-8"))
    for rel in yaml_files:
        path = REPO_ROOT / rel
        if path.is_file():
            yaml.safe_load(path.read_text(encoding="utf-8"))

    schema = json.loads((REPO_ROOT / "schema" / "mobench.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft7Validator.check_schema(schema)

    templates = sorted(
        list((REPO_ROOT / "examples").glob("**/*.yml"))
        + list((REPO_ROOT / "configs" / "templates").glob("*.yml"))
    )
    for path in templates:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=data, schema=schema)

    sys.path.insert(0, str(REPO_ROOT))
    from mobench.config import BUILTIN_ENGINES

    schema_engines = tuple(schema["properties"]["engine"]["enum"])
    adapter_engines = tuple(
        sorted(
            p.stem
            for p in (REPO_ROOT / "mobench" / "adapters").glob("*.py")
            if p.name not in ("__init__.py", "base.py")
        )
    )
    if tuple(sorted(BUILTIN_ENGINES)) != tuple(sorted(schema_engines)) or tuple(sorted(BUILTIN_ENGINES)) != adapter_engines:
        return (
            False,
            f"engine 枚举不一致: BUILTIN={BUILTIN_ENGINES}, schema={schema_engines}, adapters={adapter_engines}",
        )
    return True, f"校验 {len(json_files)} 个 JSON, {len(yaml_files)} 个 YAML, {len(templates)} 个模板通过"


def check_repo_hygiene() -> tuple[bool, str]:
    personal_rx = re.compile(
        r"("
        + r"/ho" + r"me/[a-zA-Z0-9._-]+/"
        + r"|/Us" + r"ers/[a-zA-Z0-9._-]+/"
        + r"|~/" + r"project/"
        + r"|[A-Za-z]:\\\\"
        + r"|/m" + r"nt/[a-zA-Z]/"
        + r")"
    )
    stale_rx = re.compile(r"(" + r"open" + r"spec/" + r"|" + r"src/" + r"engopt/" + r")")
    excluded = {".github/workflows/ci.yml", "tools/adb-unc-wrapper.sh", "scripts/verify.py"}

    hits = []
    for rel in _tracked_files():
        if rel in excluded:
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if personal_rx.search(line) or stale_rx.search(line):
                hits.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    if hits:
        return False, "发现个人路径或过期内部引用:\n      " + "\n      ".join(hits[:10])
    return True, "未发现个人本地路径或过期内部引用"


def _signature_shape(fn_node: ast.FunctionDef) -> list[str]:
    args = [a.arg for a in fn_node.args.posonlyargs + fn_node.args.args if a.arg != "self"]
    return args


def _has_bare_open(tree: ast.AST) -> list[int]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            parent = parents.get(id(node))
            if not (isinstance(parent, ast.withitem) and parent.context_expr is node):
                lines.append(node.lineno)
    return lines


def _extract_non_doc_strings(tree: ast.AST) -> list[tuple[int, str]]:
    doc_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                doc_nodes.add(id(first.value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_nodes:
            out.append((node.lineno, node.value))
    return out


def check_ast_gates(strict_signatures: bool, strict_open: bool, strict_boundaries: bool) -> tuple[bool, str]:
    problems = []

    for test_path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or):
                if any(isinstance(v, ast.Constant) and v.value is True for v in node.test.values):
                    problems.append(f"{test_path.relative_to(REPO_ROOT)}:{node.lineno} 含永真断言 `or True`")

    if strict_signatures:
        base_tree = ast.parse((REPO_ROOT / "mobench" / "platform" / "base.py").read_text(encoding="utf-8"))
        base_sigs = {}
        for cls in ast.walk(base_tree):
            if isinstance(cls, ast.ClassDef) and cls.name == "BasePlatform":
                for item in cls.body:
                    if isinstance(item, ast.FunctionDef) and item.name in ("preflight", "run", "control", "restore"):
                        base_sigs[item.name] = _signature_shape(item)
        for plat_file, cls_name in (("host.py", "HostPlatform"), ("adb.py", "AdbPlatform")):
            ptree = ast.parse((REPO_ROOT / "mobench" / "platform" / plat_file).read_text(encoding="utf-8"))
            for cls in ast.walk(ptree):
                if isinstance(cls, ast.ClassDef) and cls.name == cls_name:
                    for item in cls.body:
                        if isinstance(item, ast.FunctionDef) and item.name in base_sigs:
                            got = _signature_shape(item)
                            want = base_sigs[item.name]
                            if got != want:
                                problems.append(
                                    f"mobench/platform/{plat_file}:{item.lineno} {cls_name}.{item.name}{got} != BasePlatform.{item.name}{want}"
                                )

    if strict_open:
        for py in sorted((REPO_ROOT / "mobench").glob("**/*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for lineno in _has_bare_open(tree):
                problems.append(f"{py.relative_to(REPO_ROOT)}:{lineno} 存在未使用 with 的裸 open() 调用")

    if strict_boundaries:
        forbidden = ("llama-bench", "llm_bench", "MNNConvert", "convert_hf_to_gguf", "cache_type_k")
        for py in sorted((REPO_ROOT / "mobench").glob("**/*.py")):
            rel = py.relative_to(REPO_ROOT).as_posix()
            if rel.startswith("mobench/adapters/"):
                continue
            if rel in ("mobench/config.py", "mobench/cli.py"):
                continue
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for lineno, text in _extract_non_doc_strings(tree):
                for tok in forbidden:
                    if tok in text:
                        problems.append(f"{rel}:{lineno} 核心模块包含引擎硬编码 {tok!r}")

    if problems:
        return False, "\n      ".join(problems[:15])
    return True, (
        f"AST 守门通过 (signatures={strict_signatures}, open={strict_open}, boundaries={strict_boundaries})"
    )


def check_core_deps_isolation() -> tuple[bool, str]:
    code = (
        "import sys\n"
        "for m in ('scipy', 'fastmcp', 'markdown_it'):\n"
        "    sys.modules[m] = None\n"
        "from mobench.cli import main\n"
        "assert main(['--help']) == 0\n"
        "assert main(['--version']) == 0\n"
        "import mobench.run, mobench.analysis.stats, mobench.analysis.regression, "
        "mobench.analysis.power, mobench.report, mobench.validation\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip() or f"rc={proc.returncode}"
    return True, "无 scipy/fastmcp/markdown_it 下核心 CLI 与模块导入通过"


def check_ruff(require_ruff: bool) -> tuple[bool, str]:
    ruff_bin = shutil.which("ruff")
    if not ruff_bin:
        if require_ruff:
            return False, "ruff 未安装（传入了 --require-ruff）"
        return True, "skipped (ruff not installed locally)"
    proc = subprocess.run(
        [ruff_bin, "check", "mobench", "tests", "scripts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, (proc.stdout + proc.stderr).strip()
    return True, "ruff check 通过"


def check_pytest() -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--cov=mobench",
            "--cov-branch",
            "--cov-fail-under=80",
        ],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        return False, f"pytest 失败 (rc={proc.returncode})"
    return True, "pytest 全部通过且覆盖率 >= 80%"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="mobench 统一验证入口")
    ap.add_argument("--skip-pytest", action="store_true", help="跳过 pytest（用于 CI 轻量卫生检查 job）")
    ap.add_argument("--require-ruff", action="store_true", help="缺少 ruff 时直接判失败")
    ap.add_argument("--strict-signatures", action=argparse.BooleanOptionalAction, default=True,
                    help="校验 BasePlatform 与子类方法签名一致（默认开启）")
    ap.add_argument("--strict-open", action="store_true", help="校验 mobench/ 无裸 open() 调用")
    ap.add_argument("--strict-boundaries", action="store_true", help="校验核心模块无引擎硬编码")
    args = ap.parse_args(argv)

    ok = True
    ok &= _run_step("compileall", check_compileall)
    ok &= _run_step("version-sync", check_version_sync)
    ok &= _run_step("schema-and-yaml", check_schema_and_yaml)
    ok &= _run_step("repo-hygiene", check_repo_hygiene)
    ok &= _run_step(
        "ast-checks",
        lambda: check_ast_gates(args.strict_signatures, args.strict_open, args.strict_boundaries),
    )
    ok &= _run_step("core-deps-isolation", check_core_deps_isolation)
    ok &= _run_step("ruff", lambda: check_ruff(args.require_ruff))
    if not args.skip_pytest:
        ok &= _run_step("pytest", check_pytest)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
