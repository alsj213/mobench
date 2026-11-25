#!/usr/bin/env python3
"""docs/*.md → 自包含 HTML 报告。

沿用 `docs/archive/compare-qwen3-0.6b-adb-overview.html` 的设计语言：
暖中性底色 + 引擎强调色 + 卡片/结论框/胶囊比值，light/dark 双主题，无外部依赖。

用法：
    python3 scripts/md2report.py docs/               # 转换 docs/ 下所有带 HTML 孪生的 md
    python3 scripts/md2report.py docs/ a.md b.md     # 只转换指定文件
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

# CommonMark 解析器：能正确处理列表项内缩进的围栏代码块
# （python-markdown 的 fenced_code 会把它们吞成行内 code）
_MD = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])


def _render(md_text: str) -> str:
    return _MD.render(md_text)


def _inline(text: str) -> str:
    """渲染单行 markdown，去掉包裹的 <p>。"""
    return _MD.render(text).strip().removeprefix("<p>").removesuffix("</p>")


# ── ```chart 围栏块 → 柱状图 ────────────────────────────────────────────────
# 语法（每个图一个围栏块）：
#   title:   卡片标题（可选）
#   series:  系列名，逗号分隔（可选；有则每行给多个值，前淡后实）
#   group:   指标分组名（可选；每组独立标尺，避免 prefill/decode 同尺误导）
#   <标签>:  <值>[, <值>...]
_ENGINE_VAR = (("llama", "--llama"), ("official", "--int4"),
               ("int4", "--int4"), ("q4", "--q4k"))


def _engine_var(label: str) -> str:
    """按标签里的引擎关键词取强调色；认不出就用主度量色。"""
    low = label.lower()
    for key, var in _ENGINE_VAR:
        if key in low:
            return f"var({var})"
    return "var(--q4k)"


def _render_chart(spec: str) -> str:
    title, series, groups = "", [], []
    for raw in spec.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, val = (s.strip() for s in line.split(":", 1))
        if key == "title":
            title = val
        elif key == "series":
            series = [s.strip() for s in val.split(",") if s.strip()]
        elif key == "group":
            groups.append((val, []))
        else:
            if not groups:
                groups.append(("", []))
            # 保留原始字面量（28.80 不写成 28.8），float 只用于算条长
            groups[-1][1].append((key, [(s.strip(), float(s.strip())) for s in val.split(",")]))

    rows_html, values = [], []
    for gname, rows in groups:
        if not rows:
            continue
        top = max(v for _, vs in rows for _, v in vs) or 1.0
        body = []
        for label, vs in rows:
            color = _engine_var(label)
            n = len(vs)
            bars = "".join(
                f'<div class="track"><div class="fill" style="width:{v / top * 100:.4g}%;'
                f'background:{color};opacity:{1.0 if n == 1 else 0.38 + 0.62 * (i / max(n - 1, 1)):.2f}">'
                f'</div></div>'
                for i, (_, v) in enumerate(vs))
            shown = " / ".join(raw for raw, _ in vs)
            body.append(f'<div class="chartrow"><div class="clab">{html.escape(label)}</div>'
                        f'<div class="bars">{bars}</div><div class="cval">{shown}</div></div>')
            values.extend(raw for raw, _ in vs)
        head = (f'<p class="cgtitle"><span>{html.escape(gname)}</span>'
                f'<span class="cmax">满尺 {top:g}</span></p>') if gname else ""
        rows_html.append(f'<div class="cgroup">{head}{"".join(body)}</div>')

    lgd = ""
    if series:
        # 色块用中性色：柱子是按引擎着色的，图例只表达"前淡后实"这一层信息
        sw = "".join(
            f'<span><i style="background:var(--muted);opacity:{0.38 + 0.62 * (i / max(len(series) - 1, 1)):.2f}"></i>'
            f'{html.escape(s)}</span>' for i, s in enumerate(series))
        lgd = f'<div class="lgd">{sw}</div>'

    head = f'<p class="ctitle">{html.escape(title)}</p>' if title else ""
    return (f'<div class="chart" data-values="{" ".join(values)}">'
            f'{head}{"".join(rows_html)}{lgd}</div>\n')


def _verify_charts(body: str) -> list[str]:
    """每个图上的数值都必须能在相邻表格里找到 —— 防止表格改了图没改的静默漂移。"""
    warns = []
    for m in re.finditer(r'<div class="chart" data-values="([^"]*)"', body):
        values = [v for v in html.unescape(m.group(1)).split() if v]
        if not values:
            continue
        before, after = body[:m.start()], body[m.end():]
        near = ""
        i = before.rfind("<table")
        if i != -1:
            near += before[i:]
        j = after.find("<table")
        if j != -1:
            near += after[j:after.find("</table>", j) + 8]
        cells = _TAGS.sub(" ", near)
        for v in values:
            if v not in cells:
                warns.append(f"图表数值 {v} 未在相邻表格中找到（图与表已漂移）")
    return warns


def _install_chart_fence() -> None:
    """把 ```chart 围栏块接到柱状图渲染器上。"""
    rules = _MD.renderer.rules
    default = rules.get("fence")

    def fence(tokens, idx, options, env):
        tok = tokens[idx]
        if tok.info.strip().lower() == "chart":
            return _render_chart(tok.content)
        if default is not None:
            return default(tokens, idx, options, env)
        return _MD.renderer.renderToken(tokens, idx, options)

    rules["fence"] = fence


_install_chart_fence()

# ── 设计令牌：直接取自归档总览页 ────────────────────────────────────────────
CSS = """
:root{
  --bg:#f6f4ef; --fg:#1c1917; --muted:#78716c; --surface:#ffffff; --surface2:#f0ede7;
  --line:#e7e3dc; --line2:#d6d0c4;
  --llama:#c2410c; --llama-ink:#9a3412;
  --q4k:#0f766e; --q4k-ink:#115e59;
  --int4:#6d28d9; --int4-ink:#5b21b6;
  --ok:#047857; --warn:#b45309; --bad:#b91c1c;
  --chipbg:#f4f1ea;
  --shadow:0 1px 2px rgba(28,25,23,.06);
  --radius:14px;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#15130f; --fg:#e9e3d8; --muted:#a8a29e; --surface:#201d16; --surface2:#2a261d;
    --line:#393426; --line2:#4a4333;
    --llama:#fb923c; --llama-ink:#fdba74;
    --q4k:#2dd4bf; --q4k-ink:#5eead4;
    --int4:#a78bfa; --int4-ink:#c4b5fd;
    --ok:#34d399; --warn:#fbbf24; --bad:#f87171;
    --chipbg:#29251b;
    --shadow:0 1px 2px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --bg:#15130f; --fg:#e9e3d8; --muted:#a8a29e; --surface:#201d16; --surface2:#2a261d;
  --line:#393426; --line2:#4a4333;
  --llama:#fb923c; --llama-ink:#fdba74;
  --q4k:#2dd4bf; --q4k-ink:#5eead4;
  --int4:#a78bfa; --int4-ink:#c4b5fd;
  --ok:#34d399; --warn:#fbbf24; --bad:#f87171;
  --chipbg:#29251b;
  --shadow:0 1px 2px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
  "PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1020px;margin:0 auto;padding:40px 24px 72px}

/* ── 页头 ── */
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);font-weight:600;margin:0 0 10px}
h1{font-size:31px;line-height:1.2;margin:0 0 12px;text-wrap:balance;letter-spacing:-.015em}
.lede{font-size:15.5px;color:var(--muted);max-width:74ch;margin:0 0 4px}
.lede b,.lede strong{color:var(--fg);font-weight:600}
.metagrid{margin:18px 0 6px;background:var(--surface);border:1px solid var(--line);
  border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.mrow{display:grid;grid-template-columns:88px 1fr;gap:14px;padding:9px 16px;
  border-bottom:1px solid var(--line);font-size:13px}
.mrow:last-child{border-bottom:0}
.mk{color:var(--muted);font-weight:600;white-space:nowrap}
.mv{color:var(--fg);line-height:1.6}
.mv code{font-size:12.4px}

hr.sep{border:0;border-top:1px solid var(--line);margin:34px 0}
section{margin:0}
h2{font-size:19px;margin:0 0 6px;letter-spacing:-.01em}
h3{font-size:15.5px;margin:26px 0 8px;color:var(--fg);letter-spacing:-.005em}
h3::before{content:"";display:inline-block;width:3px;height:13px;background:var(--line2);
  border-radius:2px;margin-right:9px;vertical-align:-1px}
h4{font-size:14px;margin:18px 0 6px;color:var(--muted)}
.sect-sub{color:var(--muted);font-size:13.5px;margin:0 0 16px;max-width:82ch}

p{margin:11px 0}
ul,ol{margin:11px 0;padding-left:22px}
li{margin:6px 0}
li::marker{color:var(--muted)}
strong{font-weight:650}
a{color:var(--int4-ink);text-decoration:none;border-bottom:1px solid transparent}
a:hover{border-bottom-color:currentColor}

code{background:var(--surface2);padding:1.5px 5px;border-radius:5px;
  font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  font-size:.86em}
pre{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:14px 16px;overflow-x:auto;box-shadow:var(--shadow)}
pre code{background:none;padding:0;font-size:12.5px;line-height:1.66;color:var(--muted)}

/* ── 结论框（markdown 引用块）── */
blockquote{margin:16px 0;background:var(--surface);border:1px solid var(--line);
  border-left:4px solid var(--int4);border-radius:var(--radius);
  padding:14px 18px;box-shadow:var(--shadow)}
blockquote.warn{border-left-color:var(--warn)}
blockquote p{margin:7px 0}
blockquote p:first-child{margin-top:0}
blockquote p:last-child{margin-bottom:0}
blockquote.warn{background:color-mix(in srgb, var(--warn) 6%, var(--surface))}

/* ── 柱状图（```chart 围栏块）── */
.chart{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  padding:18px 20px 14px;box-shadow:var(--shadow);margin:16px 0}
.ctitle{font-size:13px;font-weight:650;color:var(--fg);margin:0 0 14px}
.cgroup{margin:0 0 14px}
.cgroup:last-of-type{margin-bottom:2px}
.cgtitle{display:flex;justify-content:space-between;gap:12px;font-size:11.5px;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  margin:0 0 8px;padding-bottom:5px;border-bottom:1px dashed var(--line2)}
.cmax{font-weight:600;letter-spacing:0;text-transform:none;opacity:.75}
.chartrow{display:grid;grid-template-columns:172px 1fr 70px;gap:10px;align-items:center;margin:5px 0}
.clab{font-size:12.5px;color:var(--fg);text-align:right;line-height:1.3;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bars{display:flex;flex-direction:column;gap:3px;min-width:0}
.track{background:var(--surface2);border-radius:5px;height:15px;overflow:hidden}
.fill{height:100%;border-radius:5px;min-width:2px}
.cval{font-size:12px;color:var(--muted);text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap}
.lgd{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin-top:12px;
  padding-top:10px;border-top:1px dashed var(--line2)}
.lgd span{display:inline-flex;align-items:center;gap:6px}
.lgd i{width:10px;height:10px;border-radius:3px;display:inline-block}

/* ── 表格 ── */
.tw{overflow-x:auto;margin:16px 0;-webkit-overflow-scrolling:touch;border-radius:var(--radius)}
table{width:100%;border-collapse:collapse;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--radius);overflow:hidden;
  font-variant-numeric:tabular-nums;font-size:13.4px}
th,td{padding:9px 13px;text-align:right;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600;background:var(--surface2);white-space:nowrap}
th.l,td.l{text-align:left}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:color-mix(in srgb, var(--surface2) 55%, transparent)}
td code{white-space:nowrap}

/* 比值胶囊：中性为默认；仅"变化/提升"类列带方向色 */
.ratio{display:inline-block;font-size:12px;font-weight:700;padding:1px 8px;
  border-radius:99px;font-variant-numeric:tabular-nums;
  background:var(--chipbg);color:var(--fg);border:1px solid var(--line)}
.ratio.up{background:color-mix(in srgb, var(--ok) 14%, transparent);
  color:var(--ok);border-color:transparent}
.ratio.tie{background:var(--surface2);color:var(--muted);border-color:transparent}
.ratio.down{background:color-mix(in srgb, var(--bad) 13%, transparent);
  color:var(--bad);border-color:transparent}

.note{font-size:12px;color:var(--muted);margin:10px 0}
footer{margin-top:44px;color:var(--muted);font-size:12px;
  border-top:1px solid var(--line);padding-top:16px;line-height:1.75}

@media (max-width:820px){
  .wrap{padding:28px 16px 56px}
  h1{font-size:25px}
  .mrow{grid-template-columns:1fr;gap:2px}
  .mk{font-size:11.5px}
}
"""

# 表头/表体单元格里"看起来是纯数值"的判定：只含数字与数学/单位标点
_NUM = re.compile(r"^[\d\s.,±×%°\-+−~<>/()\[\]=]*\d[\d\s.,±×%°\-+−~<>/()\[\]=]*$")
_RATIO_CELL = re.compile(r"^\s*(?:<strong>)?[×x]\s*([\d.]+)(?:</strong>)?\s*$"
                         r"|^\s*(?:<strong>)?([\d.]+)\s*×(?:</strong>)?\s*$")
_TAGS = re.compile(r"<[^>]+>")


def _text(cell_html: str) -> str:
    return html.unescape(_TAGS.sub("", cell_html)).strip()


def _split_row(row_html: str) -> list[str]:
    return re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", row_html, re.S)


def _align_tables(body: str) -> str:
    """按列判定数值/文本，数值列右对齐、文本列左对齐，并把纯比值单元格做成胶囊。"""
    def numeric_cols(rows: list[list[str]], width: int) -> list[bool]:
        """某一列若 60% 以上（且至少 2 个）单元格是纯数值，则该列右对齐。"""
        cols = []
        for i in range(width):
            cells = [c for c in (_text(r[i]) for r in rows if i < len(r)) if c]
            hits = sum(bool(_NUM.match(c)) for c in cells)
            cols.append(len(cells) >= 2 and hits >= len(cells) * 0.6)
        return cols

    def fix_table(m: re.Match) -> str:
        tbl = m.group(0)
        head = re.search(r"<thead\b[^>]*>(.*?)</thead>", tbl, re.S)
        body = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", tbl, re.S)
        head_rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", head.group(1), re.S) if head else []
        body_rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", body.group(1), re.S) if body else []
        width = max([len(_split_row(r)) for r in body_rows + head_rows], default=0)
        cols = numeric_cols([_split_row(r) for r in body_rows], width)
        # 只有表头明确表达"变化/提升"的列，比值才有好坏方向；其余列（如"比值""倍率"）
        # 是描述性数字，套用绿/红会误导，一律用中性胶囊。
        headers = _split_row(head_rows[0]) if head_rows else []
        evaluative = [bool(re.search(r"提升|变化|加速|收益|降幅", _text(h))) for h in headers]

        def mark(row: str, is_head: bool) -> str:
            i = 0

            def repl(m2: re.Match) -> str:
                nonlocal i
                tag, attrs, inner = m2.group(1), m2.group(2), m2.group(3)
                idx = i
                i += 1
                cls = "n" if (idx < len(cols) and cols[idx]) and not is_head else "l"
                extra = f' class="{cls}"' if "class=" not in attrs else attrs
                if not is_head:
                    ratio = _RATIO_CELL.match(inner)
                    if ratio:
                        num = ratio.group(1) or ratio.group(2)
                        v = float(num)
                        kind = ""
                        if idx < len(evaluative) and evaluative[idx]:
                            kind = " up" if v >= 1.05 else (" down" if v < 0.95 else " tie")
                        inner = f'<span class="ratio{kind}">×{num}</span>'
                return f"<{tag}{extra}>{inner}</{tag}>"

            return re.sub(r"<(t[hd])\b([^>]*)>(.*?)</\1>", repl, row, flags=re.S)

        if head:
            tbl = tbl.replace(head.group(0),
                              "<thead>" + "".join(f"<tr>{mark(r, True)}</tr>" for r in head_rows) + "</thead>")
        if body:
            tbl = tbl.replace(body.group(0),
                              "<tbody>" + "".join(f"<tr>{mark(r, False)}</tr>" for r in body_rows) + "</tbody>")
        return f'<div class="tw">{tbl}</div>'

    return re.sub(r"<table\b.*?</table>", fix_table, body, flags=re.S)


def _wrap_sections(body: str) -> str:
    """把每个 h2 及其后续兄弟节点包进 <section>，与归档页结构一致。"""
    parts = re.split(r"(?=<h2\b)", body)
    out = []
    for part in parts:
        if part.lstrip().startswith("<h2"):
            out.append(f"<section>{part.rstrip()}</section>")
        else:
            out.append(part)
    return "\n".join(out)


def _mark_subtitles(body: str) -> str:
    """h2 之后紧跟的短段落标为 .sect-sub（归档页的小字副标题）。"""
    def repl(m: re.Match) -> str:
        para, inner = m.group(1), m.group(2)
        plain = _text(inner)
        if len(plain) <= 130 and "\n" not in inner:
            return f'{para}<p class="sect-sub">{inner}</p>'
        return m.group(0)

    return re.sub(r"(</h2>)\s*<p>(.*?)</p>", repl, body, flags=re.S)


def _mark_blockquotes(body: str) -> str:
    """以 ⚠ 开头的引用块是"注意/前提"警示，换警示色边条。"""
    return re.sub(
        r"<blockquote>(.*?)</blockquote>",
        lambda m: (f'<blockquote class="warn">{m.group(1)}</blockquote>'
                   if _text(m.group(1)).startswith("⚠") else m.group(0)),
        body, flags=re.S)


def _header(md_text: str) -> tuple[str, str, str]:
    """解析 H1 与首个 --- 之间的元信息块 → (标题, 副标题, metagrid HTML)。"""
    lines = md_text.splitlines()
    title = next((l.lstrip("# ").strip() for l in lines if l.startswith("# ")), "")
    start = next(i for i, l in enumerate(lines) if l.startswith("# "))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "---"), len(lines))

    fields: list[tuple[str, str]] = []
    for raw in lines[start + 1:end]:
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\*\*(.+?)\*\*[:：]\s*(.+)$", line) or re.match(r"^-\s*(.+?)[:：]\s*(.+)$", line)
        if m:
            fields.append((m.group(1).strip(), m.group(2).strip()))

    lede = _inline(next((v for k, v in fields if k in ("主题", "目的")), ""))
    # 已作为副标题呈现的字段不再重复列进元信息表
    grid = "".join(
        f'<div class="mrow"><span class="mk">{html.escape(k)}</span>'
        f'<span class="mv">{_inline(v)}</span></div>'
        for k, v in fields if k not in ("主题", "目的"))
    grid = f'<div class="metagrid">{grid}</div>' if grid else ""
    return title, lede, grid


def _eyebrow(src: Path, md_text: str) -> str:
    kind = "对比报告" if src.stem.startswith("compare-") else "分析笔记"
    date = next((m.group(1).strip() for m in re.finditer(r"^[-*]?\s*(?:\*\*)?日期(?:\*\*)?[:：]\s*(.+)$",
                                                        md_text, re.M)), "")
    return " · ".join(x for x in (kind, date) if x)


TMPL = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">

  <p class="eyebrow">{eyebrow}</p>
  <h1>{title}</h1>
{lede}{grid}

  <hr class="sep">
{body}

  <footer>
    由 <code>scripts/md2report.py</code> 从 <code>{src}</code> 生成 · 全部数值可追溯至 <code>results/</code> 下的 canonical records 与原始日志。
  </footer>
</div>
</body>
</html>
"""


def convert(src: Path, out_dir: Path) -> tuple[Path, int, list[str]]:
    md_text = src.read_text(encoding="utf-8")
    title, lede, grid = _header(md_text)

    # 去掉已经被页头消费的元信息块，只留正文
    lines = md_text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("# "))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == "---"), len(lines))
    body_md = "\n".join(lines[end + 1:])

    body = _render(body_md)
    body = _align_tables(body)
    body = _mark_subtitles(body)
    body = _mark_blockquotes(body)
    body = re.sub(r"<hr\s*/?>", '<hr class="sep">', body)
    body = _wrap_sections(body)

    doc = TMPL.format(
        title=html.escape(title), css=CSS, body=body,
        eyebrow=html.escape(_eyebrow(src, md_text)), src=html.escape(src.name),
        lede=f'  <p class="lede">{lede}</p>\n' if lede else "",
        grid=f"  {grid}\n" if grid else "")

    warnings = _verify_charts(body)
    dst = out_dir / (src.stem + ".html")
    dst.write_text(doc, encoding="utf-8")
    return dst, len(doc), warnings


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    names = sys.argv[2:]
    if not names:
        # 默认：docs/ 下已有 HTML 孪生的 md（即"报告类"文档）
        names = [p.name for p in sorted(out.glob("*.md")) if p.with_suffix(".html").exists()]
    bad = 0
    for name in names:
        path = Path(name)
        dst, size, warns = convert(path if path.is_absolute() else out / path.name, out)
        print(f"  {dst.name:46s} {size / 1024:6.1f} KB")
        for w in warns:
            bad += 1
            print(f"    ✗ {path.name}: {w}", file=sys.stderr)
    sys.exit(1 if bad else 0)
