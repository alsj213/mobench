#!/bin/bash
# WSL Linux → Windows adb.exe 路径映射适配层。
#
# mobench 在 WSL Linux 侧用 Linux 路径做 os.path.isfile 校验，再把这些路径
# 传给 adb 的 push/shell 参数。若使用 Windows 侧的 adb.exe，它无法解析 Linux
# 路径。本 wrapper 在中间把「本地确实存在的 Linux 路径」映射成 Windows 路径：
#   /mnt/<盘>/...  ->  <盘>:\...                    (Windows 盘符，直连最稳)
#   /...           ->  \\wsl.localhost\<distro>\... (WSL 虚拟盘，经 9P/UNC)
# 其余参数（设备序列号、远程 /data/... 路径、subcommand 名等）原样透传。
#
# 用法：
#   1. 设置环境变量 ADB_EXE（指向 Windows 侧 adb.exe 的 WSL 路径），
#      以及可选的 WSL_DISTRO（默认取 WSL_DISTRO_NAME 或 Ubuntu-22.04）
#   2. 在 mobench.yml 中将 platform.adb 指向本脚本。

set -euo pipefail

ADB_EXE="${ADB_EXE:-$(command -v adb.exe 2>/dev/null || true)}"
if [[ -z "$ADB_EXE" || ! -x "$ADB_EXE" ]]; then
    echo "adb-unc-wrapper.sh: 请通过环境变量 ADB_EXE 指定可执行的 Windows adb.exe 路径" >&2
    exit 1
fi

WSL_DISTRO="${WSL_DISTRO:-${WSL_DISTRO_NAME:-Ubuntu-22.04}}"
UNC_PREFIX="\\\\wsl.localhost\\${WSL_DISTRO}\\"

args=()
for a in "$@"; do
    if [[ "$a" == /* ]] && [ -e "$a" ]; then
        # 优先识别 /mnt/<drv>/... 形式 -> <drv>:\... (Windows 原生盘，避免 9P 开销)
        if [[ "$a" =~ ^/mnt/([a-zA-Z])/(.+)$ ]]; then
            drv="${BASH_REMATCH[1],,}"
            rest="${BASH_REMATCH[2]//\//\\}"
            args+=("${drv}:\\${rest}")
        else
            rel="${a#/}"
            rel="${rel//\//\\}"
            args+=("${UNC_PREFIX}${rel}")
        fi
    else
        args+=("$a")
    fi
done

exec "$ADB_EXE" "${args[@]}"
