#!/usr/bin/env bash
# npu_lora 패키지 소스를 lmd-top 의 assets/npu-lora/ 로 복사한다.
# lmd-top 은 이 파일들을 include_str! 로 바이너리에 넣고, `--plan train` 이 만드는 Job 의
# ConfigMap 으로 내보낸다 — lmd-top 바이너리 하나로 학습 Job 을 만들 수 있게 하려는 것이다.
#   usage: scripts/vendor-into-lmd-top.sh [LMD_TOP_DIR]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dst="${1:-$here/../lmd-top}/assets/npu-lora"
mkdir -p "$dst"
rm -f "$dst"/*.py
cp "$here"/npu_lora/*.py "$dst"/
ver=$(python3 -c "import re;print(re.search(r'__version__ = \"([^\"]+)\"', open('$here/npu_lora/__init__.py').read()).group(1))")
src_rev=$(git -C "$here" rev-parse --short HEAD 2>/dev/null || echo "uncommitted")
printf '%s\n' "$ver" > "$dst/VERSION"
printf 'vendored from npu-lora %s (%s) by scripts/vendor-into-lmd-top.sh — edit upstream, not here\n' "$ver" "$src_rev" > "$dst/README"
ls "$dst"
