#!/usr/bin/env bash
# 폐쇄망 번들 (linux-x64, py3.11): wheels + SadTalker 소스 + 가중치 + 보조 모델 캐시.  ./pack.sh [cpu|cu124]
set -euo pipefail; cd "$(dirname "$0")"
V="${1:-cpu}"; S=dist-offline/avatar-local-linux-x64-$V; rm -rf "$S"; mkdir -p "$S/wheels"
IDX=(); [ "$V" = cpu ] && IDX=(--extra-index-url https://download.pytorch.org/whl/cpu) || IDX=(--extra-index-url https://download.pytorch.org/whl/$V)
venv/bin/pip download -q --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all: -d "$S/wheels" "${IDX[@]}" torch torchaudio torchvision -r requirements.txt f5-tts
cp -r app.py gpu_pick.py studio_worker.py ui.html selftest.py setup.sh requirements.txt README.md NOTICE LICENSE shim weights "$S/"
git clone -q --depth 1 https://github.com/OpenTalker/SadTalker.git "$S/vendor/SadTalker"; rm -rf "$S/vendor/SadTalker/.git"
# 첫 실행 때 받는 보조 모델: face_alignment(torch hub 캐시), facexlib(site-packages/facexlib/weights), F5-TTS(HF 캐시)
mkdir -p "$S/cache"; cp -r ~/.cache/torch/hub/checkpoints "$S/cache/torch_hub" 2>/dev/null || true
cp -r venv/lib/python3.11/site-packages/facexlib/weights "$S/cache/facexlib" 2>/dev/null || true
cp -r ~/.cache/huggingface/hub/models--SWivid--F5-TTS "$S/cache/" 2>/dev/null || true
cat > "$S/INSTALL.md" <<'INS'
1. python3.11 -m venv venv && venv/bin/pip install --no-index --find-links wheels torch torchaudio torchvision -r requirements.txt f5-tts
2. 보조 모델: cp -r cache/torch_hub/* ~/.cache/torch/hub/checkpoints/ ; cp -r cache/facexlib/* venv/lib/python3.11/site-packages/facexlib/weights/ ; cp -r cache/models--SWivid--F5-TTS ~/.cache/huggingface/hub/
3. HF_HUB_OFFLINE=1 DEVICE=cuda PORT=8777 bash setup.sh
INS
tar -C dist-offline -czf "$S.tar.gz" "$(basename "$S")"; ls -lh "$S.tar.gz"
