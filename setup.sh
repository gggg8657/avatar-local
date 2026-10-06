#!/usr/bin/env bash
# avatar-local — 원샷 설치·실행 (macOS / Linux). bash setup.sh | bash setup.sh stop
# env: PORT(8777) DEVICE(cpu|cuda) TTS_BASE_URL(선택, 예 http://localhost:8771/v1)
set -euo pipefail
cd "$(dirname "$0")"
ST_COMMIT=cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8
PORT="${PORT:-8777}"
ok(){ printf '  ✔ %s\n' "$*"; }; die(){ printf '  ✘ %s\n' "$*" >&2; exit 1; }
if [ "${1:-}" = stop ]; then [ -f .studio.pid ] && kill "$(cat .studio.pid)" 2>/dev/null; rm -f .studio.pid  # 가상 캐릭터 그림·영상 작업 프로세스
  [ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null && rm -f .server.pid && ok "서버 종료" || echo "  실행 중 아님"; exit 0; fi
command -v ffmpeg >/dev/null || die "ffmpeg 필요"
PY=""; for c in python3.11 python3.10 python3.12 python3; do command -v $c >/dev/null && $c -c 'import sys;sys.exit(0 if (3,10)<=sys.version_info<(3,13) else 1)' 2>/dev/null && { PY=$c; break; }; done
[ -n "$PY" ] || die "Python 3.10~3.12 필요"; ok "$($PY --version)"
if [ ! -x venv/bin/python ]; then command -v uv >/dev/null && uv venv --python "$(command -v $PY)" venv -q || $PY -m venv venv; fi
# GPU 드라이버에 맞는 torch — PyPI 기본 torch 는 CUDA 13 빌드라 드라이버가 CUDA 12.x 면 GPU 를 못 쓴다(드라이버 570 = 12.8)
TORCH_CU=""; TORCH_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  _cu=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  if [ -n "$_cu" ] && [ "$_cu" -lt 130 ]; then TORCH_CU=cu121; [ "$_cu" -ge 124 ] && TORCH_CU=cu124; [ "$_cu" -ge 126 ] && TORCH_CU=cu126
    if command -v uv >/dev/null; then TORCH_ARGS=(--torch-backend "$TORCH_CU"); else TORCH_ARGS=(--extra-index-url "https://download.pytorch.org/whl/$TORCH_CU"); fi; fi
fi
torch_gpu_fix() {  # 이미 깔린 torch 가 GPU 를 못 잡으면 드라이버에 맞는 빌드로 다시 (폐쇄망 wheels 번들은 건드리지 않음)
  local py=$1; shift; [ -n "$TORCH_CU" ] && [ ! -d wheels ] || return 0
  "$py" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && return 0
  echo "  · torch 를 GPU 드라이버에 맞는 $TORCH_CU 빌드로 다시 설치"
  if command -v uv >/dev/null; then uv pip install -q -p "$py" --reinstall-package torch --torch-backend "$TORCH_CU" "$@"
  else "$py" -m pip install -q --force-reinstall --no-deps --extra-index-url "https://download.pytorch.org/whl/$TORCH_CU" "$@"; fi
}
PIP_ARGS=(); [ -d wheels ] && PIP_ARGS=(--no-index --find-links wheels) || PIP_ARGS=("${TORCH_ARGS[@]}")
venv/bin/python -c "import torch, face_alignment, f5_tts" 2>/dev/null || {
  echo "  · 의존성 설치 (torch·F5-TTS 포함, 수 분)"
  if command -v uv >/dev/null; then VIRTUAL_ENV=$PWD/venv uv pip install -q "${PIP_ARGS[@]}" torch torchaudio torchvision -r requirements.txt f5-tts
  else venv/bin/pip install -q "${PIP_ARGS[@]}" torch torchaudio torchvision -r requirements.txt f5-tts; fi
}; torch_gpu_fix venv/bin/python torch torchaudio torchvision; ok "venv 준비"
# 한국어 목소리 복제: team-lucid/F5-TTS-ko (기본 F5 는 영·중 학습이라 한국어가 알아들을 수 없게 나온다)
if [ ! -d wheels ] && [ -z "${F5_CKPT:-}" ] && [ ! -f models/f5-ko/model.safetensors ]; then echo "  · 한국어 F5-TTS 체크포인트 (1.3GB)"; venv/bin/python scripts/f5_ko.py && ok "한국어 목소리 복제 모델"; fi
# 목소리 복제(F5-TTS)는 torchaudio→torchcodec 으로 오디오를 읽는데, FFmpeg *공유* 라이브러리가 필요하다(정적 ffmpeg 로는 안 됨)
FFLIB="${FFMPEG_LIB_DIR:-$HOME/.local/ffmpeg-shared/lib}"
if [ ! -d wheels ] && ! LD_LIBRARY_PATH="$FFLIB:${LD_LIBRARY_PATH:-}" venv/bin/python -c "import torchcodec.decoders" 2>/dev/null; then
  CONDA=$(command -v mamba || command -v conda || ls "$HOME"/miniforge3/bin/mamba "$HOME"/miniconda3/bin/conda 2>/dev/null | head -1)
  if [ -n "$CONDA" ]; then echo "  · FFmpeg 공유 라이브러리 설치 (conda, 목소리 복제용)"; "$CONDA" create -y -q -p "$HOME/.local/ffmpeg-shared" -c conda-forge "ffmpeg=7" >/dev/null && ok "FFmpeg 공유 라이브러리"
  else echo "  ! 목소리 복제엔 FFmpeg 공유 라이브러리(libavcodec 등)가 필요합니다: apt install ffmpeg 또는 FFMPEG_LIB_DIR 지정 (샘플 없이 TTS 목소리는 동작)"; fi
fi
mkdir -p shim/gfpgan; [ -f shim/sitecustomize.py ] || printf 'import numpy as _np\n_np.float, _np.int = float, int\n' > shim/sitecustomize.py; [ -f shim/gfpgan/__init__.py ] || printf 'class GFPGANer:  # ponytail: 얼굴 보정(gfpgan)은 안 씀 — basicsr 가 최신 torchvision 과 깨져서 스텁\n    pass\n' > shim/gfpgan/__init__.py
if [ ! -f vendor/SadTalker/inference.py ]; then
  command -v git >/dev/null || die "git 필요 (또는 pack.sh 번들)"
  git clone -q https://github.com/OpenTalker/SadTalker.git vendor/SadTalker && git -C vendor/SadTalker checkout -q $ST_COMMIT 2>/dev/null || true
fi; venv/bin/python shim/patch_sadtalker.py vendor/SadTalker >/dev/null; ok "SadTalker"
mkdir -p weights/checkpoints
for f in mapping_00109-model.pth.tar mapping_00229-model.pth.tar SadTalker_V0.0.2_256.safetensors; do
  [ -s weights/checkpoints/$f ] || { echo "  · $f 다운로드"; curl -fsSL -o weights/checkpoints/$f https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/$f || die "다운로드 실패 — 폐쇄망이면 pack.sh 번들"; }
done; ok "가중치 $(du -sh weights | cut -f1) (face_alignment·facexlib 보조 모델은 첫 실행 때 자동 다운로드, 번들에는 포함)"
env -u WORKSPACE venv/bin/python selftest.py >/dev/null && ok "selftest 통과" || die "selftest 실패"
[ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null || true
PORT=$PORT nohup python3 app.py > server.log 2>&1 & echo $! > .server.pid
for _ in $(seq 1 30); do curl -fsS "http://localhost:$PORT/api/status" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "http://localhost:$PORT/api/status" >/dev/null || { cat server.log; die "서버 기동 실패"; }
ok "http://localhost:$PORT  (종료: bash setup.sh stop)"
case "$(uname -s)" in Darwin*) open "http://localhost:$PORT";; Linux*) command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT" >/dev/null 2>&1 || true;; esac
