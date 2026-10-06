#!/usr/bin/env python3
"""한국어 F5-TTS 체크포인트 받기·변환 — team-lucid/F5-TTS-ko (Apache-2.0, F5TTS_v1_Base 구조, 한글 자모 단위 어휘).
   models/f5-ko/ 에 model.safetensors(키 앞에 ema_model.) + vocab.txt(F5 형식)를 만든다. setup.sh 가 부른다.
   venv/bin/python scripts/f5_ko.py"""
import json, os, sys, urllib.request
import torch
from safetensors.torch import save_file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "models", "f5-ko"); os.makedirs(D, exist_ok=True)
if os.path.exists(os.path.join(D, "model.safetensors")) and os.path.exists(os.path.join(D, "vocab.txt")):
    sys.exit(0)
URL = "https://huggingface.co/team-lucid/F5-TTS-ko/resolve/main/"
for f in ("pytorch_model.bin", "vocab.json"):
    if not os.path.exists(os.path.join(D, f)):
        urllib.request.urlretrieve(URL + f, os.path.join(D, f))
v = json.load(open(os.path.join(D, "vocab.json"), encoding="utf-8"))
open(os.path.join(D, "vocab.txt"), "w", encoding="utf-8").write("\n".join(sorted(v, key=v.get)) + "\n")  # 줄 순서 = 토큰 번호
sd = torch.load(os.path.join(D, "pytorch_model.bin"), map_location="cpu", weights_only=False)
save_file({"ema_model." + k: t.contiguous() for k, t in sd.items()}, os.path.join(D, "model.safetensors"))  # F5 가 EMA 가중치로 읽는 형식
os.remove(os.path.join(D, "pytorch_model.bin"))
print("f5-ko ready")
