#!/usr/bin/env python3
"""가상 캐릭터 스튜디오 작업 프로세스 — app.py 가 필요할 때 venv 파이썬으로 띄우고, 127.0.0.1 에서만 받는다.
큰 모델은 한 번에 하나만 GPU 에 올려 둔다(그림 Qwen-Image 약 57GB ↔ 영상 Wan2.2 TI2V-5B 약 34GB).
응답은 줄마다 JSON 하나(NDJSON) — 진행률을 흘리고 마지막 줄이 {"done": …} 또는 {"error": …}.

  POST /imagine {prompt, negative, width, height, n, seed, steps, out_dir}   → out_dir/cand_<i>.png
  POST /video   {image, prompt, negative, width, height, frames, steps, out} → out(mp4, 24fps)
  POST /mouth   {image}                                                      → {x, y, w} (그림 크기 대비 비율)

env: STUDIO_PORT(8787) STUDIO_DEVICE(기본: GPU 가 둘 이상이면 cuda:1 — 0번은 Ollama·TTS 가 씀)
     IMAGE_MODEL(Qwen/Qwen-Image) VIDEO_MODEL(Wan-AI/Wan2.2-TI2V-5B-Diffusers)"""
import gc, json, os, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

PORT = int(os.environ.get("STUDIO_PORT", "8787"))
DEV = os.environ.get("STUDIO_DEVICE") or ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "Qwen/Qwen-Image")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
LOCK = threading.Lock()
_cur = {"name": None, "pipe": None}


def pipe(name):
    """name 의 파이프라인을 올린다. 다른 게 올라 있으면 먼저 내린다(GPU 한 장에 둘 다는 안 들어감)."""
    if _cur["name"] == name:
        return _cur["pipe"]
    _cur["pipe"] = None; _cur["name"] = None; gc.collect(); torch.cuda.empty_cache()
    if name == "image":
        from diffusers import QwenImagePipeline
        p = QwenImagePipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.bfloat16).to(DEV)
    elif name == "video":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        vae = AutoencoderKLWan.from_pretrained(VIDEO_MODEL, subfolder="vae", torch_dtype=torch.float32)
        p = WanImageToVideoPipeline.from_pretrained(VIDEO_MODEL, vae=vae, torch_dtype=torch.bfloat16).to(DEV)
    elif name == "face":
        import face_alignment
        p = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=DEV)
    else:
        raise ValueError(name)
    _cur.update(name=name, pipe=p)
    return p


def fit(im, w, h):
    """가운데 기준으로 잘라 w×h 비율에 맞춘 뒤 크기 조정"""
    r = w / h
    W, H = im.size
    if W / H > r: nw = int(H * r); im = im.crop(((W - nw) // 2, 0, (W - nw) // 2 + nw, H))
    else: nh = int(W / r); im = im.crop((0, (H - nh) // 2, W, (H - nh) // 2 + nh))
    return im.resize((w, h), Image.LANCZOS)


def imagine(q, emit):
    emit({"stage": "load", "msg": "그림 모델 준비 (처음엔 1~2분)"})
    p = pipe("image")
    os.makedirs(q["out_dir"], exist_ok=True)
    paths = []
    for i in range(int(q.get("n", 4))):
        g = torch.Generator(DEV).manual_seed(int(q.get("seed", 0)) + i)
        im = p(prompt=q["prompt"], negative_prompt=q.get("negative") or " ", width=int(q["width"]), height=int(q["height"]),
               num_inference_steps=int(q.get("steps", 30)), true_cfg_scale=float(q.get("cfg", 4.0)), generator=g).images[0]
        path = os.path.join(q["out_dir"], f"cand_{i}.png"); im.save(path); paths.append(path)
        emit({"image": i, "path": path})
    return {"paths": paths}


def video(q, emit):
    from diffusers.utils import export_to_video
    emit({"stage": "load", "msg": "영상 모델 준비 (처음엔 1~2분)"})
    p = pipe("video")
    w, h, steps = int(q["width"]), int(q["height"]), int(q.get("steps", 40))
    img = fit(Image.open(q["image"]).convert("RGB"), w, h)

    def cb(pp, i, t, kw):
        emit({"step": i + 1, "total": steps}); return kw
    frames = p(image=img, prompt=q["prompt"], negative_prompt=q.get("negative") or "", width=w, height=h, num_frames=int(q.get("frames", 121)),
               guidance_scale=float(q.get("cfg", 5.0)), num_inference_steps=steps, generator=torch.Generator(DEV).manual_seed(int(q.get("seed", 0))),
               callback_on_step_end=cb).frames[0]
    export_to_video(frames, q["out"], fps=24)
    return {"path": q["out"]}


def mouth(q, emit):
    import numpy as np
    im = np.array(Image.open(q["image"]).convert("RGB")); H, W = im.shape[:2]
    lm = pipe("face").get_landmarks(im)
    if not lm:
        return {"mouth": None}
    m = lm[0][48:60]  # 바깥 입술
    return {"mouth": {"x": round(float(m[:, 0].mean() / W), 4), "y": round(float(m[:, 1].mean() / H), 4), "w": round(float((m[:, 0].max() - m[:, 0].min()) / W), 4)}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        b = json.dumps({"ok": True, "device": DEV, "loaded": _cur["name"]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        fn = {"/imagine": imagine, "/video": video, "/mouth": mouth}.get(self.path)
        self.send_response(200 if fn else 404); self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()

        def emit(ev):
            self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode()); self.wfile.flush()
        if not fn:
            return emit({"error": "not found"})
        with LOCK:  # GPU 작업은 한 번에 하나
            try:
                emit({"done": fn(q, emit)})
            except Exception as e:
                traceback.print_exc()
                emit({"error": f"{type(e).__name__}: {e}"[:600]})


if __name__ == "__main__":
    print(f"studio worker → 127.0.0.1:{PORT} device={DEV}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
