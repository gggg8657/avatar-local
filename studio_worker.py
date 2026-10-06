#!/usr/bin/env python3
"""가상 캐릭터 스튜디오 작업 프로세스 — app.py 가 필요할 때 venv 파이썬으로 띄우고, 127.0.0.1 에서만 받는다.
큰 모델은 한 번에 하나만 GPU 에 올려 둔다(그림 Qwen-Image-2512 약 58GB ↔ 영상 Wan2.2 I2V-A14B — 14B 전문가 둘은 CPU 에 두고 차례로 GPU 에).
응답은 줄마다 JSON 하나(NDJSON) — 진행률을 흘리고 마지막 줄이 {"done": …} 또는 {"error": …}.

  POST /imagine {prompt, negative, width, height, n, seed, steps, out_dir}   → out_dir/cand_<i>.png
  POST /video   {image, prompt, negative, width, height, frames, steps, cfg, fps, out} → out(mp4, 16fps)
  POST /mouth   {image}                                                      → {x, y, w} (그림 크기 대비 비율)

GPU 는 고정하지 않는다: 모델을 올릴 때마다 그 순간 여유 메모리가 가장 큰 GPU 를 고르고(gpu_pick.py),
GPU_IDLE_UNLOAD_S(기본 600초) 동안 안 쓰면 내려서 VRAM 을 돌려준다.
env: STUDIO_PORT(8787) STUDIO_DEVICE(비우면 자동 선택, cuda:1 처럼 주면 고정) GPU_POOL GPU_IDLE_UNLOAD_S
     IMAGE_MODEL(Qwen/Qwen-Image-2512) VIDEO_MODEL(Wan-AI/Wan2.2-I2V-A14B-Diffusers)"""
import json, os, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gpu_pick import IDLE_S, free_cuda, gpus, label, pick, release, torch_device  # torch 보다 먼저 (CUDA_DEVICE_ORDER)
import torch
from PIL import Image

PORT = int(os.environ.get("STUDIO_PORT", "8787"))
DEV = os.environ.get("STUDIO_DEVICE", "")  # 비우면 모델 올릴 때마다 자동
NEED_MB = {"image": 60000, "video": 48000, "face": 2000}  # 모델별로 GPU 에 필요한 여유(MiB, 대략)
NEED_MB.update({k.strip(): int(v) for k, v in (x.split("=") for x in os.environ.get("STUDIO_NEED_MB", "").split(",") if "=" in x)})  # 예: STUDIO_NEED_MB=image=70000,video=90000
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "Qwen/Qwen-Image-2512")
VIDEO_MODEL = os.environ.get("VIDEO_MODEL", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
LOCK = threading.Lock()
_cur = {"name": None, "pipe": None, "dev": "cpu", "where": "아직 안 올림", "last": time.time(), "g": None}


def unload():
    release(_cur["g"])  # 영상 모델이 잡아 둔 GPU 예약도 푼다
    _cur.update(name=None, pipe=None, where="아직 안 올림", g=None); free_cuda()


def device_for(name):
    """지금 여유 메모리가 가장 큰 GPU → (torch 장치, 표시용 글, 예약 g). 큰 모델이 들어갈 GPU 가 없으면 에러.
    영상은 CPU 오프로드라 돌 때만 GPU 메모리를 잡으므로 모델을 내릴 때까지 예약을 둔다(unload 가 풂). 나머지는 올린 뒤 바로 푼다."""
    if DEV:
        return DEV, DEV, None
    if not torch.cuda.is_available():
        return "cpu", "CPU", None
    g = pick(NEED_MB[name], hold_s=86400 if name == "video" else None)
    if not g:
        best = max((x["free"] for x in gpus()), default=0)
        raise RuntimeError(f"GPU 여유 메모리 부족 — 약 {NEED_MB[name] // 1024}GB 필요, 가장 넉넉한 GPU 가 {best // 1024}GB. 잠시 뒤 다시 해 보세요")
    return torch_device(g), label(g), g


def pipe(name, emit=lambda ev: None):
    """name 의 파이프라인을 올린다. 다른 게 올라 있으면 먼저 내리고(GPU 한 장에 둘 다는 안 들어감), 그때 가장 한가한 GPU 를 새로 고른다."""
    _cur["last"] = time.time()
    if _cur["name"] == name:
        return _cur["pipe"]
    unload()
    dev, where, g = device_for(name)
    print(f"[studio] {name} 모델 {where} 에서 로드", flush=True); emit({"stage": "load", "msg": f"{where} 에 모델 올리는 중"})
    try:
        if name == "image":
            from diffusers import QwenImagePipeline
            p = QwenImagePipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.bfloat16).to(dev)
        elif name == "video":
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            vae = AutoencoderKLWan.from_pretrained(VIDEO_MODEL, subfolder="vae", torch_dtype=torch.float32)
            p = WanImageToVideoPipeline.from_pretrained(VIDEO_MODEL, vae=vae, torch_dtype=torch.bfloat16)
            p.enable_model_cpu_offload(device=dev)  # 전문가 둘(고노이즈·저노이즈, 각 14B)을 한 GPU 에 다 올리면 70GB+ → 쓰는 차례에만 GPU 로, 화질은 같음
        elif name == "face":
            import face_alignment
            p = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=dev)
        else:
            raise ValueError(name)
    except BaseException:
        release(g); raise
    if name != "video": release(g); g = None  # 가중치가 GPU 에 올라갔으니 nvidia-smi 에 보임 → 예약은 그만
    _cur.update(name=name, pipe=p, dev=dev, where=where, last=time.time(), g=g)
    if dev.startswith("cuda"): torch.cuda.reset_peak_memory_stats(dev)  # VRAM 최고치(로그용)를 이 GPU 기준으로 다시 (가중치는 그대로 잡혀 있어 포함됨)
    return p


def reaper():
    """GPU_IDLE_UNLOAD_S 동안 안 쓰면 모델을 내려 VRAM 을 다른 도구에 돌려준다"""
    while IDLE_S > 0:
        time.sleep(30)
        if _cur["name"] and time.time() - _cur["last"] > IDLE_S and LOCK.acquire(blocking=False):
            try:
                name = _cur["name"]; unload(); print(f"[studio] {name} 모델 내림 — {int(IDLE_S)}초 동안 안 씀", flush=True)
            finally:
                LOCK.release()


def fit(im, w, h):
    """가운데 기준으로 잘라 w×h 비율에 맞춘 뒤 크기 조정"""
    r = w / h
    W, H = im.size
    if W / H > r: nw = int(H * r); im = im.crop(((W - nw) // 2, 0, (W - nw) // 2 + nw, H))
    else: nh = int(W / r); im = im.crop((0, (H - nh) // 2, W, (H - nh) // 2 + nh))
    return im.resize((w, h), Image.LANCZOS)


def imagine(q, emit):
    emit({"stage": "load", "msg": "그림 모델 준비 (처음엔 1~2분)"})
    p = pipe("image", emit)
    os.makedirs(q["out_dir"], exist_ok=True)
    paths = []
    for i in range(int(q.get("n", 4))):
        g = torch.Generator(_cur["dev"]).manual_seed(int(q.get("seed", 0)) + i)
        im = p(prompt=q["prompt"], negative_prompt=q.get("negative") or " ", width=int(q["width"]), height=int(q["height"]),
               num_inference_steps=int(q.get("steps", 50)), true_cfg_scale=float(q.get("cfg", 4.0)), generator=g).images[0]
        path = os.path.join(q["out_dir"], f"cand_{i}.png"); im.save(path); paths.append(path)
        emit({"image": i, "path": path})
    return {"paths": paths}


def video(q, emit):
    from diffusers.utils import export_to_video
    emit({"stage": "load", "msg": "영상 모델 준비 (처음엔 1~2분)"})
    p = pipe("video", emit)
    w, h, steps, cfg = int(q["width"]), int(q["height"]), int(q.get("steps", 40)), float(q.get("cfg", 3.5))
    img = fit(Image.open(q["image"]).convert("RGB"), w, h)

    def cb(pp, i, t, kw):
        emit({"step": i + 1, "total": steps}); return kw
    frames = p(image=img, prompt=q["prompt"], negative_prompt=q.get("negative") or "", width=w, height=h, num_frames=int(q.get("frames", 81)),
               guidance_scale=cfg, guidance_scale_2=cfg, num_inference_steps=steps, generator=torch.Generator(_cur["dev"]).manual_seed(int(q.get("seed", 0))),
               callback_on_step_end=cb).frames[0]
    export_to_video(frames, q["out"], fps=int(q.get("fps", 16)))
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
        b = json.dumps({"ok": True, "device": _cur["where"], "loaded": _cur["name"]}, ensure_ascii=False).encode()
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
                t0 = time.time()
                cuda = lambda: str(_cur["dev"]).startswith("cuda") and _cur["name"] is not None  # 이번 작업의 VRAM 최고치(로그용). 새로 올리면 pipe() 가 다시 잼
                if cuda(): torch.cuda.reset_peak_memory_stats(_cur["dev"])
                r = fn(q, emit)
                peak = torch.cuda.max_memory_allocated(_cur["dev"]) / 2 ** 30 if cuda() else 0
                print(f"[studio] {self.path} 끝 — {time.time() - t0:.0f}초, {_cur['where']} 최고 {peak:.1f}GB", flush=True)
                emit({"done": r}); _cur["last"] = time.time()
            except Exception as e:
                traceback.print_exc()
                emit({"error": f"{type(e).__name__}: {e}"[:600]})


if __name__ == "__main__":
    print(f"studio worker → 127.0.0.1:{PORT} device={DEV or '자동(모델 올릴 때마다 여유 많은 GPU)'}", flush=True)
    threading.Thread(target=reaper, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
