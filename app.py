#!/usr/bin/env python3
"""avatar-local — 내 얼굴 아바타 · 가상 캐릭터 스튜디오 (토이). 사진 + 대사 → 립싱크 mp4.
서버는 stdlib, 추론은 venv 서브프로세스(SadTalker, Apache-2.0). 목소리는 TTS 도구(tts-local, MeloTTS)만 쓴다 — 목소리 복제는 없다.
규칙: 본인(또는 명시적으로 동의한 사람)의 얼굴만. 타인 영상에 얼굴을 바꿔 끼우는 기능은 없고, 넣지 않는다.

  bash setup.sh                                     # http://localhost:8777
  python3 app.py --cli photo.jpg "대사" -o out.mp4

env: PORT(8777) DEVICE(cpu|cuda, 기본 cpu — SadTalker 는 mps 미지원)
     TTS_BASE_URL(OpenAI 호환 /v1, 기본 tts-local http://127.0.0.1:8771/v1) TTS_VOICE(KR) TTS_MODEL(melo)"""
import base64, datetime, json, os, re, secrets, shutil, subprocess, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gpu_pick import env_for, label, pick, release

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("WORKSPACE") or os.path.join(ROOT, "_workspace")  # 포털이 AGENT_DATA/<도구> 로 모아 줌
ST = os.path.join(ROOT, "vendor", "SadTalker")
PY = os.path.join(ROOT, "venv", "bin", "python")
PORT = int(os.environ.get("PORT", "8777"))
DEVICE = os.environ.get("DEVICE", "cpu")
TTS = (os.environ.get("TTS_BASE_URL") or "http://127.0.0.1:8771/v1").rstrip("/")
TTS_VOICE = os.environ.get("TTS_VOICE", "KR")
TTS_MODEL = os.environ.get("TTS_MODEL", "melo")
RUN_RE = r"\d{4}-\d{2}-\d{2}-[0-9a-f]{4}"


def ffmpeg(*a):
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], capture_output=True, text=True)


def tts(text, out_wav):
    """대사 → wav. TTS 도구(tts-local, OpenAI 호환 /v1/audio/speech)만 쓴다 — 목소리 복제 없음(라이선스)."""
    req = urllib.request.Request(TTS + "/audio/speech", json.dumps({"model": TTS_MODEL, "input": text, "voice": TTS_VOICE, "response_format": "wav"}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r, open(out_wav, "wb") as f:
            f.write(r.read())
    except urllib.error.HTTPError as e:  # 서버가 준 오류 내용을 그대로("HTTP Error 500" 만으로는 원인을 모름)
        raise RuntimeError(f"TTS 도구 오류 {e.code}: {e.read().decode(errors='replace')[:300]}")
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"TTS 도구(tts-local)를 켜 주세요 — 목소리를 만들 수 없습니다 ({TTS}: {getattr(e, 'reason', e)})")
    return "tts-local"


def lipsync(photo, wav, d, emit, full=False):
    """SadTalker 서브프로세스 → mp4 경로. full=True 면 얼굴만 움직여 원래 그림(상반신·전신)에 다시 붙인다."""
    cmd = [PY, "inference.py", "--driven_audio", wav, "--source_image", photo, "--result_dir", d, "--checkpoint_dir", os.path.join(ROOT, "weights", "checkpoints"),
           "--still", "--preprocess", "full" if full else "crop", "--size", "256", "--batch_size", "4"] + (["--cpu"] if DEVICE == "cpu" else [])
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "shim"), "PYTHONUNBUFFERED": "1"}
    g = None
    if DEVICE == "cuda":  # 실행할 때마다 여유 메모리가 가장 큰 GPU 1장만 보이게 (SadTalker 는 그 안에서 cuda)
        g = pick(6000); env = env_for(g, env); emit({"log": f"립싱크 {label(g)} 에서 실행"})
        if not g: cmd.append("--cpu")
    out, log = None, []
    try:
        p = subprocess.Popen(cmd, cwd=ST, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            line = line.strip()
            if not line:
                continue
            log.append(line)
            m = re.search(r"generated video is named:\s*(\S+\.mp4)", line)
            if m:
                out = m.group(1)
            if "%" in line or "it/s" in line or "named" in line:  # 진행 표시만 UI 로
                emit({"log": line[-120:]})
        p.wait()
    finally:
        release(g)  # pick 이 남긴 GPU 예약을 푼다
    if p.returncode or not out or not os.path.exists(out):
        if any("landmark" in l for l in log):  # SadTalker 가 얼굴을 못 찾음 — 원문 스택 대신 고칠 방법을
            raise RuntimeError("사진에서 얼굴을 찾지 못했습니다. 얼굴이 정면으로 크게 나온 사진(안경·측면·그림 실루엣 X)을 쓰세요.")
        raise RuntimeError("립싱크 실패:\n" + "\n".join(log[-8:]))
    return out


def generate(photo, text, consent=False, emit=lambda ev: None):
    if not consent:
        raise ValueError("본인(또는 동의한 사람)의 얼굴임을 확인해야 합니다")
    if not text.strip():
        raise ValueError("대사가 비었습니다")
    run_id = f"{datetime.date.today()}-{secrets.token_hex(2)}"
    d = os.path.join(WS, run_id)
    os.makedirs(d)
    shutil.copy(photo, os.path.join(d, "photo" + os.path.splitext(photo)[1].lower()))
    photo = os.path.join(d, "photo" + os.path.splitext(photo)[1].lower())
    emit({"stage": "tts", "msg": "목소리 합성 (TTS 도구)"})
    wav = os.path.join(d, "speech.wav")
    engine = tts(text, wav)
    emit({"stage": "lipsync", "msg": "립싱크 영상 생성 (CPU 는 수 분)"})
    mp4 = lipsync(photo, wav, d, emit)
    final = os.path.join(d, "final.mp4")
    shutil.move(mp4, final)
    for junk in [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]:  # SadTalker 중간 산출물
        shutil.rmtree(os.path.join(d, junk), ignore_errors=True)
    meta = {"run_id": run_id, "text": text, "engine": engine, "device": DEVICE, "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return meta



# ── 가상 캐릭터 스튜디오: 말로 묘사 → 그림(Qwen-Image-2512) → 말하기(Wan2.2-S2V | 빠르게 SadTalker) / 움직이기(Wan2.2 I2V-A14B) / persona 로 ──
# 실존 인물이 아닌 생성 캐릭터만 다룬다(사진 업로드 없음). 그래서 동의 체크 없이 쓰고, 실사 사진을 올리는 /api/run 은 그대로 동의를 받는다.
STUDIO_URL = "http://127.0.0.1:" + os.environ.get("STUDIO_PORT", "8787")
PERSONA_URL = os.environ.get("PERSONA_URL", "http://localhost:8776").rstrip("/")
LLM_API = os.environ.get("LLM_API", "ollama")
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1" if LLM_API == "openai" else "http://localhost:11434").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
STYLES = {  # 키: (이름, 그림 프롬프트에 붙는 스타일)
    "real": ("실사", "photorealistic RAW photo, natural skin texture, soft daylight, 85mm lens"),
    "virtual": ("버추얼 휴먼", "flawless virtual human, AI influencer look, perfectly symmetrical face, porcelain smooth skin, glossy studio lighting, hyperreal CGI"),
    "3d": ("3D 애니메이션", "3D animated movie character, Pixar-like stylized proportions, soft subsurface scattering, cinematic render"),
    "anime": ("애니메", "anime style, cel shading, clean line art, vivid colors"),
    "webtoon": ("웹툰", "Korean webtoon style, clean line art, flat soft shading"),
    "illust": ("일러스트", "digital painting illustration, painterly brushwork, soft palette"),
}
SHOTS = {  # 키: (이름, 구도, 그림 크기) — 영상 크기는 vsize() 가 그림 비율로
    "face": ("얼굴", "close-up head and shoulders portrait", (1328, 1328)),  # 그림 크기는 Qwen-Image-2512 모델 카드 권장 비율
    "half": ("상반신", "upper body from the waist up", (1104, 1472)),
    "full": ("전신", "full body shot from head to toe, standing", (928, 1664)),
}
# 품질: (영상 넓이, 프레임, 단계) — Wan2.2 I2V-A14B 모델 카드: 81프레임(16fps, 5초)·40단계·guidance 3.5, 720p | 480p
MOVE_Q = {"best": (720 * 1280, 81, 40), "fast": (480 * 832, 81, 30)}
WAN = os.path.join(ROOT, "vendor", "Wan2.2")  # 공식 Wan2.2 코드(setup.sh 가 받음) — 말하기 최고 품질(S2V-14B 는 diffusers 에 아직 없음)
S2V_MODEL = os.environ.get("S2V_MODEL", "Wan-AI/Wan2.2-S2V-14B")
S2V_NEED_MB = 64000  # S2V 한 번에 GPU 에 필요한 여유(실측 최고 약 60GB, offload 켬)


def llm_json(system, user):
    """작은 LLM 호출 → JSON (그림·동작 프롬프트 만들기, persona 설정 쓰기)"""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if LLM_API == "openai":
        url, body = LLM_BASE + "/chat/completions", {"model": LLM_MODEL, "messages": msgs, "temperature": 0.4}
    else:
        url, body = LLM_BASE + "/api/chat", {"model": LLM_MODEL, "messages": msgs, "stream": False, "think": False, "options": {"temperature": 0.4}}
    with urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300) as r:
        j = json.load(r)
    out = j["choices"][0]["message"]["content"] if LLM_API == "openai" else j["message"]["content"]
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", out, flags=re.S), re.S)
    if not m:
        raise RuntimeError("LLM 이 JSON 을 쓰지 못했습니다: " + out[:200])
    return json.loads(m.group(0))


def worker(path, body, emit=lambda ev: None):
    """스튜디오 작업 프로세스에 요청(없으면 띄움) → 마지막 결과. 진행률은 emit 으로."""
    try:
        urllib.request.urlopen(STUDIO_URL + "/", timeout=2)
    except Exception:
        emit({"stage": "worker", "msg": "그림·영상 작업 프로세스 시작"})
        wp = subprocess.Popen([PY, os.path.join(ROOT, "studio_worker.py")], cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
                              stdout=open(os.path.join(ROOT, "studio.log"), "ab"), stderr=subprocess.STDOUT, start_new_session=True)
        open(os.path.join(ROOT, ".studio.pid"), "w").write(str(wp.pid))  # setup.sh stop 이 같이 끈다(GPU 메모리 반환)
        for _ in range(180):
            try:
                urllib.request.urlopen(STUDIO_URL + "/", timeout=2); break
            except Exception:
                import time; time.sleep(1)
        else:
            raise RuntimeError("작업 프로세스가 뜨지 않았습니다 (studio.log 확인)")
    with urllib.request.urlopen(urllib.request.Request(STUDIO_URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200) as r:
        for line in r:
            ev = json.loads(line)
            if "error" in ev: raise RuntimeError(ev["error"])
            if "done" in ev: return ev["done"]
            emit(ev)
    raise RuntimeError("작업 프로세스 응답이 끊겼습니다")


def char_dir(run_id):
    if not re.fullmatch(RUN_RE, run_id or ""): raise ValueError("잘못된 ID")
    d = os.path.join(WS, run_id); meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    if meta.get("kind") != "character": raise ValueError("가상 캐릭터가 아닙니다")
    return d, meta


def imagine(desc, style="real", shot="half", n=4, emit=lambda ev: None):
    """묘사 → 후보 그림 n 장"""
    if not desc.strip(): raise ValueError("어떤 캐릭터인지 묘사해 주세요")
    style = style if style in STYLES else "real"; shot = shot if shot in SHOTS else "half"; n = max(1, min(int(n or 4), 6))
    emit({"stage": "prompt", "msg": "묘사를 그림 프롬프트로"})
    pr = llm_json("You write prompts for a text-to-image model. Output only JSON: {\"prompt\": \"...\", \"negative\": \"...\"}. "
                  "The prompt is English, 40-80 words, describes ONE fictional (not a real or famous) adult person exactly as the user describes "
                  "(age, gender, hair, face, clothes, mood) — always a clearly adult person (20+), fully clothed, non-explicit — facing the camera with eyes looking at the viewer, mouth gently closed, "
                  "plain softly lit background, nothing covering the face. The negative lists things to avoid.",
                  f"[묘사] {desc}\n[스타일] {STYLES[style][1]}\n[구도] {SHOTS[shot][1]}")
    prompt = f"{pr['prompt']}, {SHOTS[shot][1]}, {STYLES[style][1]}, Ultra HD, 4K"
    negative = (pr.get("negative") or "") + ", text, watermark, logo, extra fingers, deformed hands, cropped head, multiple people, blurry, " \
        "低分辨率，低画质，肢体畸形，手指畸形，人脸无细节，构图混乱，文字模糊，扭曲, " \
        "nsfw, nude, nudity, nipples, lingerie, underwear, explicit, child, underage"  # 사내 포털 도구 — 노출·미성년 느낌은 어떤 묘사에도 막는다 (가운데는 Qwen-Image-2512 카드의 화질 negative)
    run_id = f"{datetime.date.today()}-{secrets.token_hex(2)}"; d = os.path.join(WS, run_id); os.makedirs(d)
    meta = {"run_id": run_id, "kind": "character", "desc": desc, "style": style, "shot": shot, "prompt": prompt, "negative": negative,
            "n": n, "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    w, h = SHOTS[shot][2]
    emit({"stage": "draw", "msg": f"{STYLES[style][0]} · {SHOTS[shot][0]} 그리는 중 ({n}장)"})
    try:
        worker("/imagine", {"prompt": prompt, "negative": negative, "width": w, "height": h, "n": n, "seed": secrets.randbelow(10 ** 6), "out_dir": d},
               lambda ev: emit({"image": ev["image"], "run_id": run_id} if "image" in ev else ev))
    except Exception:
        shutil.rmtree(d, ignore_errors=True)  # 실패한 그리기는 흔적을 남기지 않는다
        raise
    meta["candidates"] = [f"cand_{i}.png" for i in range(n)]
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return meta


def vsize(shot, area):
    """그림 비율 그대로, 넓이 area 안에서 16 의 배수인 영상 크기 (w, h)"""
    w, h = SHOTS[shot][2]; r = h / w
    return int((area / r) ** .5) // 16 * 16, int((area * r) ** .5) // 16 * 16


def s2v(photo, wav, out, prompt, emit):
    """Wan2.2-S2V-14B 서브프로세스(공식 generate.py) — 그림 + 음성 → 얼굴·상반신이 말하는 mp4(16fps, 음성 포함). 5초 조각마다 H100 에서 약 23분."""
    if not os.path.exists(os.path.join(WAN, "generate.py")):
        raise RuntimeError("Wan2.2 코드가 없습니다 — bash setup.sh 로 vendor/Wan2.2 를 받으세요 (또는 '빠르게' 로)")
    emit({"stage": "s2v", "msg": "말하기 모델 확인 (처음이면 약 49GB 받기)"})
    ckpt = os.environ.get("S2V_CKPT") or subprocess.run([PY, "-c", f"from huggingface_hub import snapshot_download as s; print(s({S2V_MODEL!r}))"],
                                                        capture_output=True, text=True).stdout.strip().split("\n")[-1]
    if not os.path.isdir(ckpt):
        raise RuntimeError(f"{S2V_MODEL} 를 받지 못했습니다 (S2V_CKPT 로 폴더 지정 가능)")
    # S2V 는 5초(80프레임) 조각 단위로 만들고, 영상과 음성 중 짧은 쪽에 맞춰 자른다 → 말끝이 잘리지 않게 같은 조각 수 안에서 뒤에 무음을 덧댄다
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", wav], capture_output=True, text=True).stdout or 0)
    padded = os.path.splitext(out)[0] + "_s2v.wav"
    ffmpeg("-i", wav, "-af", f"apad=whole_dur={min(dur + 0.6, (int(dur / 5) + 1) * 5 - 0.05):.3f}", padded)
    cmd = [PY, "generate.py", "--task", "s2v-14B", "--size", "1024*704", "--ckpt_dir", ckpt, "--offload_model", "True", "--convert_model_dtype",
           "--prompt", prompt, "--image", photo, "--audio", padded, "--save_file", out, "--base_seed", str(secrets.randbelow(10 ** 6))]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.path.join(ROOT, "shim", "wan")}  # flash_attn 대역(SDPA) — nvcc 없이 빌드 불가
    g = None
    if DEVICE == "cuda":
        g = pick(S2V_NEED_MB, hold_s=7200)  # 오프로드라 메모리를 올렸다 내렸다 함 → 끝날 때까지 예약
        if not g: raise RuntimeError(f"GPU 여유 메모리 부족 — 약 {S2V_NEED_MB // 1024}GB 필요. 잠시 뒤 다시 하거나 '빠르게' 로")
        env = env_for(g, env); emit({"log": f"말하기 영상 {label(g)} 에서 실행"})
    log, clip, last, gen, p = [], 1, 0, False, None
    try:
        p = subprocess.Popen(cmd, cwd=WAN, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:  # tqdm 의 \r 도 줄바꿈으로 들어온다
            line = line.strip()
            if not line: continue
            log.append(line); log = log[-40:]
            gen = gen or "Generating video" in line  # 그 전의 진행 막대(가중치 읽기)는 단계로 치지 않는다
            m = gen and re.search(r"(\d+)/(\d+) \[", line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if i < last: clip += 1  # 대사가 길면 5초 단위 조각(clip)을 이어 만든다
                if i != last: emit({"step": i, "total": n, "clip": clip}); last = i
            else:
                for k, msg in (("Creating WanS2V", "모델 올리는 중 (약 1분)"), ("Generating video", "영상 만드는 중 — 5초 조각마다 40단계"), ("Saving generated", "저장·음성 합치는 중")):
                    if k in line: emit({"log": msg})
        p.wait()
    finally:
        if p and p.poll() is None: p.kill(); p.wait()  # 화면을 닫아 중간에 끊기면 GPU 를 계속 잡지 않게
        release(g)
        if os.path.exists(padded): os.remove(padded)
    if p.returncode or not os.path.exists(out):
        raise RuntimeError("말하기 영상 실패:\n" + "\n".join(log[-8:]))
    return out


def speak(run_id, idx, text, quality="best", emit=lambda ev: None):
    """고른 그림이 대사를 말하는 영상. 최고 품질 = Wan2.2-S2V(얼굴·상반신·입모양, 5초마다 약 23분) / 빠르게 = SadTalker(얼굴만, 수십 초)"""
    d, meta = char_dir(run_id)
    if not text.strip(): raise ValueError("대사가 비었습니다")
    img = os.path.join(d, f"cand_{int(idx)}.png")
    emit({"stage": "tts", "msg": "목소리 합성 (TTS 도구)"})
    wav = os.path.join(d, "talk.wav"); engine = tts(text, wav)
    out = os.path.join(d, "talk.mp4")
    if quality == "fast":
        emit({"stage": "lipsync", "msg": "립싱크 (빠르게 — 얼굴만)"})
        mp4 = lipsync(img, wav, d, emit, full=meta["shot"] != "face"); shutil.move(mp4, out)
    else:
        emit({"stage": "s2v", "msg": "말하는 영상 (최고 품질 — 대사 5초마다 약 25분)"})
        s2v(img, wav, out, f"{meta['desc']}. The person in the picture is talking to the camera, lips moving clearly with the speech, natural facial "
                           "expressions and gentle head and hand movements, static camera, same face, clothes and background as the picture.", emit)
    for junk in [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]:
        shutil.rmtree(os.path.join(d, junk), ignore_errors=True)
    model = "SadTalker" if quality == "fast" else "Wan2.2-S2V-14B"
    meta = char_dir(run_id)[1]  # 그동안 같은 캐릭터로 '움직이기' 가 끝났을 수 있어 다시 읽고 덧붙인다
    meta.update(picked=int(idx), talk={"text": text, "engine": engine, "model": model}); json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return {"run_id": run_id, "video": "talk.mp4", "engine": engine, "model": model}


def move(run_id, idx, action, text="", quality="best", emit=lambda ev: None):
    """고른 그림이 동작하는 5초 영상(Wan2.2 I2V-A14B, 최고 품질 720p | 빠르게 480p). 대사가 있으면 목소리를 입힌다(입모양 동기는 아님)."""
    d, meta = char_dir(run_id)
    if not action.strip(): raise ValueError("어떻게 움직일지 적어 주세요")
    emit({"stage": "prompt", "msg": "동작을 영상 프롬프트로"})
    pr = llm_json("You write prompts for an image-to-video model that animates a given picture. Output only JSON: {\"prompt\": \"...\"}. "
                  "English, 30-60 words: describe the motion of the person in the picture (body, hands, head, expression), camera mostly static, "
                  "keep identity, clothes and background unchanged, smooth natural movement.", f"[그림 속 인물] {meta['desc']}\n[동작] {action}")
    quality = quality if quality in MOVE_Q else "best"
    area, frames, steps = MOVE_Q[quality]; w, h = vsize(meta["shot"], area)
    emit({"stage": "video", "msg": f"동작 영상 5초 ({w}×{h}, {'빠르게 — 약 7분' if quality == 'fast' else '최고 품질 — 약 30~35분'})"})
    out = os.path.join(d, "move_raw.mp4")
    worker("/video", {"image": os.path.join(d, f"cand_{int(idx)}.png"), "prompt": pr["prompt"], "width": w, "height": h, "frames": frames, "steps": steps, "cfg": 3.5, "fps": 16,
                      "negative": "blurry, distorted face, extra limbs, deformed hands, flicker, static image, text, watermark", "out": out}, emit)
    final = os.path.join(d, "move.mp4")
    if text.strip():
        emit({"stage": "tts", "msg": "대사 음성 입히기"})
        wav = os.path.join(d, "move.wav"); tts(text, wav)
        dur = lambda f: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f], capture_output=True, text=True).stdout or 0)
        if dur(wav) > dur(out):  # 대사가 길면 영상을 반복해 대사 길이에 맞추고
            ffmpeg("-stream_loop", "-1", "-i", out, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final)
        else:  # 짧으면 영상 길이 그대로, 뒤는 무음 (apad 를 영상 길이까지만 — -shortest+copy 조합은 끝나지 않는다)
            ffmpeg("-i", out, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", f"apad=whole_dur={dur(out):.3f}", "-c:a", "aac", final)
    else:
        shutil.copy(out, final)
    meta = char_dir(run_id)[1]  # 그동안 끝난 '말하기' 기록을 지우지 않게 다시 읽고 덧붙인다
    meta.update(picked=int(idx), move={"action": action, "prompt": pr["prompt"], "text": text, "quality": quality, "model": "Wan2.2-I2V-A14B"}); json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return {"run_id": run_id, "video": "move.mp4"}


def to_persona(run_id, idx, name, personality):
    """고른 그림 + 성격 → persona-local 새 캐릭터 (입 위치도 자동)"""
    d, meta = char_dir(run_id)
    if not name.strip(): raise ValueError("이름을 정해 주세요")
    img = os.path.join(d, f"cand_{int(idx)}.png")
    m = worker("/mouth", {"image": img}).get("mouth")
    body = {"name": name.strip(), "look": meta["desc"], "personality": personality.strip(), "mouth": m,
            "image_b64": base64.b64encode(open(img, "rb").read()).decode()}
    try:
        with urllib.request.urlopen(urllib.request.Request(PERSONA_URL + "/api/persona_new", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"persona 오류 {e.code}: {e.read().decode(errors='replace')[:300]}")

def list_runs():
    out = []
    for n in sorted(os.listdir(WS), key=lambda n: os.path.getmtime(os.path.join(WS, n)), reverse=True)[:50] if os.path.isdir(WS) else []:
        p = os.path.join(WS, n, "meta.json")
        if os.path.exists(p):
            m = json.load(open(p, encoding="utf-8"))
            if m.get("kind") == "character" and not (m.get("talk") or m.get("move")):
                continue  # 갤러리엔 완성된 영상만 — 그려 보기만 한 후보는 올리지 않는다
            out.append(m)
    return out


TRASH = os.path.join(WS, ".trash")  # 지운 항목은 .trash/<시각>/<run_id> 로 옮겨 둔다 (되살리기 가능)


def run_path(run_id):
    """run_id → WS 바로 아래 실제 폴더. 형식이 틀리거나 WS 밖(링크·../)이면 ValueError"""
    if not isinstance(run_id, str) or not re.fullmatch(RUN_RE, run_id): raise ValueError("잘못된 항목입니다")
    d = os.path.join(WS, run_id)
    if os.path.islink(d) or os.path.dirname(os.path.realpath(d)) != os.path.realpath(WS): raise ValueError("잘못된 항목입니다")
    return d


def trash_runs(ids):
    """갤러리 항목(폴더 통째로: 영상·음성·후보 그림·meta)을 휴지통으로. 한 번 지운 것들은 한 묶음(batch)"""
    if not isinstance(ids, list) or not ids or len(ids) > 200: raise ValueError("지울 항목을 고르세요")
    paths = [run_path(i) for i in ids]  # 하나라도 틀리면 아무것도 옮기지 않는다
    batch = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    dst = os.path.join(TRASH, batch); deleted, missing = [], []
    for i, d in zip(ids, paths):
        if not os.path.isdir(d): missing.append(i); continue
        os.makedirs(dst, exist_ok=True); shutil.move(d, os.path.join(dst, i)); deleted.append(i)
    return {"deleted": deleted, "missing": missing, "batch": batch if deleted else None}


def restore_trash(batch):
    if not isinstance(batch, str) or not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", batch): raise ValueError("잘못된 묶음입니다")
    src = os.path.join(TRASH, batch)
    if not os.path.isdir(src): raise ValueError("휴지통에 없습니다")
    restored = []
    for i in sorted(os.listdir(src)):
        if re.fullmatch(RUN_RE, i) and not os.path.exists(os.path.join(WS, i)):
            shutil.move(os.path.join(src, i), os.path.join(WS, i)); restored.append(i)
    if not os.listdir(src): os.rmdir(src)
    return {"restored": restored}


def status():
    try:  # TTS 도구가 켜져 있는지 (목소리는 이것만 씀)
        urllib.request.urlopen(TTS + "/models", timeout=2); tts_up = True
    except Exception:
        tts_up = False
    return {"sadtalker": os.path.exists(os.path.join(ROOT, "weights", "checkpoints", "SadTalker_V0.0.2_256.safetensors")) and os.path.exists(os.path.join(ST, "inference.py")),
            "tts_server": tts_up, "device": DEVICE, "s2v": os.path.exists(os.path.join(WAN, "generate.py"))}


HTML = open(os.path.join(ROOT, "ui.html"), encoding="utf-8").read()

# ── 저작권 표기 (LICENSE·NOTICE 참고) ─────────────────────────────────────
_SIG = __import__("base64").b64decode("wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==").decode()
_SIG_A = __import__("base64").b64decode("Z2dnZzg2NTcgPGRvbmdqdWtpbS5kZXZAZ21haWwuY29tPg==").decode()


def signed(html):
    """화면에 저작권 표기를 붙인다. ui.html 에서 지워져도 서버가 내보낼 때 다시 붙는다."""
    name, mail = _SIG.split(" · ")
    if 'name="author"' not in html:
        meta = f'<meta name="author" content="{name[7:]} <{mail}>">'
        html = html.replace("<head>", "<head>" + meta, 1) if "<head>" in html else meta + html
    if "data-sig" not in html:
        tag = (f'<!-- {_SIG} --><div data-sig title="{mail}" style="text-align:center;font-size:11px;color:#9aa0a6;'
               f'opacity:.55;margin:28px 0 8px">{name}</div>')
        html = html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag
    return html


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("X-Author", _SIG_A); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/status":
            return self._send(status())
        if p == "/api/runs":
            return self._send(list_runs())
        if p == "/api/studio":  # 스튜디오 선택지 (스타일·구도)
            return self._send({"styles": {k: v[0] for k, v in STYLES.items()}, "shots": {k: v[0] for k, v in SHOTS.items()}})
        m = re.fullmatch(rf"/api/runs/({RUN_RE})/(final\.mp4|talk\.mp4|move\.mp4|cand_\d\.png)", p)
        if m:
            f = os.path.join(WS, m.group(1), m.group(2))
            if os.path.exists(f):
                return self._send(open(f, "rb").read(), "image/png" if f.endswith(".png") else "video/mp4")
            return self._send({"error": "삭제됐거나 없는 파일입니다"}, code=404)
        self._send(signed(HTML).encode(), "text/html; charset=utf-8")

    def studio(self, req):
        if self.path == "/api/to_persona":
            try:
                return self._send(to_persona(req.get("run_id"), req.get("idx", 0), req.get("name") or "", req.get("personality") or ""))
            except Exception as e:
                return self._send({"error": f"{type(e).__name__}: {e}"}, code=400)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8"); self.send_header("Cache-Control", "no-cache"); self.end_headers()

        def emit(ev):
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()
        try:
            if self.path == "/api/imagine":
                r = imagine(req.get("desc") or "", req.get("style") or "real", req.get("shot") or "half", req.get("n") or 4, emit)
            elif self.path == "/api/speak":
                r = speak(req.get("run_id"), req.get("idx", 0), req.get("text") or "", req.get("quality") or "best", emit)
            else:
                r = move(req.get("run_id"), req.get("idx", 0), req.get("action") or "", req.get("text") or "", req.get("quality") or "best", emit)
            emit({"done": r})
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}"})

    def do_DELETE(self):
        m = re.fullmatch(rf"/api/runs/({RUN_RE})", self.path)
        if not m:
            return self._send({"error": "not found"}, code=404)
        return self.delete_runs([m.group(1)])

    def delete_runs(self, ids):
        try:
            r = trash_runs(ids)
        except ValueError as e:
            return self._send({"error": str(e)}, code=400)
        return self._send(r, code=200 if r["deleted"] else 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/runs/delete":  # {run_ids:[…]} (또는 run_id 하나) → 휴지통으로
            return self.delete_runs(req.get("run_ids") or [req.get("run_id")])
        if self.path == "/api/runs/restore":  # {batch} → 그 삭제 묶음을 되살린다
            try:
                return self._send(restore_trash(req.get("batch")))
            except ValueError as e:
                return self._send({"error": str(e)}, code=400)
        if self.path in ("/api/imagine", "/api/speak", "/api/move", "/api/to_persona"):  # 가상 캐릭터 스튜디오
            return self.studio(req)
        if self.path != "/api/run":
            return self._send({"error": "not found"}, code=404)
        up = os.path.join(WS, "_upload"); os.makedirs(up, exist_ok=True)

        def save(key, name_key, allowed):
            if not req.get(key):
                return None
            name = re.sub(r"[^\w.\-]", "_", os.path.basename(req.get(name_key) or "file"))
            if not name.lower().endswith(allowed):
                raise ValueError(f"허용 확장자: {allowed}")
            p = os.path.join(up, secrets.token_hex(3) + "_" + name)
            with open(p, "wb") as f:
                f.write(base64.b64decode(req[key]))
            return p
        self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8"); self.send_header("Cache-Control", "no-cache"); self.end_headers()

        def emit(ev):
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()); self.wfile.flush()
        photo = None
        try:
            photo = save("photo_b64", "photo_name", (".jpg", ".jpeg", ".png"))
            if not photo:
                raise ValueError("사진이 필요합니다")
            emit({"done": generate(photo, req.get("text") or "", bool(req.get("consent")), emit)})
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}"})
        finally:
            if photo and os.path.exists(photo):
                os.remove(photo)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        a = sys.argv[2:]
        out = a[a.index("-o") + 1] if "-o" in a else "out.mp4"
        pos = [x for i, x in enumerate(a) if x != "-o" and (i == 0 or a[i - 1] != "-o")]
        photo, text = pos[0], pos[-1]
        m = generate(photo, text, consent=True, emit=lambda ev: print(ev.get("msg") or ev.get("log", ""), file=sys.stderr))
        shutil.copy(os.path.join(WS, m["run_id"], "final.mp4"), out); print(out)
        sys.exit(0)
    print(f"avatar-local → http://localhost:{PORT}  {status()}  {_SIG}")
    ThreadingHTTPServer(("", PORT), H).serve_forever()
