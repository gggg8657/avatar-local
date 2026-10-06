#!/usr/bin/env python3
"""avatar-local — 내 얼굴·내 목소리 아바타 (토이). 사진 + 음성 샘플 + 대사 → 립싱크 mp4.
서버는 stdlib, 추론은 venv 서브프로세스(SadTalker, Apache-2.0 / F5-TTS, MIT).
규칙: 본인(또는 명시적으로 동의한 사람)의 얼굴·목소리만. 타인 영상에 얼굴을 바꿔 끼우는 기능은 없고, 넣지 않는다.

  bash setup.sh                                     # http://localhost:8777
  python3 app.py --cli photo.jpg voice.wav "대사" -o out.mp4 [--ref-text "샘플에서 말한 문장"]

env: PORT(8777) DEVICE(cpu|cuda, 기본 cpu — SadTalker 는 mps 미지원) VOICE_ENGINE(f5|server, 기본 f5 샘플 있으면 f5)
     TTS_BASE_URL(OpenAI 호환 /v1, 샘플 없을 때 폴백) TTS_VOICE F5_MODEL(F5TTS_v1_Base) F5_CKPT(한국어 등 커스텀 ckpt 경로)"""
import base64, datetime, json, os, re, secrets, shutil, subprocess, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("WORKSPACE") or os.path.join(ROOT, "_workspace")  # 포털이 AGENT_DATA/<도구> 로 모아 줌
ST = os.path.join(ROOT, "vendor", "SadTalker")
PY = os.path.join(ROOT, "venv", "bin", "python")
PORT = int(os.environ.get("PORT", "8777"))
DEVICE = os.environ.get("DEVICE", "cpu")
TTS = os.environ.get("TTS_BASE_URL", "").rstrip("/")
TTS_VOICE = os.environ.get("TTS_VOICE", "KR")
TTS_MODEL = os.environ.get("TTS_MODEL", "melo")
FFMPEG_LIB = os.environ.get("FFMPEG_LIB_DIR", os.path.expanduser("~/.local/ffmpeg-shared/lib"))  # setup.sh 가 conda 로 깔아 두는 자리
F5_MODEL, F5_CKPT, F5_VOCAB = os.environ.get("F5_MODEL", "F5TTS_v1_Base"), os.environ.get("F5_CKPT", ""), os.environ.get("F5_VOCAB", "")
_KO = os.path.join(ROOT, "models", "f5-ko")  # scripts/f5_ko.py 가 만드는 한국어 체크포인트(자모 단위) — 있으면 기본으로
if not F5_CKPT and os.path.exists(os.path.join(_KO, "model.safetensors")):
    F5_CKPT, F5_VOCAB = os.path.join(_KO, "model.safetensors"), os.path.join(_KO, "vocab.txt")
F5_JAMO = bool(F5_VOCAB) and "ᄀ" in open(F5_VOCAB, encoding="utf-8").read()  # 어휘가 한글 자모면 글을 NFD 로 풀어 넣는다
RUN_RE = r"\d{4}-\d{2}-\d{2}-[0-9a-f]{4}"


def ffmpeg(*a):
    return subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *a], capture_output=True, text=True)


def to_wav(src, dst):
    r = ffmpeg("-i", src, "-ac", "1", "-ar", "24000", dst)
    if r.returncode:
        raise RuntimeError("음성 변환 실패: " + r.stderr[-200:])


def tts(text, out_wav, ref_wav=None, ref_text=""):
    """목소리 복제(F5-TTS, 샘플 필요) 또는 TTS 서버 폴백. 둘 다 없으면 에러."""
    if ref_wav and os.path.exists(os.path.join(ROOT, "venv", "bin", "f5-tts_infer-cli")):
        import unicodedata
        nfd = (lambda s: unicodedata.normalize("NFD", s)) if F5_JAMO else (lambda s: s)
        cmd = [os.path.join(ROOT, "venv", "bin", "f5-tts_infer-cli"), "--model", F5_MODEL, "--ref_audio", ref_wav, "--ref_text", nfd(ref_text),
               "--gen_text", nfd(text), "--output_dir", os.path.dirname(out_wav), "--output_file", os.path.basename(out_wav), "--device", DEVICE]
        if F5_CKPT:
            cmd += ["--ckpt_file", F5_CKPT]
        if F5_VOCAB:
            cmd += ["--vocab_file", F5_VOCAB]
        env = dict(os.environ)
        if os.path.isdir(FFMPEG_LIB):  # torchaudio(torchcodec)가 오디오를 읽을 FFmpeg 공유 라이브러리 — 정적 ffmpeg 만 있는 서버용
            env["LD_LIBRARY_PATH"] = FFMPEG_LIB + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        if r.returncode or not os.path.exists(out_wav):
            raise RuntimeError("F5-TTS 실패: " + (r.stderr or r.stdout)[-300:])
        return "f5"
    if TTS:
        req = urllib.request.Request(TTS + "/audio/speech", json.dumps({"model": TTS_MODEL, "input": text, "voice": TTS_VOICE, "response_format": "wav"}).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r, open(out_wav, "wb") as f:
                f.write(r.read())
        except urllib.error.HTTPError as e:  # 서버가 준 오류 내용을 그대로("HTTP Error 500" 만으로는 원인을 모름)
            raise RuntimeError(f"TTS 서버 오류 {e.code}: {e.read().decode(errors='replace')[:300]}")
        return "server"
    raise RuntimeError("음성 엔진 없음: 목소리 샘플을 올리거나 TTS_BASE_URL 을 지정하세요")


def lipsync(photo, wav, d, emit, full=False):
    """SadTalker 서브프로세스 → mp4 경로. full=True 면 얼굴만 움직여 원래 그림(상반신·전신)에 다시 붙인다."""
    cmd = [PY, "inference.py", "--driven_audio", wav, "--source_image", photo, "--result_dir", d, "--checkpoint_dir", os.path.join(ROOT, "weights", "checkpoints"),
           "--still", "--preprocess", "full" if full else "crop", "--size", "256", "--batch_size", "4"] + (["--cpu"] if DEVICE == "cpu" else [])
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "shim"), "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(cmd, cwd=ST, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, log = None, []
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
    if p.returncode or not out or not os.path.exists(out):
        if any("landmark" in l for l in log):  # SadTalker 가 얼굴을 못 찾음 — 원문 스택 대신 고칠 방법을
            raise RuntimeError("사진에서 얼굴을 찾지 못했습니다. 얼굴이 정면으로 크게 나온 사진(안경·측면·그림 실루엣 X)을 쓰세요.")
        raise RuntimeError("립싱크 실패:\n" + "\n".join(log[-8:]))
    return out


def generate(photo, text, voice=None, ref_text="", consent=False, emit=lambda ev: None):
    if not consent:
        raise ValueError("본인(또는 동의한 사람)의 얼굴·목소리임을 확인해야 합니다")
    if not text.strip():
        raise ValueError("대사가 비었습니다")
    run_id = f"{datetime.date.today()}-{secrets.token_hex(2)}"
    d = os.path.join(WS, run_id)
    os.makedirs(d)
    shutil.copy(photo, os.path.join(d, "photo" + os.path.splitext(photo)[1].lower()))
    photo = os.path.join(d, "photo" + os.path.splitext(photo)[1].lower())
    ref = None
    if voice:
        emit({"stage": "voice", "msg": "음성 샘플 변환"})
        ref = os.path.join(d, "ref.wav"); to_wav(voice, ref)
    emit({"stage": "tts", "msg": "목소리 합성 " + ("(내 목소리 복제)" if ref else "(TTS 서버)")})
    wav = os.path.join(d, "speech.wav")
    engine = tts(text, wav, ref, ref_text)
    emit({"stage": "lipsync", "msg": "립싱크 영상 생성 (CPU 는 수 분)"})
    mp4 = lipsync(photo, wav, d, emit)
    final = os.path.join(d, "final.mp4")
    shutil.move(mp4, final)
    for junk in [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]:  # SadTalker 중간 산출물
        shutil.rmtree(os.path.join(d, junk), ignore_errors=True)
    meta = {"run_id": run_id, "text": text, "engine": engine, "device": DEVICE, "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return meta



# ── 가상 캐릭터 스튜디오: 말로 묘사 → 그림(Qwen-Image) → 말하기(립싱크) / 움직이기(Wan2.2 그림→영상) / persona 로 ──────────
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
SHOTS = {  # 키: (이름, 구도, 그림 크기, 영상 크기)
    "face": ("얼굴", "close-up head and shoulders portrait", (1024, 1024), (960, 960)),
    "half": ("상반신", "upper body from the waist up", (1104, 1472), (704, 1280)),
    "full": ("전신", "full body shot from head to toe, standing", (928, 1664), (704, 1280)),
}


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
        "nsfw, nude, nudity, nipples, lingerie, underwear, explicit, child, underage"  # 사내 포털 도구 — 노출·미성년 느낌은 어떤 묘사에도 막는다
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


def speak(run_id, idx, text, voice=None, ref_text="", emit=lambda ev: None):
    """고른 그림이 대사를 말하는 립싱크 영상 (상반신·전신은 얼굴만 움직임)"""
    d, meta = char_dir(run_id)
    if not text.strip(): raise ValueError("대사가 비었습니다")
    img = os.path.join(d, f"cand_{int(idx)}.png")
    ref = None
    if voice:
        ref = os.path.join(d, "ref.wav"); to_wav(voice, ref)
    emit({"stage": "tts", "msg": "목소리 합성 " + ("(목소리 복제)" if ref else "(TTS)")})
    wav = os.path.join(d, "talk.wav"); engine = tts(text, wav, ref, ref_text)
    emit({"stage": "lipsync", "msg": "립싱크"})
    mp4 = lipsync(img, wav, d, emit, full=meta["shot"] != "face")
    out = os.path.join(d, "talk.mp4"); shutil.move(mp4, out)
    for junk in [x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))]:
        shutil.rmtree(os.path.join(d, junk), ignore_errors=True)
    meta.update(picked=int(idx), talk={"text": text, "engine": engine}); json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    return {"run_id": run_id, "video": "talk.mp4", "engine": engine}


def move(run_id, idx, action, text="", fast=False, emit=lambda ev: None):
    """고른 그림이 동작하는 영상(Wan2.2). 대사가 있으면 목소리를 입힌다(입모양 동기는 아님)."""
    d, meta = char_dir(run_id)
    if not action.strip(): raise ValueError("어떻게 움직일지 적어 주세요")
    emit({"stage": "prompt", "msg": "동작을 영상 프롬프트로"})
    pr = llm_json("You write prompts for an image-to-video model that animates a given picture. Output only JSON: {\"prompt\": \"...\"}. "
                  "English, 30-60 words: describe the motion of the person in the picture (body, hands, head, expression), camera mostly static, "
                  "keep identity, clothes and background unchanged, smooth natural movement.", f"[그림 속 인물] {meta['desc']}\n[동작] {action}")
    w, h = SHOTS[meta["shot"]][3]
    frames, steps = (73, 30) if fast else (121, 40)
    emit({"stage": "video", "msg": f"동작 영상 ({'약 3초' if fast else '5초'}, 수 분)"})
    out = os.path.join(d, "move_raw.mp4")
    worker("/video", {"image": os.path.join(d, f"cand_{int(idx)}.png"), "prompt": pr["prompt"], "width": w, "height": h, "frames": frames, "steps": steps,
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
    meta.update(picked=int(idx), move={"action": action, "prompt": pr["prompt"], "text": text}); json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
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


def status():
    return {"sadtalker": os.path.exists(os.path.join(ROOT, "weights", "checkpoints", "SadTalker_V0.0.2_256.safetensors")) and os.path.exists(os.path.join(ST, "inference.py")),
            "f5": os.path.exists(os.path.join(ROOT, "venv", "bin", "f5-tts_infer-cli")), "tts_server": bool(TTS), "device": DEVICE,
            "f5_korean": bool(F5_CKPT)}


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
        voice = None
        try:
            if self.path == "/api/imagine":
                r = imagine(req.get("desc") or "", req.get("style") or "real", req.get("shot") or "half", req.get("n") or 4, emit)
            elif self.path == "/api/speak":
                if req.get("voice_b64"):
                    up = os.path.join(WS, "_upload"); os.makedirs(up, exist_ok=True)
                    voice = os.path.join(up, secrets.token_hex(3) + "_" + re.sub(r"[^\w.\-]", "_", os.path.basename(req.get("voice_name") or "voice.wav")))
                    open(voice, "wb").write(base64.b64decode(req["voice_b64"]))
                r = speak(req.get("run_id"), req.get("idx", 0), req.get("text") or "", voice, req.get("ref_text") or "", emit)
            else:
                r = move(req.get("run_id"), req.get("idx", 0), req.get("action") or "", req.get("text") or "", bool(req.get("fast")), emit)
            emit({"done": r})
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}"})
        finally:
            if voice and os.path.exists(voice): os.remove(voice)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
        photo = voice = None
        try:
            photo = save("photo_b64", "photo_name", (".jpg", ".jpeg", ".png"))
            if not photo:
                raise ValueError("사진이 필요합니다")
            voice = save("voice_b64", "voice_name", (".wav", ".mp3", ".m4a", ".webm", ".ogg"))
            emit({"done": generate(photo, req.get("text") or "", voice, req.get("ref_text") or "", bool(req.get("consent")), emit)})
        except Exception as e:
            emit({"error": f"{type(e).__name__}: {e}"})
        finally:
            for p in (photo, voice):
                if p and os.path.exists(p):
                    os.remove(p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        a = sys.argv[2:]
        out = a[a.index("-o") + 1] if "-o" in a else "out.mp4"
        ref_text = a[a.index("--ref-text") + 1] if "--ref-text" in a else ""
        pos = [x for i, x in enumerate(a) if x not in ("-o", "--ref-text") and (i == 0 or a[i - 1] not in ("-o", "--ref-text"))]
        photo, voice, text = pos[0], (pos[1] if len(pos) > 2 else None), pos[-1]
        m = generate(photo, text, voice, ref_text, consent=True, emit=lambda ev: print(ev.get("msg") or ev.get("log", ""), file=sys.stderr))
        shutil.copy(os.path.join(WS, m["run_id"], "final.mp4"), out); print(out)
        sys.exit(0)
    print(f"avatar-local → http://localhost:{PORT}  {status()}  {_SIG}")
    ThreadingHTTPServer(("", PORT), H).serve_forever()
