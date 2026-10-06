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
F5_MODEL, F5_CKPT = os.environ.get("F5_MODEL", "F5TTS_v1_Base"), os.environ.get("F5_CKPT", "")
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
        cmd = [os.path.join(ROOT, "venv", "bin", "f5-tts_infer-cli"), "--model", F5_MODEL, "--ref_audio", ref_wav, "--ref_text", ref_text,
               "--gen_text", text, "--output_dir", os.path.dirname(out_wav), "--output_file", os.path.basename(out_wav), "--device", DEVICE]
        if F5_CKPT:
            cmd += ["--ckpt_file", F5_CKPT]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode or not os.path.exists(out_wav):
            raise RuntimeError("F5-TTS 실패: " + (r.stderr or r.stdout)[-300:])
        return "f5"
    if TTS:
        req = urllib.request.Request(TTS + "/audio/speech", json.dumps({"model": "tts", "input": text, "voice": TTS_VOICE, "response_format": "wav"}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r, open(out_wav, "wb") as f:
            f.write(r.read())
        return "server"
    raise RuntimeError("음성 엔진 없음: 목소리 샘플을 올리거나 TTS_BASE_URL 을 지정하세요")


def lipsync(photo, wav, d, emit):
    """SadTalker 서브프로세스 → mp4 경로."""
    cmd = [PY, "inference.py", "--driven_audio", wav, "--source_image", photo, "--result_dir", d, "--checkpoint_dir", os.path.join(ROOT, "weights", "checkpoints"),
           "--still", "--preprocess", "crop", "--size", "256", "--batch_size", "4"] + (["--cpu"] if DEVICE == "cpu" else [])
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


def list_runs():
    out = []
    for n in sorted(os.listdir(WS), reverse=True)[:50] if os.path.isdir(WS) else []:
        p = os.path.join(WS, n, "meta.json")
        if os.path.exists(p):
            out.append(json.load(open(p, encoding="utf-8")))
    return out


def status():
    return {"sadtalker": os.path.exists(os.path.join(ROOT, "weights", "checkpoints", "SadTalker_V0.0.2_256.safetensors")) and os.path.exists(os.path.join(ST, "inference.py")),
            "f5": os.path.exists(os.path.join(ROOT, "venv", "bin", "f5-tts_infer-cli")), "tts_server": bool(TTS), "device": DEVICE,
            "f5_korean": bool(F5_CKPT)}


HTML = open(os.path.join(ROOT, "ui.html"), encoding="utf-8").read()

# ── 저작권 표기 (LICENSE·NOTICE 참고) ─────────────────────────────────────
_SIG = __import__("base64").b64decode("wqkgMjAyNiDquYDrj5nso7wgwrcgZG9uZ2p1a2ltLmRldkBnbWFpbC5jb20=").decode()
_SIG_A = __import__("base64").b64decode("RG9uZ0p1IEtpbSA8ZG9uZ2p1a2ltLmRldkBnbWFpbC5jb20+").decode()


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
        m = re.fullmatch(rf"/api/runs/({RUN_RE})/final\.mp4", p)
        if m:
            f = os.path.join(WS, m.group(1), "final.mp4")
            if os.path.exists(f):
                return self._send(open(f, "rb").read(), "video/mp4")
        self._send(signed(HTML).encode(), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/run":
            return self._send({"error": "not found"}, code=404)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
