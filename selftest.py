#!/usr/bin/env python3
"""LLM·모델 없이 파이프라인만: 가짜 TTS(사인파) + 가짜 립싱크(ffmpeg 색상 소스) → 동의 게이트·산출물·목록·상대경로.  python3 selftest.py"""
import os, shutil, subprocess, tempfile, wave, struct, math
os.environ["WORKSPACE"] = tempfile.mkdtemp()
import app

def fake_tts(text, out_wav, ref_wav=None, ref_text=""):
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(24000)))
    return "fake-clone" if ref_wav else "fake-server"

def fake_lipsync(photo, wav, d, emit):
    out = os.path.join(d, "gen.mp4"); emit({"log": "fake 50%"})
    assert subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=1", "-i", wav, "-shortest", out]).returncode == 0
    return out

app.tts, app.lipsync = fake_tts, fake_lipsync
d = tempfile.mkdtemp(); photo = os.path.join(d, "p.png"); voice = os.path.join(d, "v.wav")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=white:s=64x64:d=1", "-frames:v", "1", photo], check=True)
fake_tts("", voice)
# 동의 없으면 거부
try: app.generate(photo, "안녕", voice, consent=False); raise SystemExit("동의 게이트 실패")
except ValueError: pass
events = []
m = app.generate(photo, "안녕하세요", voice, "hi", consent=True, emit=events.append)
assert m["engine"] == "fake-clone" and os.path.exists(os.path.join(app.WS, m["run_id"], "final.mp4"))
assert [e["stage"] for e in events if "stage" in e] == ["voice", "tts", "lipsync"] and any("log" in e for e in events)
m2 = app.generate(photo, "두 번째", None, consent=True)
assert m2["engine"] == "fake-server" and app.list_runs()[0]["run_id"] == m2["run_id"]
assert "src=\"/" not in app.HTML and "fetch('/" not in app.HTML and "'/api/" not in app.HTML, "ui.html 에 절대경로"
assert set(app.status()) >= {"sadtalker", "f5", "tts_server"}
shutil.rmtree(d); print("selftest OK")
# 저작권 표기: 서버가 화면에 붙이는 코드가 있어야 한다 (LICENSE·NOTICE)
_src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "app.py"), encoding="utf-8").read()
assert "wqkgMjAyNiDquYDrj5nso7wgwrcgZG9uZ2p1a2ltLmRldkBnbWFpbC5jb20=" in _src and "signed(" in _src and "X-Author" in _src, "저작권 표기 누락"

