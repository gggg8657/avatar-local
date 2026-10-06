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
# 갤러리 지우기: 휴지통(.trash/<묶음>/)으로 옮김 · 경로 조작/링크 거부 · 되살리기 · HTTP
import json, threading, urllib.request, urllib.error
from http.server import ThreadingHTTPServer
for bad in (["../etc"], ["2026-01-01-zzzz"], [m["run_id"], "../../x"], "abc", [], [None], ["/" + m["run_id"]]):
    try: app.trash_runs(bad); raise SystemExit(f"경로 검사 실패: {bad!r}")
    except ValueError: pass
assert os.path.isdir(os.path.join(app.WS, m["run_id"])), "틀린 ID 가 섞이면 아무것도 옮기지 않아야 함"
link = os.path.join(app.WS, "2026-01-01-abcd"); os.symlink(d, link)  # WS 밖을 가리키는 링크
try: app.trash_runs(["2026-01-01-abcd"]); raise SystemExit("링크 거부 실패")
except ValueError: pass
assert os.path.isdir(d); os.remove(link)
r = app.trash_runs([m["run_id"], m2["run_id"]])
assert r["deleted"] == [m["run_id"], m2["run_id"]] and app.list_runs() == []
assert os.path.exists(os.path.join(app.TRASH, r["batch"], m["run_id"], "final.mp4")) and os.path.exists(os.path.join(app.TRASH, r["batch"], m2["run_id"], "meta.json"))
assert app.trash_runs([m["run_id"]]) == {"deleted": [], "missing": [m["run_id"]], "batch": None}
for bad in ("../x", "", None, r["batch"] + "/.."):
    try: app.restore_trash(bad); raise SystemExit(f"묶음 검사 실패: {bad!r}")
    except ValueError: pass
assert sorted(app.restore_trash(r["batch"])["restored"]) == sorted([m["run_id"], m2["run_id"]]) and len(app.list_runs()) == 2
srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H); threading.Thread(target=srv.serve_forever, daemon=True).start()
U = f"http://127.0.0.1:{srv.server_address[1]}"
def post(path, body):
    try:
        with urllib.request.urlopen(urllib.request.Request(U + path, json.dumps(body).encode(), {"Content-Type": "application/json"})) as x: return x.status, json.load(x)
    except urllib.error.HTTPError as e: return e.code, json.load(e)
assert post("/api/runs/delete", {"run_ids": ["../../etc/passwd"]})[0] == 400
assert post("/api/runs/delete", {"run_ids": ["2026-01-01-ffff"]})[0] == 404
c, j = post("/api/runs/delete", {"run_ids": [m2["run_id"]]}); assert c == 200 and j["deleted"] == [m2["run_id"]]
try: urllib.request.urlopen(U + f"/api/runs/{m2['run_id']}/final.mp4"); raise SystemExit("지운 파일이 아직 열림")
except urllib.error.HTTPError as e: assert e.code == 404
try: urllib.request.urlopen(urllib.request.Request(U + "/api/runs/..%2f..%2fetc", method="DELETE")); raise SystemExit("DELETE 경로 검사 실패")
except urllib.error.HTTPError as e: assert e.code == 404
assert post("/api/runs/restore", {"batch": j["batch"]}) == (200, {"restored": [m2["run_id"]]})
srv.shutdown()
shutil.rmtree(d); print("selftest OK")
# 저작권 표기: 서버가 화면에 붙이는 코드가 있어야 한다 (LICENSE·NOTICE)
_src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "app.py"), encoding="utf-8").read()
assert "wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==" in _src and "signed(" in _src and "X-Author" in _src, "저작권 표기 누락"

