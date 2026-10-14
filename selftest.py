#!/usr/bin/env python3
"""LLM·모델 없이 파이프라인만: 가짜 TTS(사인파) + 가짜 립싱크(ffmpeg 색상 소스) → 동의 게이트·산출물·목록·상대경로.  python3 selftest.py"""
import os, shutil, subprocess, tempfile, wave, struct, math
os.environ["WORKSPACE"] = tempfile.mkdtemp()
import app

def fake_tts(text, out_wav):
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(24000)))
    return "fake-server"

def fake_lipsync(photo, wav, d, emit):
    out = os.path.join(d, "gen.mp4"); emit({"log": "fake 50%"})
    assert subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=1", "-i", wav, "-shortest", out]).returncode == 0
    return out

real_tts = app.tts
app.tts, app.lipsync = fake_tts, fake_lipsync
d = tempfile.mkdtemp(); photo = os.path.join(d, "p.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=white:s=64x64:d=1", "-frames:v", "1", photo], check=True)
# 동의 없으면 거부
try: app.generate(photo, "안녕", consent=False); raise SystemExit("동의 게이트 실패")
except ValueError: pass
events = []
m = app.generate(photo, "안녕하세요", consent=True, emit=events.append)
assert m["engine"] == "fake-server" and os.path.exists(os.path.join(app.WS, m["run_id"], "final.mp4"))
assert [e["stage"] for e in events if "stage" in e] == ["tts", "lipsync"] and any("log" in e for e in events)
m2 = app.generate(photo, "두 번째", consent=True)
assert m2["engine"] == "fake-server" and app.list_runs()[0]["run_id"] == m2["run_id"]
assert "src=\"/" not in app.HTML and "fetch('/" not in app.HTML and "'/api/" not in app.HTML, "ui.html 에 절대경로"
assert set(app.status()) >= {"sadtalker", "tts_server", "s2v"} and "f5" not in app.status()
# 목소리는 TTS 도구만: 꺼져 있으면 알아듣기 쉬운 한국어 오류
_url, app.TTS = app.TTS, "http://127.0.0.1:9/v1"  # 아무도 안 듣는 포트
try: real_tts("안녕", os.path.join(d, "x.wav")); raise SystemExit("TTS 꺼짐 오류 실패")
except RuntimeError as e: assert "TTS 도구(tts-local)를 켜 주세요" in str(e), e
app.TTS = _url
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
# 사용자별 폴더(포털 X-Agent-User): alice 기록은 alice 폴더에만, bob 은 목록·열기·지우기 모두 못 함, owner 목록에 u/ 안 섞임
def req(path, user, body=None):
    r = urllib.request.Request(U + path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json", "X-Agent-User": user})
    try:
        with urllib.request.urlopen(r) as x: return x.status, x.read()
    except urllib.error.HTTPError as e: return e.code, e.read()
ma = app.generate(photo, "앨리스", consent=True, ws=os.path.join(app.WS, "u", "alice"))
assert os.path.exists(os.path.join(app.WS, "u", "alice", ma["run_id"], "final.mp4"))
assert [x["run_id"] for x in json.loads(req("/api/runs", "alice")[1])] == [ma["run_id"]] and json.loads(req("/api/runs", "bob")[1]) == []
assert ma["run_id"] not in [x["run_id"] for x in app.list_runs()]
assert req(f"/api/runs/{ma['run_id']}/final.mp4", "bob")[0] == 404 and req(f"/api/runs/{ma['run_id']}/final.mp4", "alice")[0] == 200
assert req("/api/runs/delete", "bob", {"run_ids": [ma["run_id"]]})[0] == 404 and os.path.isdir(os.path.join(app.WS, "u", "alice", ma["run_id"]))
assert req("/api/speak", "bob", {"run_id": ma["run_id"], "text": "x"})[1].count(b"error") == 1
c, b = req("/api/runs/delete", "alice", {"run_ids": [ma["run_id"]]}); assert c == 200
assert req("/api/runs/restore", "bob", {"batch": json.loads(b)["batch"]})[0] == 400 and req("/api/runs/restore", "alice", {"batch": json.loads(b)["batch"]})[0] == 200
srv.shutdown()
# 스튜디오 말하기 품질: best → Wan2.2-S2V(s2v), fast → SadTalker(lipsync). 영상 크기는 그림 비율·16의 배수
import json as _j
cid = "2026-01-02-c0de"; cd = os.path.join(app.WS, cid); os.makedirs(cd)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=white:s=64x64:d=1", "-frames:v", "1", os.path.join(cd, "cand_0.png")], check=True)
_j.dump({"run_id": cid, "kind": "character", "desc": "가상 인물", "shot": "half"}, open(os.path.join(cd, "meta.json"), "w"))
called = []
app.s2v = lambda photo, wav, out, prompt, emit: (called.append("s2v"), shutil.copy(fake_lipsync(photo, wav, cd, emit), out))
app.lipsync = lambda *a, **k: (called.append("sadtalker"), fake_lipsync(*a[:4]))[1]
assert app.speak(cid, 0, "안녕")["model"] == "Wan2.2-S2V-14B" and app.speak(cid, 0, "안녕", quality="fast")["model"] == "SadTalker" and called == ["s2v", "sadtalker"]
assert os.path.exists(os.path.join(cd, "talk.mp4")) and _j.load(open(os.path.join(cd, "meta.json")))["talk"]["model"] == "SadTalker"
for shot in app.SHOTS:
    for area, *_ in app.MOVE_Q.values():
        w, h = app.vsize(shot, area); W, H = app.SHOTS[shot][2]
        assert w % 16 == 0 and h % 16 == 0 and w * h <= area and abs(w / h - W / H) < 0.03, (shot, w, h)
# 포털 작업 알림(🔔): 환경변수 없으면 notify 는 False 이고 생성은 그대로 / 가짜 포털에 running → done|failed 가 토큰 헤더와 함께 간다
import base64, portal_jobs
from http.server import BaseHTTPRequestHandler
for k in ("PORTAL_JOBS_URL", "PORTAL_JOBS_TOKEN", "PORTAL_JOBS_TOOL"): os.environ.pop(k, None)
assert portal_jobs.notify("alice", "running", id="x") is False
GOT = []
class FakePortal(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        GOT.append((self.headers.get("X-Portal-Jobs-Token"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
portal = ThreadingHTTPServer(("127.0.0.1", 0), FakePortal); threading.Thread(target=portal.serve_forever, daemon=True).start()
srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H); threading.Thread(target=srv.serve_forever, daemon=True).start()
U = f"http://127.0.0.1:{srv.server_address[1]}"
def sse(path, body, user="owner"):
    _, b = req(path, user, body); return [json.loads(l[6:]) for l in b.decode().split("\n\n") if l.startswith("data: ")]
pb = {"consent": True, "text": "알림", "photo_name": "p.png", "photo_b64": base64.b64encode(open(photo, "rb").read()).decode()}
assert "done" in sse("/api/run", pb)[-1] and GOT == []  # 꺼져 있어도 도구는 그대로
os.environ.update(PORTAL_JOBS_URL=f"http://127.0.0.1:{portal.server_address[1]}/api/internal/jobs", PORTAL_JOBS_TOKEN="tok-test", PORTAL_JOBS_TOOL="avatar-local")
assert portal_jobs.notify("alice", "queued", id="j1", title="t", link="/?run=x", message="m", progress=-5) and portal_jobs.flush()
assert GOT[-1] == ("tok-test", {"tool": "avatar-local", "user": "alice", "status": "queued", "title": "t", "link": "/?run=x", "message": "m", "id": "j1", "progress": 0}), GOT
r1 = sse("/api/run", pb)[-1]["done"]; r2 = sse("/api/speak", {"run_id": cid, "idx": 0, "text": "안녕", "quality": "fast"})[-1]["done"]
sse("/api/move", {"run_id": "2026-01-01-ffff", "action": "손 흔들기"}); portal_jobs.flush()
st = [g[1] for g in GOT[1:]]
assert [x["status"] for x in st] == ["running", "done"] * 2 + ["running", "failed"], st
assert st[0]["id"] == st[1]["id"] != st[2]["id"] and st[1]["link"] == f"/?run={r1['run_id']}" and st[1]["title"] == "내 얼굴 아바타" and st[1]["user"] == "owner"
assert st[2]["link"] == st[3]["link"] == f"/?run={cid}" and st[3]["title"] == "말하는 영상" and st[3]["message"] == "SadTalker"
assert st[5]["title"] == "동작 영상" and app.WS not in st[5]["message"] and all(g[0] == "tok-test" for g in GOT)
# 알림 링크 /?run=<ID> 가 여는 기록: 내 것만
assert json.loads(req(f"/api/runs/{cid}", "owner")[1])["kind"] == "character" and req(f"/api/runs/{cid}", "bob")[0] == 404
srv.shutdown(); portal.shutdown()
for k in ("PORTAL_JOBS_URL", "PORTAL_JOBS_TOKEN", "PORTAL_JOBS_TOOL"): os.environ.pop(k, None)
shutil.rmtree(d); print("selftest OK")
# 저작권 표기: 서버가 화면에 붙이는 코드가 있어야 한다 (LICENSE·NOTICE)
_src = open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "app.py"), encoding="utf-8").read()
assert "wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==" in _src and "signed(" in _src and "X-Author" in _src, "저작권 표기 누락"


# GPU 후보: 부모 후보가 0,1,2 면 3(운영 vLLM 전용)을 절대 고르지 않음 — nvidia-smi 출력은 가짜, 예약 장부는 임시 폴더
import gpu_pick as _gp, os as _os, tempfile as _tf, shutil as _sh
_smi0, _dir0, _env0 = _gp._smi, _gp.RES_DIR, dict(_os.environ)
try:
    _gp._smi = lambda: "0, 1000, 90\n1, 2000, 90\n2, 3000, 90\n3, 90000, 0\n"  # 3 이 가장 한가해 보이게
    _gp.RES_DIR = _tf.mkdtemp()
    for _k in ("GPU_POOL", "GPU_RESERVED", "VLLM_GPUS"): _os.environ.pop(_k, None)
    for _res in ("none", "3"):
        _os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2", GPU_RESERVED=_res)
        assert [g["index"] for g in _gp.gpus()] == ["0", "1", "2"]
        assert _gp.pick(0, hold_s=0)["index"] == "2" and _gp.torch_device(_gp.pick(0, hold_s=0)) == "cuda:2"
        assert _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == "0,1,2" == _gp.env_for(None, _os.environ)["CUDA_VISIBLE_DEVICES"]
    # 셸(~/.zshrc)에서 2,3 을 물려받아도 예약된 3 은 빼고 2 만 — 보이는 번호는 cuda:0
    _os.environ.update(CUDA_VISIBLE_DEVICES="2,3", GPU_RESERVED="3")
    assert _gp.pick(0, hold_s=0)["index"] == "2" and _gp.torch_device(_gp.pick(0, hold_s=0)) == "cuda:0"
    assert _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == "2"
    _os.environ.pop("GPU_RESERVED"); _os.environ["VLLM_GPUS"] = "3"  # GPU_RESERVED 없으면 VLLM_GPUS
    assert _gp.pick(0, hold_s=0)["index"] == "2"
    _os.environ["GPU_POOL"] = "0,1"  # 교집합이 비면 GPU 를 하나도 안 보임(CPU)
    assert _gp.pick(0, hold_s=0) is None and _gp.pool_env(_os.environ)["CUDA_VISIBLE_DEVICES"] == ""
    if hasattr(app, "worker_env"):  # 작업자(그림·영상) 프로세스 환경
        assert app.worker_env({"CUDA_VISIBLE_DEVICES": "0,1,2", "GPU_RESERVED": "none"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "0,1,2"
        assert app.worker_env({"CUDA_VISIBLE_DEVICES": "2,3", "GPU_RESERVED": "3"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "2"
        assert app.worker_env({"GPU_POOL": "0,1,2,3", "VLLM_GPUS": "3"}, ["0", "1", "2", "3"])["CUDA_VISIBLE_DEVICES"] == "0,1,2"
finally:
    _sh.rmtree(_gp.RES_DIR, ignore_errors=True); _gp._smi, _gp.RES_DIR = _smi0, _dir0; _os.environ.clear(); _os.environ.update(_env0)
print("selftest GPU 후보 OK — 0,1,2 에서 3 안 고름")
