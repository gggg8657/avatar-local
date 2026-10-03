# ponytail: SadTalker 가 numpy 1.24+ 에서 깨지는 한 줄 (inhomogeneous array) — clone 뒤 1회 적용. 멱등.
import os, sys
p = os.path.join(sys.argv[1], "src", "face3d", "util", "preprocess.py")
s = open(p, encoding="utf-8").read()
s2 = s.replace("trans_params = np.array([w0, h0, s, t[0], t[1]])", "trans_params = np.array([w0, h0, s, float(t[0]), float(t[1])])")
open(p, "w", encoding="utf-8").write(s2); print("patched" if s2 != s else "already")
