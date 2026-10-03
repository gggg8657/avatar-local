# ponytail: SadTalker 가 np.float/np.int 옛 별칭을 씀 (numpy 1.24+ 에서 제거) — PYTHONPATH=shim 로 자동 로드
import numpy as _np
_np.float, _np.int = float, int
