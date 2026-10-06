# avatar-local — 내 얼굴·내 목소리 아바타 (토이)

> **한 줄 요약** — 본인 사진 한 장과 10초 목소리 샘플, 대사를 넣으면 내 얼굴이 내 목소리로 말하는 짧은 영상을 만드는 개인 토이입니다.
> 립싱크는 SadTalker(Apache-2.0), 목소리 복제는 F5-TTS(MIT), 전부 로컬. 회사 에이전트 페이지에는 숨김(hidden) 등록만 해 두고 개인 실험용으로 씁니다.

## 규칙 (README·UI 동의 체크 공통)
- **본인 얼굴·목소리, 또는 명시적으로 동의한 사람의 것만** 씁니다.
- 타인 영상에 얼굴을 바꿔 끼우는 기능(face swap)은 **없고, 넣지 않습니다**. 동의 없는 타인 합성은 성폭력처벌법(딥페이크)·초상권 위반입니다.
- 결과물 외부 배포 금지. 데이터는 `_workspace/`(또는 `WORKSPACE`)에만 남습니다.

## 실행
```bash
bash setup.sh                     # venv + SadTalker + 가중치(1GB) + F5-TTS → http://localhost:8777
TTS_BASE_URL=http://localhost:8771/v1 bash setup.sh      # 샘플 없을 때 tts-local 목소리로
python3 app.py --cli photo.jpg voice.wav "대사" -o out.mp4 --ref-text "샘플에서 말한 문장"
```
| env | 기본 | 설명 |
|---|---|---|
| `PORT` | 8777 | |
| `DEVICE` | cpu | `cuda` 권장. SadTalker 는 mps 미지원 → 맥은 cpu(256px 10초 영상에 수 분) |
| `TTS_BASE_URL` | 없음 | 목소리 샘플 없을 때 OpenAI 호환 TTS 서버(tts-local) |
| `F5_MODEL` / `F5_CKPT` | F5TTS_v1_Base | F5-TTS 는 **영어·중국어** 학습. 한국어 복제는 한국어 파인튜닝 ckpt 를 `F5_CKPT` 로 주거나 CosyVoice2(Apache, ko 지원)로 교체 |
| `WORKSPACE` | ./_workspace | 결과 폴더 |

## 흐름
사진 → (음성 샘플 → F5-TTS 복제 | TTS 서버) → wav → SadTalker(still, crop, 256) → mp4. UI 에서 녹음(MediaRecorder 15초 제한)·업로드 둘 다 가능.

## 폐쇄망 반입
`./pack.sh cu124` → `dist-offline/avatar-local-linux-x64-cu124.tar.gz` (wheels + SadTalker 소스 + 가중치 + 보조 모델 캐시 + INSTALL.md). 서버에서 `HF_HUB_OFFLINE=1 DEVICE=cuda bash setup.sh`.

## 이 맥에서 확인한 것 (PoC)
- `selftest.py` 통과(가짜 TTS·립싱크), `bash setup.sh` 처음부터 끝까지 통과(venv·SadTalker·가중치 988MB·F5-TTS·서버 기동), `/api/status` 전 엔진 준비됨.
- 실제 생성은 CPU(M1 Max, mps 미지원)에서 3초 대사 영상이 30분 안에 끝나지 않아 중단 → **GPU 서버에서 확인 필요**. 전처리(얼굴 정렬)와 TTS 단계까지는 정상 통과.
- SadTalker 호환 패치 2건을 `shim/`에 둠: numpy 1.24+ `np.float` 별칭(sitecustomize), `align_img` inhomogeneous array 한 줄(patch_sadtalker.py). setup.sh 가 자동 적용.

## 한계
- 한국어 목소리 복제는 기본 모델로는 어색함(영·중 모델). 샘플 없이 tts-local 목소리면 자연스럽지만 "내 목소리"는 아님.
- 256px 얼굴 크롭 기준 화질. 고화질(512)·보정(gfpgan)은 꺼 둠(basicsr 가 최신 torchvision 과 충돌).
- 정면·단일 얼굴 사진만. 안경·측면은 흔들림.

### 한국어 목소리 복제 (2026-10-06)
기본 F5-TTS 는 영·중 학습이라 한국어가 알아들을 수 없게 나온다. setup.sh 가 [team-lucid/F5-TTS-ko](https://huggingface.co/team-lucid/F5-TTS-ko)(Apache-2.0, 한글 자모 어휘)를 받아
`scripts/f5_ko.py` 로 `models/f5-ko/`(model.safetensors + vocab.txt)로 변환하고, app 은 이게 있으면 기본으로 쓰며 글을 자모(NFD)로 풀어 넣는다.
GPU 에서 한 문장 약 30초, 받아쓰기로 확인한 문장이 목표와 일치. 오디오 읽기(torchcodec)에 FFmpeg 공유 라이브러리가 필요해 없으면 conda 로 `~/.local/ffmpeg-shared` 에 깐다(`FFMPEG_LIB_DIR`).
