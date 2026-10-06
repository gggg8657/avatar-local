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
| GPU 고르기 | 자동 | 고정 번호 없음(`gpu_pick.py`): 목소리 합성·립싱크는 실행마다, 스튜디오 그림(Qwen-Image-2512 약 60GB)·영상(Wan2.2 I2V-A14B 약 48GB, CPU offload)은 모델을 올릴 때마다, 말하기(Wan2.2-S2V 약 60GB)는 실행마다 여유 메모리가 가장 큰 GPU. `GPU_POOL=2,3` 후보 제한, `GPU_IDLE_UNLOAD_S`(600) 동안 안 쓰면 스튜디오 모델을 내림, `STUDIO_DEVICE=cuda:1` 이면 고정 |
| `TTS_BASE_URL` | 없음 | 목소리 샘플 없을 때 OpenAI 호환 TTS 서버(tts-local) |
| `F5_MODEL` / `F5_CKPT` | F5TTS_v1_Base | F5-TTS 는 **영어·중국어** 학습. 한국어 복제는 한국어 파인튜닝 ckpt 를 `F5_CKPT` 로 주거나 CosyVoice2(Apache, ko 지원)로 교체 |
| `WORKSPACE` | ./_workspace | 결과 폴더 |

## 흐름
사진 → (음성 샘플 → F5-TTS 복제 | TTS 서버) → wav → SadTalker(still, crop, 256) → mp4. UI 에서 녹음(MediaRecorder 15초 제한)·업로드 둘 다 가능.

## 가상 캐릭터 스튜디오 — 모델·품질·시간 (2026-10-06, H100 NVL 96GB 1장 실측)
| 단계 | 최고 품질 (기본) | 빠르게 |
|---|---|---|
| 그림 | [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) 50단계, 장당 약 40초 (처음 로드 +20초), VRAM 약 60GB | — |
| 말하기 | [Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) — 음성에 맞춰 입·표정·고개·손(상반신)까지. 5초 조각마다 약 23~25분(40단계), VRAM 최고 약 60GB | SadTalker — 얼굴만, 수십 초 |
| 움직이기 | [Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) 720p급(상반신 816×1104) 81프레임(16fps, 5초) 40단계, 약 32분 | 같은 모델 480p급(상반신 544×720) 30단계, 약 7분, VRAM 약 31GB |

- 말하기 최고 품질은 diffusers 에 아직 파이프라인이 없어 공식 코드([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), `setup.sh` 의 `WAN_COMMIT` 으로 고정)를 `vendor/Wan2.2` 에 받아
  `generate.py --task s2v-14B --size 1024*704 --offload_model True --convert_model_dtype` 를 서브프로세스로 돌린다(SadTalker 와 같은 방식). 가중치는 처음 쓸 때 HF 캐시로 받는다(`S2V_CKPT` 로 폴더 지정 가능).
  공식 코드는 flash_attn 을 꼭 쓰는데 이 서버엔 nvcc 가 없어 빌드할 수 없어서, `shim/wan/flash_attn` 대역(torch SDPA)을 그 서브프로세스에만 PYTHONPATH 로 넣는다(vendor 코드는 수정하지 않음).
- 목소리는 그대로 F5-TTS-ko(샘플이 있을 때) 또는 tts-local. S2V 는 대사 5초마다 조각(clip)을 하나씩 더 만들어 이어 붙인다 → 5초를 넘으면 시간이 두 배.
- 움직이기는 diffusers `WanImageToVideoPipeline` + `enable_model_cpu_offload` — 14B 전문가 둘(고노이즈·저노이즈)을 CPU 에 두고 쓰는 차례에만 GPU 로(화질 같음, 시간 차이 미미). 모델 카드 권장값: guidance 3.5, 40단계, 16fps.
- 그림 크기는 모델 카드 권장 비율(얼굴 1328², 상반신 1104×1472, 전신 928×1664), 영상은 그림 비율 그대로.
- 작업 프로세스는 `studio.log` 에 작업마다 `끝 — N초, GPU n 최고 X GB` 를 남긴다.

## 폐쇄망 반입
`./pack.sh cu124` → `dist-offline/avatar-local-linux-x64-cu124.tar.gz` (wheels + SadTalker·Wan2.2 소스 + 가중치 + 보조 모델 캐시 + INSTALL.md). 스튜디오 모델(HF 캐시 약 235GB: Qwen-Image-2512·Wan2.2 I2V-A14B·S2V-14B)은 따로 반입. 서버에서 `HF_HUB_OFFLINE=1 DEVICE=cuda bash setup.sh`.

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

## 출처·감사 (Credits)

- [SadTalker](https://github.com/OpenTalker/SadTalker) (Apache-2.0) — 립싱크. setup 때 `vendor/` 로 받아 쓰며 수정하지 않음(호환 패치는 `shim/`)
- [F5-TTS](https://github.com/SWivid/F5-TTS) (코드 MIT, F5TTS_v1_Base 가중치 CC-BY-NC-4.0) — 목소리 복제
- [team-lucid/F5-TTS-ko](https://huggingface.co/team-lucid/F5-TTS-ko) (Apache-2.0) — 한국어 F5 체크포인트
- 가상 캐릭터 스튜디오 (모델은 처음 쓸 때 받고 재배포하지 않음):
  - 그림: [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) (Apache-2.0) — [diffusers](https://github.com/huggingface/diffusers) (Apache-2.0) 로 실행
  - 움직이기: [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) (Apache-2.0)
  - 말하기(최고 품질): [Wan-AI/Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B) (Apache-2.0) — 공식 코드 [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) (Apache-2.0) 를 `vendor/` 로 받아 수정 없이 실행
  - 말하기(빠르게): 위 SadTalker 를 그대로 유지
- face_alignment (BSD-3), facexlib (MIT), PyTorch/torchvision (BSD)
- **LLM 실행** — OpenAI 호환 API 로 호출합니다(모델 가중치는 동봉하지 않음). 기본 배포는 [Ollama](https://github.com/ollama/ollama) (MIT) 위의 Google [Gemma](https://ai.google.dev/gemma) `gemma4:31b` — 모델 이용 조건은 Gemma 배포처 참고.
- 이 도구는 [agent-page-portal](https://github.com/gggg8657/agent-page-portal) 에 연결해 쓰도록 만들었습니다(단독 실행도 됨).

저작권 표기·전체 목록은 `NOTICE` 를 보세요.
