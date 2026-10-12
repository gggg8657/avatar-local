# 제3자 구성요소 고지 (THIRD_PARTY_NOTICES)

avatar-local 의 원저작물(코드·UI·프롬프트·문서)은 DongJu Kim (gggg8657) 이 작성했고 MIT 라이선스입니다(`LICENSE`, `NOTICE`).
아래 제3자 구성요소는 **각자의 라이선스**를 따르며 이 저장소의 라이선스로 바뀌지 않습니다.
조사일 2026-10-12 · 1차 출처(설치된 패키지 메타데이터, vendor 의 LICENSE 파일, HF API `cardData.license`, GitHub LICENSE)로 확인. 법률 판단이 아니며, 상업 이용 전에는 원문을 다시 확인하세요.

> ⚠️ 연구·내부 시험용 프로토타입입니다(배포·상업 서비스용 아님).

## 1. 내려받아 실행하는 코드 (저장소에 없음, `setup.sh` 가 `vendor/` 로 받음, 수정 없음)

| 이름 | 버전 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|---|
| SadTalker | commit `cd4c046` | 말하기(빠르게)·내 얼굴 아바타 립싱크 | Apache-2.0 (Tencent) | https://github.com/OpenTalker/SadTalker | 조건부 | 원문 `licenses/SadTalker-Apache-2.0.txt`. README Disclaimer: 초상권 침해 등 금지, Tencent 이름·로고 사용 금지. 학습에 BFM·Wav2Lip 사용 언급 → [#1](https://github.com/gggg8657/avatar-local/issues/1) |
| Wan2.2 (공식 코드) | commit `1ea34ff` | 말하기(최고 품질, S2V-14B) | Apache-2.0 | https://github.com/Wan-Video/Wan2.2 | 가능(고지) | 원문 `licenses/Wan2.2-Apache-2.0.txt` |

`shim/` 은 이 저장소 작성 코드(호환 패치·스텁). `shim/gfpgan` 은 import 만 통과시키는 스텁으로 **GFPGAN 코드·가중치는 쓰지 않습니다**.

## 2. 내려받는 모델 가중치 (저장소·git 에 없음, 처음 쓸 때 받음)

| 이름 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|
| Qwen/Qwen-Image-2512 | 캐릭터 그림 | Apache-2.0 | https://huggingface.co/Qwen/Qwen-Image-2512 | 가능(고지) | [#4](https://github.com/gggg8657/avatar-local/issues/4) |
| Wan-AI/Wan2.2-I2V-A14B-Diffusers | 그림 → 움직이는 영상 | Apache-2.0 | https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers | 가능(고지·이용 조건) | 모델 카드: 법 위반·개인에게 해·허위정보 금지 · [#4](https://github.com/gggg8657/avatar-local/issues/4) |
| Wan-AI/Wan2.2-S2V-14B | 음성 → 말하는 영상 | Apache-2.0 | https://huggingface.co/Wan-AI/Wan2.2-S2V-14B | 가능(고지·이용 조건) | 위와 같음 |
| SadTalker 가중치 (`SadTalker_V0.0.2_256.safetensors`, `mapping_00109/00229-model.pth.tar`) | 립싱크 | 저장소 Apache-2.0 / HF `vinthony/SadTalker` 카드 MIT — 가중치 전용 문서 미확인 | https://github.com/OpenTalker/SadTalker/releases/tag/v0.0.2-rc | 미확인 | [#1](https://github.com/gggg8657/avatar-local/issues/1) |
| facexlib `detection_Resnet50_Final.pth` (RetinaFace) | 얼굴 검출 | MIT (원 출처 biubug6/Pytorch_Retinaface) | https://github.com/xinntao/facexlib | 미확인 | 학습 데이터 WIDER FACE 조건 확인 필요 · [#1](https://github.com/gggg8657/avatar-local/issues/1) |
| facexlib `alignment_WFLW_4HG.pth` (AdaptiveWingLoss) | 얼굴 랜드마크 | Apache-2.0 (원 출처) | https://github.com/protossw512/AdaptiveWingLoss | 미확인 | 학습 데이터 WFLW 조건 확인 필요 · [#1](https://github.com/gggg8657/avatar-local/issues/1) |
| face_alignment FAN 2D | 얼굴 랜드마크 | BSD-3-Clause (코드) | https://github.com/1adrianb/face-alignment | 미확인 | 가중치 학습 데이터 조건 확인 필요 · [#1](https://github.com/gggg8657/avatar-local/issues/1) |

`pack.sh`(폐쇄망 번들)는 SadTalker 가중치·보조 모델 캐시·휠을 **함께 묶습니다** — 번들은 내부 반입용이며 외부에 넘기면 재배포가 됩니다.

## 3. 설치하는 파이썬 패키지 (`requirements.txt` + torch, 설치된 버전 기준)

| 이름 | 버전 | 라이선스 | 상업 이용 | 비고 |
|---|---|---|---|---|
| torch / torchaudio / torchvision | 2.14.1 / 2.11.0 / 0.29.1 | BSD-3 계열 (torch: Apache-2.0·BSD 등 복합) | 가능 | |
| numpy | 1.26.4 | BSD-3-Clause | 가능 | |
| scipy | 1.17.1 | BSD-3-Clause | 가능 | |
| scikit-image | 0.26.0 | BSD-3-Clause | 가능 | |
| face_alignment | 1.5.0 | BSD-3-Clause | 가능 | 가중치는 2절 |
| facexlib | 0.3.0 | MIT (GitHub) / Apache-2.0 (PyPI 메타데이터) | 가능 | 가중치는 2절 |
| kornia | 0.8.3 | Apache-2.0 | 가능 | |
| imageio | 2.38.0 | BSD-2-Clause | 가능 | |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause (래퍼) — **휠 동봉 ffmpeg 7.0.2 는 GPLv3 빌드** | 조건부 | 재배포 시 GPL 의무 · [#5](https://github.com/gggg8657/avatar-local/issues/5) |
| av (PyAV) | 18.1.0 | BSD-3-Clause — 동봉 FFmpeg 라이브러리는 LGPL/GPL | 조건부 | [#5](https://github.com/gggg8657/avatar-local/issues/5) |
| librosa / resampy | 0.11.0 / 0.4.3 | ISC | 가능 | |
| pydub | 0.25.1 | MIT | 가능 | |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | 가능 | |
| yacs | 0.1.8 | Apache-2.0 | 가능 | |
| PyYAML | 6.0.3 | MIT | 가능 | |
| joblib | 1.6.0 | BSD-3-Clause | 가능 | |
| safetensors | 0.8.0 | Apache-2.0 | 가능 | |
| opencv-python | 4.11.0.86 | Apache-2.0 | 가능 | |
| huggingface_hub | 0.36.2 | Apache-2.0 | 가능 | |
| pyarrow | 20.0.0 | Apache-2.0 | 가능 | |
| ftfy | 6.3.1 | Apache-2.0 | 가능 | |
| diffusers | 0.37.0 | Apache-2.0 | 가능 | |
| transformers / accelerate | 4.57.6 / 1.15.0 | Apache-2.0 | 가능 | diffusers 의존 |
| decord | 0.6.0 | Apache-2.0 | 가능 | Wan2.2 가 import |
| easydict | 1.13 | **LGPL-3.0** | 조건부 | 수정 없이 동적 import · [#5](https://github.com/gggg8657/avatar-local/issues/5) |
| dashscope | 1.27.7 | Apache-2.0 | 가능 | Wan2.2 import 용(외부 API 호출은 안 함) |
| peft | 0.21.2 | Apache-2.0 | 가능 | |

## 4. 외부 프로그램·서비스 (동봉 안 함)

| 이름 | 쓰임 | 라이선스 | 상업 이용 | 비고 |
|---|---|---|---|---|
| ffmpeg (시스템) | 영상·음성 합치기, H.264 인코딩 | LGPL/GPL (이 서버 빌드는 `--enable-gpl`) | 조건부 | 별도 프로세스로 실행만 함. H.264 특허 · [#5](https://github.com/gggg8657/avatar-local/issues/5) |
| tts-local (MeloTTS, MIT) | 목소리 — 포털의 다른 도구를 HTTP 로 호출 | MIT | 가능 | 이 도구에 TTS 모델 없음, 목소리 복제 없음 |
| LLM (OpenAI 호환 API) | 그림·동작 프롬프트 다듬기 | 모델마다 다름 | 확인 | 코드 기본값 `qwen3:8b`(Qwen3, Apache-2.0), 포털 기본 배포 Ollama(MIT) + Google Gemma `gemma4:31b`(Gemma 이용약관) |
| persona-local | 만든 캐릭터를 넘겨 등록 | 별도 저장소 | — | 그 도구의 모델(예: Realistic Vision, CreativeML OpenRAIL-M)은 이 도구가 쓰지 않음 — persona-local 의 THIRD_PARTY_NOTICES 참고 |

## 5. 제거한 구성요소

| 이름 | 라이선스 | 상태 |
|---|---|---|
| F5-TTS 한국어 가중치(목소리 복제) | CC-BY-NC-4.0 (비상업) | 2026-10-06 `8921390` 에서 제거 · [#3](https://github.com/gggg8657/avatar-local/issues/3) (닫힘) |

## 확인 필요 (상업 이용 전)

- [#1](https://github.com/gggg8657/avatar-local/issues/1) SadTalker 가중치·얼굴 보조 모델(facexlib·face_alignment·BFM 파생) — 가중치·학습 데이터 조건 미확인. 회피: 상업 시 Wan2.2-S2V(Apache-2.0)만 쓰는 경로(코드 변경 필요), `pack.sh` 번들 외부 배포 금지.
- [#2](https://github.com/gggg8657/avatar-local/issues/2) 얼굴 사진·실사 캐릭터 영상 — 초상권·딥페이크·개인정보. 현재: 업로드 동의 체크, 목소리 복제 없음. 제안: AI 생성 표시, 실존 인물 이름 필터, 법무 검토.
- [#4](https://github.com/gggg8657/avatar-local/issues/4) Qwen-Image-2512·Wan2.2 — Apache-2.0(가능), 고지 의무와 모델 카드 이용 조건.
- [#5](https://github.com/gggg8657/avatar-local/issues/5) ffmpeg/imageio-ffmpeg/PyAV(GPL·LGPL 동봉 바이너리), easydict(LGPL-3.0), H.264 특허 — 번들 재배포 때만 문제. 회피: LGPL ffmpeg + openh264/VP9.
