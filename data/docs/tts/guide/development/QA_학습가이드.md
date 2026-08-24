# Qwen3-TTS 보고 대비 Q&A 학습 가이드 (v0.9.0 AutoLang & Polish)

> **용도**: 상급자 보고 전 예상 질문 대비 자기 학습용 문서  
> **기준**: 전체 /docs 11개 문서 및 최신 코드베이스 분석 기반 (v0.8.0 — `tts-fast-a` 분리, ITN 대정비, 회귀 테스트 포함)

---

## 목차

1. [솔루션 개요 / 왜 만들었나](#1-솔루션-개요)
2. [아키텍처 / 왜 이렇게 설계했나](#2-아키텍처)
3. [핵심 기술 선택 이유](#3-핵심-기술-선택-이유)
4. [통신 / 어떻게 동작하나](#4-통신--어떻게-동작하나)
5. [성능 / 수치로 답하기](#5-성능--수치로-답하기)
6. [보안 / 어떻게 지키나](#6-보안)
7. [장애 대응 / 스스로 고치나](#7-장애-대응--자가-치유)
8. [배포 / 납품은 어떻게 하나](#8-배포--납품)
9. [한계 / 아직 부족한 것](#9-한계--향후-계획)

---

<div style="page-break-after: always;"></div>

## 1. 솔루션 개요

---

**Q. 이 솔루션이 뭐야? 한 마디로 설명해봐.**

> **A.** 기업용 AICC(AI 콜센터) 환경에 특화된 **자체 구축형(On-Premise) 한국어 중심 다국어 TTS(Text-to-Speech) 서버 솔루션**입니다. 클라우드 API 의존 없이 사내망에서 완전히 독립적으로 구동되며, 32개 언어의 고품질 음성 합성을 실시간 스트리밍으로 제공합니다.

---

**Q. 왜 자체 구축했어? 그냥 외부 TTS API 쓰면 안 돼?**

> **A.** 3가지 이유입니다.
> 1. **보안/컴플라이언스**: 콜센터 대화 데이터, 고객 실명·계좌번호 등 민감 정보를 외부 서버에 전송하면 개인정보보호법 위반 리스크가 발생합니다.
> 2. **비용**: 콜센터는 하루 수십만 건 합성이 발생하는데, 외부 API는 건당 과금으로 장기 운용 시 비용이 급격히 증가합니다.
> 3. **커스터마이징**: 특정 도메인 전문 용어(금융, 의료 등) ITN 처리, 커스텀 보이스 클론 등을 외부 API로는 제어하기 어렵습니다.

---

**Q. 어떤 AI 모델을 쓰나?**

> **A.** 두 가지 엔진을 목적에 따라 혼용합니다.
> - **vLLM-Omni (Qwen3-TTS-0.6B)**: Alibaba의 대형 언어-오디오 모델. 한국어 포함 10개 언어 고음질 합성. GPU 필요.
> - **Supertonic-3**: 경량 다국어 엔진. 22개 언어 지원. CPU만으로 구동 가능.
> 두 엔진을 자동 라우팅하여 총 32개 언어를 커버합니다.

---

<div style="page-break-after: always;"></div>

## 2. 아키텍처

---

**Q. 전체 아키텍처를 설명해봐.**

> **A.** 3개 레이어로 구성됩니다.
> - **API Gateway Layer**: FastAPI 기반 웹서버. 인증, 배압 제어, ITN 전처리, 언어 감지 제어 분기, Redis 캐시, DSP 후처리 담당.
> - **Inference Engine Layer (v0.8.0~ 분리)**: vLLM-Omni 추론 워커(`tts-vllm-a/b`, GPU) + Supertonic 워커(`tts-fast-a`, CPU/ONNX) 별개 컨테이너로 격리. LAC 로드밸런서가 최소 활성 연결 기준으로 vLLM 워커에 분배, CB OPEN 시 자동 fallback.
> - **Data Layer**: Redis (캐시/세마포어/통계), PostgreSQL/SQLite (감사로그/계정/설정/목소리), 동적 Seeding 및 Volume 동기화 데몬.
>
> 클라이언트 → Nginx → FastAPI → (vLLM-a/b 또는 fast-a) + Redis/DB 순으로 흐릅니다.

---

**Q. 배포 에디션은 몇 가지이고, 차이가 뭐야?**

> **A.** 현재 유효 에디션은 **Internal(2)·External(4) 2종**입니다. 배포 규모에 따라 선택합니다.
>
> 현재 유효 에디션은 **2 Internal / 4 External** 2종 (1 Lite·3 Professional은 폐지→2/4). 단일 출처 [`../../deploy/공통_개념.md`](../../deploy/공통_개념.md).
>
> | 에디션 | 대상 | DB | 캐시 | HA |
> |:---|:---|:---|:---|:---|
> | **2 Internal** (현재) | 사내 소규모 | SQLite WAL | Redis | 없음 (Supertonic fallback) |
> | **4 External** | 대형 AICC | PostgreSQL | Redis | Nginx + vLLM 이중화(`USE_VLLM_B`) |
>
> 동일한 `deploy.sh` 스크립트로 에디션 선택만 바꿔 배포합니다.

---

**Q. GPU가 여러 개 있으면 어떻게 써?**

> **A.** 3가지 모드가 있습니다.
> - **단일 GPU (Mode 1)**: vLLM 하나가 GPU 1대 점유. 가장 단순.
> - **텐서 병렬 (Mode 2)**: GPU 2대에 모델을 쪼개 단일 vLLM 인스턴스가 사용. **단일 요청 지연 최소화** 목적.
> - **부하분산 (Mode 3)**: GPU 2대에 vLLM 인스턴스 2개를 독립 구동. **처리량(Throughput) 2배** + 한쪽 장애 시 자동 Failover.

---

<div style="page-break-after: always;"></div>

## 3. 핵심 기술 선택 이유

---

**Q. 왜 Redis를 썼어?**

> **A.** Redis는 이 솔루션에서 4가지 역할을 동시에 수행합니다.
> 1. **문장 캐시**: 동일 문장 재요청 시 추론 없이 50ms 이내로 즉시 반환 (TTFB 획기적 단축).
> 2. **분산 세마포어**: 멀티 워커 환경에서 동시 접속 채널 수를 원자적으로 카운트하여 GPU 과부하 방지.
> 3. **LAC 카운터**: `vllm:lac:{ip}:{port}` 키로 각 vLLM 인스턴스의 활성 연결 수를 실시간 추적, 최소 연결 워커로 요청 분배.
> 4. **텔레메트리 시계열 저장소**: 5초 주기 CPU/GPU/VRAM/채널 수 데이터 저장.

---

**Q. 왜 WebSocket을 썼어? HTTP로 하면 안 돼?**

> **A.** AICC 시나리오를 보면 알 수 있습니다. 상담원이 말을 하는 도중에 TTS가 멈추면 안 됩니다.  
> - **HTTP**: 합성이 다 끝나야 전체 오디오를 받습니다. 긴 문장이면 수초 지연 발생.
> - **WebSocket**: 추론 중 생성되는 PCM 오디오 조각을 실시간으로 Push. **첫 음성까지 약 2초, 이후 끊김 없이 스트리밍**.
>
> 양방향 채널이므로 클라이언트가 중간에 텍스트를 추가 전송하거나 `stop` 프레임으로 즉시 중단도 가능합니다.

---

**Q. SQLite에서 WAL 모드를 쓴다는데 WAL이 뭐야?**

> **A.** WAL(Write-Ahead Logging)은 SQLite의 저장 방식입니다.  
> - **기본 모드**: 쓰기 시 파일 전체에 락 → 멀티 프로세스 환경에서 충돌.
> - **WAL 모드**: 쓰기를 별도 로그 파일에 먼저 기록하고 나중에 병합 → **다중 워커(`TTS_WORKERS`, 현재 배포 4)가 동시에 쓰기를 해도 락 충돌 없음**.
>
> Internal(2) 에디션처럼 PostgreSQL 없이 SQLite만 쓰는 환경에서 필수입니다.

---

**Q. ITN이 뭐야?**

> **A.** ITN(Inverse Text Normalization, 역텍스트정규화)은 텍스트를 TTS에 입력하기 전에 **자연스럽게 읽힐 수 있도록 변환하는 전처리** 과정입니다.  
> 예시:
> - `2026-05-20` → `이천이십육년 오월 이십일`
> - `LLM` → `엘엘엠`
> - `0.85` → `영점팔오`
>
> 11개 도메인(금융, 의료, IT 등)별 전문 규칙이 있고, 런타임 중 규칙 교체 시 서비스 중단 없이 원자적으로 적용합니다(Copy-on-Write).

---

**Q. 서킷 브레이커(Circuit Breaker)가 뭐야?**

> **A.** 전기 차단기처럼 동작하는 장애 격리 패턴입니다.  
> - vLLM 추론 에러가 **연속 3회 이상** 발생하면 서킷이 **OPEN** (차단) 상태로 전환.
> - OPEN 상태에서는 신규 요청을 즉시 `503`으로 거절 (Fail-Fast). GPU를 더 이상 손상시키지 않음.
> - 관리자가 `/v1/admin/circuit-breaker/reset`으로 수동 복구하거나, 자동 복구 조건 충족 시 CLOSED로 복원.
>
> vLLM이 OPEN일 때 Supertonic(CPU 엔진)은 서킷 브레이커 대상 외라 계속 정상 서빙됩니다.

---

**Q. 배압(Backpressure) 제어가 뭐야?**

> **A.** GPU가 감당할 수 있는 요청 수를 초과하지 않도록 입구에서 막는 메커니즘입니다.
> - 동시 활성 채널이 임계치(기본: 16 × GPU수)를 넘으면 신규 요청에 즉시 `503` 반환.
> - Hysteresis 방식: 열림(OPEN) 16채널, 닫힘(CLOSE) 12채널로 차이를 두어 바운싱 방지.
> - 가중 계산: 스트리밍 1채널 = 1.0, 배치 1채널 = 2.0으로 VRAM 사용량이 많은 배치를 더 무겁게 계산.

---

<div style="page-break-after: always;"></div>

## 4. 통신 / 어떻게 동작하나

---

**Q. WebSocket 통신 흐름을 설명해봐.**

> **A.** 5단계로 진행됩니다.
> 1. 클라이언트가 `ws://서버/v1/tts/ws?token=...`으로 연결 요청.
> 2. 서버가 JWT 인증 확인 + VRAM 배압 가드 통과 → `101 Switching Protocols` 응답.
> 3. 클라이언트가 `{"type":"start", "voice_id":"...", "language":"ko"}` 시작 프레임 전송.
> 4. 이후 `{"type":"text", "text":"합성할 내용"}` 프레임을 실시간 전송 → 서버가 PCM 오디오 바이너리로 즉시 스트리밍 응답.
> 5. `{"type":"stop"}` 전송 → 서버가 잔여 오디오 마무리 후 `{"event":"audio_complete"}` 반환 및 세션 종료.

---

**Q. HTTP로도 합성할 수 있어?**

> **A.** 네, `POST /v1/tts/synthesize`로 단일 문장 합성이 가능합니다. 합성이 완전히 끝난 후 WAV 파일 전체를 한 번에 반환합니다. 실시간 스트리밍이 불필요한 배치 처리나 파일 저장 용도에 적합합니다.

---

**Q. 인증은 어떻게 해?**

> **A.** Bearer 토큰 방식을 사용합니다.
> - **HTTP**: `Authorization: Bearer {API Key 또는 JWT}` 헤더 전송.
> - **WebSocket**: HTTP 헤더 설정이 불가능한 환경을 위해 쿼리 파라미터(`?token=...`) 또는 `Sec-WebSocket-Protocol` 서브프로토콜 헤더로 대체 전송 가능.
>
> 비밀번호 변경 시 기존 발급된 JWT 전체가 즉시 무효화됩니다(`password_changed_at` 비교 Fail-Closed).

---

<div style="page-break-after: always;"></div>

## 5. 성능 / 수치로 답하기

---

**Q. 실제 성능이 어느 정도야? (L40S 기준)**

> | 엔진 | 첫 음성 지연(TTFB) | RTF | 비고 |
> |:---|:---|:---|:---|
> | vLLM (Qwen3) | **2.08초** | **0.39** | 여성 화자, 24kHz 네이티브 (HiFi 옵션 48kHz) |
> | Supertonic | **2.67초** | **0.45** | 남성 화자, 44.1kHz |
> | 캐시 히트 | **50ms 미만** | - | Redis 캐시 적중 시 즉시 반환 |
>
> **RTF(Real-Time Factor) 0.39** = 1초짜리 오디오를 0.39초 만에 생성. 실시간보다 약 2.5배 빠른 속도.

---

**Q. 동시에 몇 명이나 쓸 수 있어?**

> **A.** burst 실측 확정(0.6B · 2026-07-28) 기준 **TTFB p95 ≤500ms 로 동시 15, ≤1000ms 로 동시 30, 수용 한계 36**입니다 (초과분은 429 로 보호 — RTF 실링 ~40 에 맞춘 가드). 60시간 롱런으로 안정성 검증 완료. 정본: `reports/동시성_TTFB_스윕_결과_2026-07-27.md`.

---

**Q. 같은 문장을 두 번 요청하면 어떻게 돼?**

> **A.** Redis에서 캐시를 조회하여 **50ms 이내로 즉시 반환**합니다. GPU 추론이 전혀 발생하지 않습니다. 콜센터의 상용구("안녕하세요, OO 고객센터입니다") 같은 반복 문장에 극도로 효율적입니다. 관리자 대시보드에서 핵심 상용구를 미리 워밍업(Pre-caching)하면 첫 번째 요청부터도 캐시 히트가 가능합니다.

---

<div style="page-break-after: always;"></div>

## 6. 보안

---

**Q. 보안은 어떻게 되어 있어?**

> **A.** 5겹의 보안 레이어가 있습니다.
> 1. **인증/인가**: JWT + API Key 이중 인증, RBAC(admin/user 역할 분리), 비밀번호 변경 시 기존 JWT 전체 즉시 무효화.
> 2. **로그아웃 블랙리스트**: 로그아웃 토큰을 Redis에 등록하여 만료 전 재사용 차단.
> 3. **로그 마스킹**: 패스워드, API Key, 대용량 base64 오디오, URL 쿼리 파라미터 자동 마스킹. 로그 드라이버 유출 방지.
> 4. **감사 자가 감사**: 감사 로그 삭제 행위 자체를 `action="audit_delete"` 레코드로 강제 기록. 관리자도 증거 인멸 불가.
> 5. **ITN 입력 제한**: 대용량 ITN 규칙 등록을 통한 서버 자원 고갈 공격 방지 (빌트인 50K, 도메인 10K 등 상한 적용).

---

**Q. 납품 시 소스코드 보안은?**

> **A.** 두 가지 난독화 옵션이 있습니다.
> - **pyc 빌드**: Python 소스를 바이너리 `.pyc`로 컴파일. 소스 원문 제거.
> - **PyArmor 암호화**: AES 암호화 기반 기계어 난독화. 런타임 가드 동적 적재. 역컴파일 실질적 불가.
>
> `deploy-export.sh` 실행 시 난독화된 Docker 이미지 + 설치 스크립트가 `.tar.gz`로 패키징되어 오프라인 폐쇄망에서도 `./install.sh` 한 번으로 즉시 구동 가능합니다.

---

<div style="page-break-after: always;"></div>

## 7. 장애 대응 / 자가 치유

---

**Q. GPU가 죽으면 서비스도 죽어?**

> **A.** 아닙니다. 3단계 자가 치유가 동작합니다.
> 1. **서킷 브레이커 OPEN**: vLLM 연속 3회 에러 시 해당 워커 격리. 신규 요청은 다른 워커로 라우팅.
> 2. **VRAM 90% 도달 시 자동 차단**: 해당 GPU 워커를 일시 비활성화, 80% 이하 복구 시 자동 재활성화.
> 3. **vLLM 전체 불능 시**: Supertonic CPU 엔진이 배압 체크 우회 상태로 계속 서빙. 음질은 낮아지지만 서비스 무중단.

---

**Q. Redis가 죽으면?**

> **A.** 대부분 Fail-Open(가용성 우선)이나, rate-limit 은 예외로 보호를 유지합니다.
> - 캐시 기능이 꺼지고 매번 추론으로 처리 (성능 저하이나 서비스 유지).
> - LAC 밸런서가 동작 불가 → 로컬 메모리 기반 라운드-로빈으로 즉시 전환.
> - JWT 인증은 로컬 DB 조회로 대체하여 로그인/인증 기능 유지.
> - **Rate-limit 은 fail-open 이 아님** — 로그인/가입/API키·합성은 워커-로컬 인메모리 카운터로 폴백(`core/rate_limit_fallback.py`)해 무제한 요청 방지. me.py benchmark 는 fail-closed(503).
> - Redis 복구 후 `docker compose restart redis`로 즉각 정상화.

---

**Q. 무중단 업데이트가 가능해?**

> **A.** 네, `deploy-roll.sh`로 가능합니다.
> - vLLM 추론 엔진, Redis, DB는 **전혀 건드리지 않고** API 컨테이너만 교체.
> - 신규 API 컨테이너 헬스체크 500초 대기 → 통과 시 Nginx가 신규 컨테이너로 스왑 → 구버전 드롭.
> - 배포 다운타임: **기존 5분 → 약 10초**로 단축.

---

<div style="page-break-after: always;"></div>

## 8. 배포 / 납품

---

**Q. 처음 설치가 어려워?**

> **A.** `deploy.sh` 실행 후 17개 질문에 답하면 됩니다. 스크립트가 자동으로:
> - `nvidia-smi`로 GPU 정보 감지 및 VRAM 기반 배압 임계치 자동 계산.
> - HuggingFace 모델 캐시 경로 자동 탐색 및 안내.
> - 에디션, GPU 모드, 포트, 비밀번호, 난독화 강도 설정 후 Docker Compose 기동.
>
> 설정이 완료되면 `.env.prod` 파일에 전체 환경변수가 저장되어, 이후 재배포 시에는 Q1에서 "설정 재사용"만 선택하면 됩니다.

---

**Q. 폐쇄망(인터넷 없는 환경)에서도 돼?**

> **A.** 네, 가능합니다. `deploy-export.sh`가 생성하는 `qwen3-tts-delivery-v0.8.0.tar.gz` 패키지에는:
> - 오프라인 Docker 이미지 (vLLM-Omni + API 서버)
> - 환경 설정 템플릿 및 Nginx/DB 룰셋
> - 원클릭 설치 스크립트 (`install.sh`)
> - 32개국 다국어 시드 음원 (`config/samples/*.mp3`)
>
> 이 패키지를 폐쇄망 서버에 복사 후 `./install.sh`만 실행하면 인터넷 없이 완전 구동됩니다.

---

<div style="page-break-after: always;"></div>

## 9. 한계 / 향후 계획

---

**Q. 현재 부족한 점은 솔직하게 뭐야?**

> **A.** 현재 식별된 한계점 3가지입니다.
> 1. ~~**Supertonic이 메인 프로세스에 내장**~~ — **v0.8.0에서 해결**: `tts-fast-a` 컨테이너로 분리 완료. Python GIL 경합 해소.
> 2. **배압 제어가 하드 503**: 피크 트래픽 시 요청을 대기열에 넣지 않고 즉시 거절. → Priority Queue 도입이 다음 개선 과제. (단, vLLM CB OPEN 시 Supertonic fallback으로 503 발생 빈도는 크게 감소)
> 3. **로컬 파일 스토리지 의존**: 오디오 결과물이 컨테이너 로컬 디스크에 저장. 대용량 환경에서 디스크 고갈 리스크. → S3/MinIO 연동 필요.
> 4. **ITN 외래어 사전 한계**: 현재 약 1,300여 항목. 국립국어원 표준 외래어 표기(~5만 단어) 일괄 임포트 또는 LLM 기반 자동 제안 필요.

---

**Q. 앞으로 계획은?**

> **A.**
> 1. **~ 05.21**: Locust 30 VU 부하 테스트 최종 완료 및 P95 TTFB 지표 분석.
> 2. **05.26~05.27**: 서버 이전.
> 3. **05.27~05.30**: 사내 베타 서비스 및 실사용자 피드백 수집 (Soak Test 대체).
> 4. **06.01**: 프로덕션 릴리즈.

---

**Q. (v0.9.5 추가) 새로 개선된 다국어 프리셋 음원 시딩 및 볼륨 동기화 메커니즘이 무엇인가?**

> **A.** Docker 환경에서 Named Volume(`qwen3-tts-api-voices`)이 이미 활성화되어 있으면, 나중에 이미지 내에 새로 추가된 샘플 음원(`.mp3` 파일)이 마운트되지 못하는 Docker 볼륨 한계가 있었습니다.  
> 이를 해결하고자 `db/manager.py`에 자동화된 복사-동기화 로직을 주입했습니다. 컨테이너가 켜질 때 이미지 내 `/app/config/samples/`와 볼륨 `/app/voices_data/samples/`를 바이트 비교하여 누락된 중국어/일본어 음성을 백그라운드에서 실시간으로 강제 복사해 자가 치유합니다. 또한, 동시에 `voices.yaml`을 기준으로 SQLite/PostgreSQL 테이블에 새로운 보이스 메타데이터를 원자적으로 증분(Incremental) 주입하도록 설계했습니다.

---

**Q. (v0.9.0 갱신) 언어 자동 감지가 어떻게 진화했는가?**

> **A.** 단계별로:
> 1. **v0.9.5 초기**: regex 기반 6개 언어(ko/ja/zh/vi/en/ru) 자동 식별 + 별도 토글 카드 UI.
> 2. **v0.9.0 통합**: 자동 감지를 **LangSelect 드롭다운 최상단 항목**으로 흡수(Papago/DeepL 패턴). 트리거 라벨이 `🌐 자동 감지 · KO`로 인라인 표시되어 사용자가 한눈에 감지 결과 확인. 별도 토글 카드 제거 → 컨트롤 단순화.
> 3. **v0.9.0 라이브러리화**: regex → `tinyld` n-gram 라이브러리(~30KB)로 전환. **6개 → 32개 지원 언어 전체 식별**로 확장. 신뢰도 < 0.15 시 null 반환 + `synthAutoDetectFailed` 토스트 안내로 잘못된 자동 보정 방지.
> 4. **합성 시 reconcile**: 감지 결과로 `language`/`voice_id`/`engine` 자동 보정 + `toast.info`로 변경 통지. 사용자 선택 manual 모드는 그대로 존중.
> 5. **persistance**: `localStorage.tts_auto_detect_lang` (true/false). 미설정 시 관리자 글로벌 기본값 `default_auto_detect_lang` 적용.

---

**Q. (v0.9.0 추가) 음성 끝이 가끔 잘리는 문제는 어떻게 해결했는가?**

> **A.** 두 단계로 해결:
> 1. **vLLM silence-guard 보강** ([engines/vllm_stream.py:307-310](../qwen3-tts-api/engines/vllm_stream.py)): 이전엔 `max_silence_duration_ms` 초과 시 `break` 직전의 trigger chunk가 yield되지 않고 폐기됨. 이제 `break` 전에 `yield current_chunk` 추가 → borderline RMS chunk(자음 감쇠 등 음성 꼬리)가 보존됨.
> 2. **Trailing silence padding** ([tts_config.py:130](../qwen3-tts-api/tts_config.py), 기본 800ms): 합성 chunk 루프 종료 후 N ms 무음 PCM을 yield → decoder/player가 마지막 frame을 안정적으로 flush, 끝의 클릭/팝 제거. vLLM·Supertonic in-process·Supertonic 워커 **3개 경로 모두** 적용. `trailing_silence_ms=0` per-call override로 비활성 가능.

---

**Q. (v0.9.0 추가) Supertonic 워커 설정(MAX_CONC, INTRA, OMP)은 어떻게 결정되는가?**

> **A.** **하이브리드 자동 튜닝**:
> 1. **deploy.sh 1차 계산** ([scripts/deploy.sh:659-694](../scripts/deploy.sh)): 호스트 `nproc` + `free -m` + GPU 감지로 자동 산출. 공식 `MAX_CONC = min(RAM_GB÷0.6, CPU÷4, 8)`, `INTRA = max(2, CPU÷MAX_CONC)`. GPU 모드는 `MAX_CONC=4, INTRA=2` (CPU는 pre/post만).
> 2. **사용자 env 우선**: `TTS_SUPERTONIC__MAX_CONCURRENCY=1 ./scripts/deploy.sh` 식으로 명시한 값이 있으면 그쪽 우선. 나머지는 자동 계산.
> 3. **Python init 2차 가드** ([engines/supertonic/__init__.py:57-100](../qwen3-tts-api/engines/supertonic/__init__.py)): 컨테이너 실측 자원으로 재검증. 가용 RAM 부족 시 + INTRA×MAX_CONC가 nproc 초과 시 자동 하향 + WARN 로그. OOM kill / oversubscription 사전 차단.
> 4. **최종 결정값 노출**: 부팅 시 `[Supertonic] 합성 워커 설정 확정: ...` INFO 로그 + `/v1/diagnostics` 응답으로 운영자가 확인.

---

**Q. (v0.9.0 추가) 대시보드 감사 로그가 어떻게 강화됐는가?**

> **A.** 이미 백엔드 응답에 들어있던 필드를 모두 가시화 + UX 강화:
> 1. **TTFB(ms) 컬럼** + 색상 등급 (`< 300ms` 초록, `< 800ms` 파랑, `< 1500ms` 노랑, 그 이상 빨강) — 느린 케이스 즉시 식별.
> 2. **다중 행 펼침**: `Set<string>` state로 여러 행 동시 펼침. 펼친 행에 `language`/`speed`/`temperature`/`options` dict 전체 + `error_message` 표시.
> 3. **텍스트 셀 클릭-복사**: hover 시 `Copy` 아이콘 fade-in, 클릭 시 클립보드 복사 + `복사됨` 토스트. 행 토글과 `e.stopPropagation()`로 분리.
> 4. **엔진 필터** 드롭다운: 전체/vLLM/Supertonic — 검색어와 동시 적용.
> 5. **타임존 표시**: 헤더에 사용자 로컬 offset(예: "UTC+9"), timestamp 셀 hover 시 ISO 원본.
> 6. **CSV export 확장**: 7→12 컬럼 (Language/Speed/Temperature/TTFB/Options JSON 추가).

---

**Q. (v0.9.5 추가) 목소리 복제(ICL) 시 이상한 웃음소리가 들리는 현상(환각)의 대책은?**

> **A.** 복제에 사용하는 레퍼런스 음원의 음질이 깨끗하지 않거나(마이크 숨소리, 공백, 잡음 등), 혹은 중국어/일본어 복제 목소리로 한글 텍스트를 합성하는 '교차 언어' 합성 시 억양 예측 레이어의 정합성이 틀어지며 모델이 빈 구간에 음향적 "Giggle(웃음소리)"을 채워넣는 오디오 환각 현상이 발생합니다.  
> 이의 대책으로:
> 1. 합성 설정의 ICL 활성 툴팁에 상세 경고 및 예방 조치 가이드를 다국어로 추가 반영했습니다.
> 2. 레퍼런스 음성으로 **5~8초 길이의 잡음 없고 감정이 섞이지 않은 건조한 녹음본**을 사용하도록 안내합니다.
> 3. 신규 음성 등록 시 **`ref_audio_trim_silence`** 옵션을 제공하여 시작 묵음을 자동 파쇄 전처리합니다.
> 4. 파라미터에서 **`Temperature를 0.7~0.8`**, **`Top-P를 0.8`** 수준으로 낮추어 안전한 음향 후보군만 연산하게 통제합니다.

---

## 빠른 참조 — 핵심 수치 요약

| 항목 | 수치 |
|:---|:---|
| 지원 언어 수 | **32개국** (vLLM 10 + Supertonic 21) |
| 초도 음성 지연 TTFB | **저부하 p95 ~0.3s · 동시 30 에서 ≤1s** (0.6B·07-28 확정) / **캐시 시 ~50ms** |
| RTF (Real-Time Factor) | **0.39** (vLLM, L40S) |
| 최대 동시 채널 | 수용 36 (세마포어) · CB OPEN 56 / CLOSE 44 |
| 배포 에디션 수 | **2종** (2 Internal / 4 External; 1 Lite·3 Professional은 폐지→2/4) |
| 컨테이너 (현재 Internal) | **4개** (api/vllm-a/fast-a/redis) · External 최대 7개(+vllm-b/postgres/nginx) |
| ITN 도메인 수 | **11개** (standard/finance/tech/healthcare/education/commerce/mobility/aicc/media/game/casual) |
| ITN 룰 총 수 | **약 1,363개** (v0.8.0 중복 제거 후) |
| ITN 골든셋 회귀 테스트 | **101 케이스 / 24개 카테고리** |
| 합성 파라미터 수 | **25개** |
| 배포 다운타임 | **~10초** (deploy-roll.sh, 무중단) |
| 오디오 포맷 | WAV / FLAC / OGG / MP3 |
| 샘플레이트 | vLLM 24kHz 네이티브 (HiFi 옵션 48kHz) / Supertonic 44.1kHz |
| 응답 헤더 (v0.8.0+) | X-Engine / X-Engine-Fallback / X-Engine-Fallback-Reason / Retry-After |
