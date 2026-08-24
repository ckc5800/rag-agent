# Qwen3-TTS API 및 DB 아키텍처 명세

> 버전: v0.9.8 | 기준일: 2026-06-17

본 문서는 Qwen3-TTS 프로젝트의 데이터베이스 스키마와 API 관리 구조를 명세한다.

> v0.9.x 변경: UserFeedback 테이블 추가(피드백 제출), AuditLog latency breakdown 필드(queue_ms/inference_ms), Voice ref_audio_duration_ms 필드 추가.

## 1. API 관리 구조 (API Management)

현재 API는 **FastAPI** 프레임워크를 기반으로 모듈화 및 체계적으로 관리되고 있음 (`main.py` 중심).

### 1.1 라우터(Router) 분리
API 엔드포인트는 기능별로 분리되어 `routers/` 디렉토리 하위에서 관리됨.
- **`synthesize.py` / `ws_synthesize.py`**: 오디오 합성 핵심 API (Batch, Stream, WebSocket)
- **`auth.py`**: 사용자 인증 및 세션 토큰 관리
- **`admin.py`**: 관리자 패널(엔진 재시작, 유저 관리, 텔레메트리 등) 연동
- **`voices.py`**: 시스템에 등록된 목소리(Voice) 조회 및 관리
- **`icl.py`**: In-Context Learning (목소리 실시간 복제) API
- **`metrics.py` / `measure.py`**: 시스템 리소스 모니터링 및 성능/정확도 측정

### 1.2 미들웨어(Middleware) 체인
API 요청에 대한 보안 및 리소스 보호를 위해 다층 미들웨어를 적용함.
- **CORS Middleware**: 허용된 도메인에서의 접근만 승인함.
- **Payload Limit**: 과도한 대용량 텍스트 요청을 사전 차단함.
- **Maintenance Mode**: 점검 시간 중 인가된 권한자 외의 접근을 통제함.
- **Backpressure Circuit Breaker**: 시스템 부하(VRAM 등) 급증 시 신규 요청을 `503 Service Unavailable`로 임시 차단(Fail-fast)하여 서버 다운을 방지하는 핵심 보호 장치임.

### 1.3 인증 시스템 (Auth)
- **인증 모드 4종** (`TTS_AUTH__MODE`): public / team / solo / secret — 배포 시 결정. 토큰 검사가 켜지는 모드는 **public**뿐이며 team/solo/secret은 `DISABLED=true`. 단일 출처 [`../../deploy/공통_개념.md`](../../deploy/공통_개념.md) §3. 현재 운영=**secret**.
- **B2C (JWT Token)**: 관리자 대시보드 및 프론트엔드 UI용 단기 세션 토큰 사용 (`Bearer <token>`, public 모드).
- **B2B (API Key)**: 외부 시스템 연동용 영구 키 사용 (`Bearer sk-<key>`). Redis 캐싱을 적용하여 DB 조회 부하를 최소화함.

---

## 2. 활용 중인 주요 기능 (Core Features)

단순 TTS를 넘어 엔터프라이즈급 부가 기능들을 적용 중임.

1. **듀얼 엔진 아키텍처 (Dual-Engine)**
   - **Qwen3 (vLLM)**: 고성능 텐서 병렬 처리 기반 초저지연 엔진 (10개국어 특화, 중국어/일본어 고품질 신규 프리셋 탑재).
   - **Supertonic**: 다국어(31개국어) 및 빌트인 프리셋(M1~M5 등) 처리를 위한 내장형 경량 엔진.
2. **합성 파이프라인 (Synthesis Modes)**
   - **Batch**: 전체 문장을 고품질 오디오 파일로 일괄 변환.
   - **Stream**: 문장 완성 전 청크(Chunk) 단위로 분할 전송하여 지연 시간(TTFB) 최소화.
   - **WebSocket**: 실시간 양방향 오디오 스트리밍 통신 (AI 상담사 등 활용).
3. **In-Context Learning (ICL) 실시간 복제**: 3~10초 분량의 음성 샘플만으로 대상 화자의 톤과 억양을 실시간 복제.
4. **오디오 후처리 및 품질 향상 (DSP)**
   - **Silence Guard & Lookback**: 환각(Hallucination) 현상 조기 차단 및 첫 발음 잘림 방지.
   - **Dynamic ITN**: 숫자, 날짜, 화폐 단위 등을 자연스러운 발음으로 실시간 표준화(정규화)함.

---

## 3. 데이터베이스 관리 구조 (DB Management)

`db/manager.py`의 **DBManager (Facade 패턴)**를 통해 배포 환경에 따른 유연한 스토리지 전환을 지원함.

### 3.1 듀얼 스토리지 지원 (Storage Backend)
- **SQL (PostgreSQL/SQLite)**: 다중 워커(Multi-worker) 및 운영(Production) 환경에서 트랜잭션 무결성 보장을 위해 사용.
- **JSON (Local Files)**: 개발(Dev) 및 자원 제한 환경(Lite)에서 파일 기반의 경량 관리 지원.
- **자동 마이그레이션**: 시스템 기동 시 기존 JSON 데이터를 파악하여 SQL DB로 덮어쓰는(Seed) 자동 전환 기능을 포함함.

### 3.2 분산 아키텍처 및 자동 Seeding (v0.9.5+)
- **실시간 설정 동기화**: 설정 변경 시 Redis Pub/Sub 이벤트를 발생시켜 모든 워커에 브로드캐스트함으로써 서버 재시작 없는 정책 변경 적용.
- **영속 스토리지 = Named Volume `api_unified:/app/data`**: SQLite DB(`/app/data/db/tts.db`), 설정·발음사전·ITN(global_settings), 시드 `voices.yaml`, 샘플, 캐시까지 **단일 볼륨**에 영속됨. named volume은 "최초 생성" 시에만 이미지의 baked data를 복사하므로, 재배포로 코드의 시드 파일이 바뀌어도 자동 반영되지 않는다 — 아래 두 메커니즘이 이를 보완.
- **부팅 시점 증분 Seeding (voices)**:
  - 기동 시 `db/manager.py`가 시드 `/app/data/voices/voices.yaml`을 읽어 **DB에 없는 음성만** DB(`voices` 테이블)에 추가한다. DB가 마스터이므로 **기존 음성의 `enabled` 상태는 DB 값을 보존**(YAML로 덮어쓰지 않음) — `manager.py:205-225`.
  - 결과: 코드가 새 음성을 추가하면 재배포 시 자동 반영되되, 운영자가 비활성/삭제한 음성은 DB 상태가 우선되어 되살아나지 않는다(단, YAML에 남아있고 DB에서 완전 삭제된 음성은 "없는 음성"으로 재시드됨에 주의).
- **ITN 룰 매기동 재시드** (commit 2a626b7): 이미지 빌드 시 `data/itn/*.json`을 볼륨 밖 `/app/data_seed/itn/`에 baked 사본으로 두고, 컨테이너 CMD가 **매 기동마다** `/app/data/itn/`으로 `cp`한다(`Dockerfile`). 이 단계 덕분에 named volume이 이미 존재해도 코드의 ITN 룰 변경이 재배포 때 항상 반영된다.

---

## 4. 데이터베이스 테이블 스키마 현황 (SQLAlchemy Models)

현재 시스템은 5개의 핵심 테이블로 구성됨.

### 4.1 `users` (사용자 및 API Key 관리)
| 필드명 | 타입 | 속성 | 설명 |
|---|---|---|---|
| `id` | Integer | PK, Auto | 유저 고유 식별자 |
| `username` | String(50) | Unique, Index | 사용자 ID (로그인 목적) |
| `password_hash` | String(255) | | bcrypt로 암호화된 비밀번호 |
| `role` | String(20) | | 권한 (`super_admin`, `admin`, `user`) |
| `status` | String(20) | | 계정 활성화 상태 |
| `api_key` | String(100) | Unique, Index | 외부 연동용 API Key의 해시값 (SHA-256) |
| `api_key_hint` | String(20) | Nullable | 키 식별용 마지막 4자리 (예: `sk-...wxyz`) |
| `api_key_last_used_at` | DateTime | Nullable | API Key의 최종 사용 시간 |
| `created_at` | DateTime | | 계정 생성 시각 (UTC) |
| `password_changed_at` | BigInteger | Nullable, Index | 비밀번호 갱신 Unix 타임스탬프. JWT iat과 비교해 기존 토큰 즉시 무효화 |
| `role_changed_at` | BigInteger | Nullable, Index | 권한(role) 변경 Unix 타임스탬프. password_changed_at과 동일 무효화 패턴 |
| `org_id` | String(50) | Nullable, Index | 조직 ID (멀티테넌트, 기본값 "default") |

### 4.2 `audit_logs` (감사 로그 및 텔레메트리)
요청 내역, 성공/실패 여부, 응답 지연 시간, 합성 옵션 전체를 기록. v0.9.0~ 프런트엔드 OverviewTab의 감사 로그 패널이 아래 필드 전체를 색상 등급/펼침 행/CSV export로 가시화.
| 필드명 | 타입 | 속성 | 설명 |
|---|---|---|---|
| `id` | Integer | PK, Auto | 로그 식별자 |
| `timestamp` | DateTime | Index | 이벤트 발생 시각 (UTC). 프런트는 사용자 로컬 TZ로 표시, hover 시 ISO 원본 노출 |
| `username` | String(50) | Index | 요청 수행 주체 |
| `role` | String(20) | | 요청자 권한(`admin`/`user`/`api`) |
| `action` | String(30) | Index | 이벤트 분류 (`synthesis_stream`, `synthesis_batch`, `login` 등) |
| `voice_id` / `engine` | String | | 적용된 화자 및 합성 엔진 정보. 엔진은 프런트 필터 드롭다운(전체/vLLM/Supertonic)의 키 |
| `text` | Text | | 합성 텍스트 원본. 프런트 셀 클릭 시 클립보드 복사 |
| `language` | String(10) | | 합성 언어 코드 (auto-detect 시 reconciled 값) |
| `speed` / `temperature` | Float | | 합성 파라미터 |
| `options` | JSON | | 합성 옵션 dict 전체 (`use_itn`/`high_fidelity`/`use_split`/`smart_prosody`/`itn_domain`/`supertonic_style` 등) — 프런트 펼침 행에서 grid로 표시 |
| `latency` | Float | | 합성 총 소요 시간 (ms) |
| `ttfa` | Float | | 첫 음절 응답 지연 시간 (ms). **프런트엔드에서는 "TTFB" 라벨로 표시** + 색상 등급(<300/<800/<1500ms = 초/파/노/빨) |
| `cer` | Float | Nullable | Character Error Rate (0~1). measure_cer=true 요청 시만 |
| `queue_ms` | Float | Nullable | 큐 대기 시간 (ms). SLA 분석용 latency breakdown. NULL=미수집 |
| `inference_ms` | Float | Nullable | 추론(inference) 소요 시간 (ms). queue_ms와 함께 latency breakdown |
| `ext_info` | Text | Nullable | 추가 메타데이터(스트리밍 chunk 수, 청크 간 jitter 등) |
| `org_id` | String(50) | Nullable, Index | 조직 ID (멀티테넌트) |
| `status_code` | Integer | Nullable, Index | HTTP 응답 코드. NULL=구버전/비합성 액션 |
| `error_message` | String(500) | Nullable | 장애 발생 시 에러 사유 (최대 500자). 프런트 펼침 행 하단에 빨간 박스로 표시 |

**복합 인덱스**: `(username, timestamp)`, `(action, timestamp)` — 사용자별·이벤트별 시계열 조회 최적화.

### 4.3 `voices` (음성 모델 레지스트리)
시스템에 등록된 화자(Voice) 메타데이터를 관리함.
| 필드명 | 타입 | 속성 | 설명 |
|---|---|---|---|
| `voice_id` | String(100) | PK | 화자 고유 식별자 (예: `qwen_ja_female_nanami`) |
| `name` | String(100) | | UI에 표시되는 화자 명칭 (예: `Japanese Nanami`) |
| `language` / `gender` | String(50)/(20) | Index | 화자의 주 사용 언어 및 성별 |
| `description` | Text | Nullable | 화자 설명 (UI 표시·관리용) |
| `ref_audio` | String(255) | Nullable | ICL 복제 처리(vLLM) 또는 미리듣기 정적 파일(Supertonic) 경로. 호스트 `./data/voices/samples/` → 컨테이너 `/app/data/voices/samples/` (예: `./samples/female/qwen_cv_de_f01.wav`) |
| `ref_text` | Text | | 참조 오디오 스크립트. Supertonic 내장 스타일은 `M1`~`M5`/`F1`~`F5` 한 글자 코드 저장 |
| `engine` | String(20) | Index | 음성 처리를 전담할 타겟 엔진 (`vllm`, `supertonic`). 기본값 `vllm` |
| `enabled` | Boolean | Index | 사용 활성화 여부 플래그. false 시 합성 요청 422 |
| `ref_audio_duration_ms` | Integer | Nullable | ref_audio 발화 길이(ms). ICL 합성 시 이 길이만큼 앞 PCM을 trim (prefix output 제거). librosa로 등록 시 측정. NULL=trim 없음(legacy) |
| `created_at` | DateTime | | 등록 시각 (UTC) |

> **Supertonic 내장 음성 미리듣기 (v0.9.5+)**: Supertonic 빌트인 10종(`supertonic_m1~m5`, `supertonic_f1~f5`)은 합성 시 ref_audio를 사용하지 않지만, 관리자 페이지의 음성 카드에서 즉시 재생 가능한 정적 wav를 동일 필드에 등록한다(`./samples/supertonic/{m1..f5}.wav`). 부팅 시 `db/manager.py`의 backfill 로직이 ref_audio가 비어있던 기존 DB 행을 idempotent하게 채워주며, `/app/config/samples/supertonic/` → `/app/voices_data/samples/`로 named volume 복사도 자동 수행된다. 정적 wav는 `scripts/generate_supertonic_samples.py` 로 사전 생성·커밋된다.

### 4.4 `global_settings` (실시간 동적 설정)

키-값(key-value) 다중 행 구조. 주요 키: `global` (합성·인프라 기본값 + **발음사전 `pronunciation_rules` + 커스텀 ITN `itn_custom_rules`**까지 포함한 JSON dict), `itn_builtin` (ITN 빌트인 규칙 JSON 배열), `itn_domain_*` (도메인별 규칙 배열).

> **발음사전·커스텀 ITN은 별도 테이블이 아니라 `global_settings`의 `global` JSON 안에** (`pronunciation_rules`, `itn_custom_rules` 키) 저장된다. 즉 이들은 DB(볼륨)에 영속되며, 시드(`data/itn/*.json`)는 빌트인 ITN에만 해당.

| 필드명 | 타입 | 속성 | 설명 |
|---|---|---|---|
| `key` | String(100) | PK | 설정 네임스페이스. 주키: `global` (SettingsUpdate 전체), `itn_builtin` (도메인 규칙 배열) |
| `value` | JSON | | 딕셔너리/배열 형태의 JSON 값. `key="global"`은 SettingsUpdate 스키마 전체를 단일 JSON 오브젝트로 저장 |

> 설정 변경 흐름: `PATCH /v1/admin/settings` → DB 저장 → Redis Pub/Sub 브로드캐스트 → 모든 uvicorn 워커 캐시 즉시 무효화. Redis 장애 시 5s TTL 폴백.

---

### 4.5 `user_feedback` (사용자 합성 품질 피드백)

합성 직후 또는 이력 페이지에서 사용자가 제출하는 자유 텍스트 피드백. `trace_id`로 `audit_logs`와 cross-reference 가능.

| 필드명 | 타입 | 속성 | 설명 |
|---|---|---|---|
| `id` | Integer | PK, Auto | 피드백 고유 식별자 |
| `created_at` | DateTime | Index | 제출 시각 (UTC) |
| `username` | String(50) | Index | 제출 사용자 |
| `trace_id` | String(64) | Nullable, Index | 합성 응답의 `X-Trace-Id` 값. audit_logs와 join 키 (없어도 됨) |
| `voice_id` | String(100) | Nullable | 피드백 대상 음성 스냅샷 |
| `language` | String(20) | Nullable | 합성 언어 스냅샷 |
| `engine` | String(20) | Nullable | 엔진 스냅샷 (`vllm`/`supertonic`) |
| `text_snippet` | Text | Nullable | 합성 텍스트 앞 500자 스냅샷 (어떤 입력이 문제였는지 식별) |
| `comment` | Text | **required** | 자유 텍스트 피드백 (최대 2000자) |
| `status` | String(20) | Index | 관리자 처리 상태 (`open`/`reviewed`/`resolved`). 기본 `open` |
| `org_id` | String(50) | Nullable, Index | 조직 ID (멀티테넌트) |

**복합 인덱스**: `(username, created_at)`.

**피드백 API**:
- `POST /v1/feedback` — 피드백 제출 (인증 필요, 상세는 REST_명세.md 섹션 5.10)
- `GET /v1/admin/feedback` — 관리자 목록 조회 (status/username 필터)
- `PATCH /v1/admin/feedback/{id}` — 처리 상태 변경
- `DELETE /v1/admin/feedback/{id}` — 삭제
