# 로직 레퍼런스 — DB 계층

## 읽는 법

`로직_레퍼런스_{코어,합성파이프라인,외부호환어댑터,ITN}.md` 와 동일 포맷.
항목마다 **무엇을 / 왜·배경 / 어디서 / 어떻게 / 주의·함정**. `≈L숫자`는 근사 라인.

## 범위

대상: `db/database.py`(엔진·연결풀), `db/manager.py`(파사드, 1370줄), `db/models.py`(스키마), `db/repositories/*`(Repository 패턴 구현체).

미포함: 개별 CRUD 메서드 전수 나열(대부분 표준 SQLAlchemy select/insert), `scripts/migrations/*` 1회성 스크립트.

---

# 1. 저장소 이원화 (SQL ↔ JSON 파사드)

### 1.1 `DBManager` = SQL/JSON 저장소 파사드
- **무엇을**: `settings.audio.database_url`이 `postgresql`/`sqlite`로 시작하면 SQL Repository 구현체(`SqlUserRepository` 등)를, 아니면 JSON 파일 Repository(`JsonUserRepository` 등)를 조립한다. 호출부는 `db_manager.user_repo`/`.settings_repo`/`.voice_repo` 등 동일 인터페이스만 본다.
- **왜/배경**: Edition 1/3(legacy, JSON 파일 기반 "Lite")과 Edition 2/4(Standard+, Postgres/SQLite 기반)가 같은 서비스 코드를 공유해야 한다. `로직_레퍼런스_코어.md`의 Edition 분기와 맞물림.
- **어디서**: `db/manager.py::DBManager.__init__`(≈L35-76).
- **어떻게**: 5개 Repository ABC(`db/repositories/base.py`) — `UserRepository`/`SettingsRepository`/`AuditRepository`/`VoiceRepository`/`FeedbackRepository` — 각각 Sql/Json 두 구현체를 가진다.
- **주의/함정**: `feedback_repo`는 audit 분리 DB가 있으면 그쪽(`AuditSessionLocal`)에, 없으면 메인 DB에 붙는다 — audit 전용 인스턴스 운영 시 feedback 테이블 위치를 착각하기 쉽다.

### 1.2 워커 수 기반 커넥션 풀 동적 계산
- **무엇을**: Postgres 풀 크기를 `TTS_WORKERS` 환경변수로부터 역산한다. 공유 예산 90(≈PG `max_connections=100`의 여유)을 워커 수로 나눠 워커당 슬롯을 만들고, 그 중 55%를 pool_size, 20%를 max_overflow, 15%/10%를 audit 엔진에 배분한다.
- **왜/배경**: `pool_size × TTS_WORKERS ≤ PG max_connections`을 보장하지 않으면 워커 수를 늘렸을 때 Postgres 연결 한도를 넘겨 신규 연결이 전부 실패한다(Y1 fix).
- **어디서**: `db/database.py`(≈L8-16).
- **주의/함정**: 이 계산은 `TTS_WORKERS` 환경변수를 **직접** 읽는다(DB 설정이 아님) — `.env.prod`의 `TTS_WORKERS`와 `start_api.sh`의 `${TTS_WORKERS:-4}` 등 다른 3곳과 값이 어긋나면(코드베이스 전반에 흩어진 워커수 참조) 실제 워커 수와 계산된 풀 크기가 안 맞을 수 있다 — 합성파이프라인.md §5 "Redis 장애 폴백 시 워커 수 분할"과 동일한 종류의 함정.

### 1.3 SQLite 전용 최적화 (PRAGMA + fallback 경로)
- **무엇을**: `database_url`이 비어 있으면 `settings.db_dir` 기반 SQLite로 자동 폴백하고, 모든 SQLite 연결에 `PRAGMA synchronous=NORMAL` + `journal_mode=WAL`을 건다.
- **왜/배경**: 컨테이너 재생성 시 데이터가 유지되려면 fallback 경로가 절대경로(볼륨 마운트 지점)여야 한다. WAL은 SQLite 동시성(reader/writer 비블로킹)을 위해 필수.
- **어디서**: `db/database.py::engine`(≈L18-24, 50-60), `set_sqlite_pragma`(≈L64).
- **주의/함정**: DB URL을 로그에 남길 때 `_redact_db_url`로 크리덴셜(`user:pass@`)을 마스킹한다 — Postgres URL을 그대로 로그에 찍으면 파일/컨테이너 로그에 비밀번호가 평문 노출된다.

### 1.4 Audit DB 분리 (고빈도 쓰기 격리)
- **무엇을**: 감사 로그(`audit_logs`, `user_feedback`)를 메인 DB와 별도 엔진(`audit_engine`)으로 분리할 수 있다. 우선순위: ① `audit_database_url` 명시값 → ② 메인이 Postgres면 같은 DB 공유 → ③ 메인이 SQLite면 `{db_dir}/audit.db`.
- **왜/배경**: 합성 요청마다 쓰이는 감사 로그가 메인 트랜잭션 풀을 잠식하지 않도록 격리(Issue 3). 과거 `./db/audit.db`(이미지 내부 경로)를 쓰다가 rebuild마다 데이터가 사라졌던 버그가 있어 `db_dir` 기반 절대경로로 교정됨.
- **어디서**: `db/database.py`(≈L80-131).

### 1.5 부팅 시 스키마 마이그레이션 — idempotent ALTER + 트랜잭션 격리
- **무엇을**: `init_db()`가 `create_all`(신규 테이블) 다음에 `ALTER TABLE ... ADD COLUMN`(SQLite는 IF NOT EXISTS 미지원이라 예외 흡수, Postgres는 `IF NOT EXISTS`)로 컬럼을 추가하고, 인덱스는 `CREATE INDEX IF NOT EXISTS`로 멱등 생성한다.
- **왜/배경**: 멀티 워커가 동시에 `init_db()`를 실행해도(uvicorn 여러 프로세스) `already exists`/`duplicate` 예외를 무시하는 래퍼(`_create_all_safe`)로 레이스가 죽지 않는다.
- **어디서**: `db/database.py::init_db`(≈L133-344).
- **어떻게**: 인덱스·데이터 백필(예: `voices.engine` 재분류 UPDATE)은 메인 DDL 트랜잭션 **밖**에서 각자 독립 트랜잭션으로 실행한다.
- **주의/함정**: Postgres는 트랜잭션 안에서 한 문(statement)이 실패하면 **트랜잭션 전체가 abort**되고(파이썬 `except`로는 SAVEPOINT 없이 복구 불가) 이후 문·commit이 함께 깨진다. 문마다 독립 트랜잭션으로 격리하지 않으면 한 실패가 메인 스키마 생성 자체를 오염시킨다 — 이 문서의 다른 어떤 항목보다 자주 재발하는 함정 패턴.

---

# 2. 설정(Settings) 저장 — Sparse Override

### 2.1 Sparse Override 저장 원칙 (단일 진실원천)
- **무엇을**: DB/JSON에는 `_SETTINGS_DEFAULTS`와 **값이 다른 키만** 저장한다(`strip_default_settings`). 로드 시 `dict(defaults)` 위에 저장된 override를 overlay해 전체 설정을 복원한다.
- **왜/배경**: 코드 기본값이 바뀌면 그 키를 override하지 않은 모든 배포(신규/기존, SQLite/Postgres)에 자동 반영되어야 한다. 과거엔 전체 blob을 저장해서 "부팅마다 값을 되돌리는 저널"과 "Postgres 전용 SQL override로 인한 환경별 분열"이 있었고, 이게 실제 운영 버그(끝 단어 잘림)의 근본 원인이었다.
- **어디서**: `db/repositories/base.py::strip_default_settings`(≈L5-21), `db/repositories/settings_repo.py::save_settings`(≈L41-63).
- **어떻게**: `k not in defaults or defaults[k] != v` 로 필터. int/float/bool은 `==` 동등비교라 `750 vs 800.0`은 다름, `800 vs 800.0`은 같음으로 판정.
- **주의/함정**: `defaults`에 없는 키(`updated_at`, SMTP 실값 등)는 override 여부와 무관하게 항상 유지된다.

### 2.2 레거시 full-blob → sparse 1회 정규화 (`_sparsify_stored_settings`)
- **무엇을**: 부팅 시 저장된 override가 "sparse가 아니면"(레거시 full-blob) 기본값과 동일한 키를 벗겨 sparse로 변환·재저장한다.
- **왜/배경**: §2.1 구조가 도입되기 전 저장된 레거시 데이터를 1회성으로 정규화. `sparse == raw`면 이미 sparse이므로 no-op(idempotent) — 매 부팅마다 DB 쓰기가 발생하지 않는다.
- **어디서**: `db/manager.py::_sparsify_stored_settings`(≈L500), `_load_raw_overrides`(≈L538, defaults 미병합 원본 조회 전용 경로).
- **주의/함정**: 이 루틴은 "커스텀까지 강제 리셋"은 하지 않는다 — 관리자가 명시적으로 바꾼 값(기본값과 다른 값)은 그대로 보존. 전체 리셋이 필요한 드문 경우는 별도 1회성 마이그레이션으로 분리.

### 2.3 설정 캐시 — TTL + pub/sub + fail-open 3단 폴백
- **무엇을**: `load_settings()`는 ① 프로세스 로컬 캐시(TTL 5s, double-check lock) → ② Redis/DB 조회(`wait_for` 2s 타임아웃) → ③ 실패 시 stale 캐시 반환 → ④ 캐시도 없으면 `_SETTINGS_DEFAULTS` 반환, 4단으로 폴백한다.
- **왜/배경**: 이 경로는 **핫패스**다 — maintenance 미들웨어가 `/health` 포함 매 요청마다 호출한다. `wait_for` 없이 DB가 hang하면 `_cache_lock`을 쥔 채 워커의 모든 요청이 줄서서 멈춘다.
- **어디서**: `db/manager.py::load_settings`(≈L879-930).
- **어떻게**: 폴백 시(③) **캐시 시각을 갱신**한다(negative caching) — 안 하면 다음 요청도 TTL 만료로 판정돼 `_cache_lock`을 쥔 채 2초 DB 타임아웃을 직렬로 재시도해 워커가 ~0.5rps로 붕괴한다. TTL 동안은 fast-path로 흡수.
- **주의/함정**: Standard+(SQL)는 Redis pub/sub이 `force_reload_settings()`를 즉시 호출해 캐시를 비우므로 TTL은 safety net일 뿐. Lite는 pub/sub이 없어 TTL(5s) 경과가 유일한 갱신 트리거다.

### 2.4 Redis pub/sub 기반 멀티워커 설정 무효화
- **무엇을**: `save_settings()`가 저장 직후 `publish_config_change()`로 다른 uvicorn 워커에 알리고, 각 워커는 `subscribe_config_changes(force_reload_settings)`로 구독해 로컬 캐시(설정+voice registry+voice meta)를 즉시 비운다.
- **왜/배경**: 워커별 프로세스 로컬 캐시라 한 워커의 admin 편집이 다른 워커에 반영되려면 IPC가 필요하다.
- **어디서**: `db/manager.py::save_settings`(≈L991), `force_reload_settings`(≈L1013), `start_config_listener`(≈L1038).
- **주의/함정**: `force_reload_settings`는 설정 캐시뿐 아니라 **voice 캐시도 함께** 비운다 — 과거엔 설정만 비워서 다른 워커가 비활성화/편집된 voice를 TTL(5s/30s) 동안 계속 서빙하는 버그가 있었다(voice mutation도 같은 `config_change` 채널로 publish되므로). `save_settings` 반환값(`bool`)은 "저장 성공"이 아니라 "pub/sub 전파 성공" — `False`면 저장은 됐지만 다른 워커는 TTL 만료까지 지연 반영된다.

### 2.5 ITN 규칙 무효화는 별도 pub/sub 채널
- **무엇을**: ITN builtin/domain 규칙 변경은 `config_change`와 다른 전용 채널(`subscribe_itn_invalidation`)로 전파된다. builtin 변경은 각 워커의 in-memory `reload_builtin_rules()`를, domain 변경은 DB 재조회(`reload_domain_rules()`)를 트리거한다.
- **왜/배경**: builtin은 코드 정규식 재컴파일이 필요해 memory reload, domain은 `apply_rules`가 매 호출 DB dict를 참조하므로 clear만으로는 부족 — "clear만 하면 다음 조회가 stale JSON을 읽어 admin 표시(DB)와 실제 합성 결과가 어긋난다"는 주석이 명시됨.
- **어디서**: `db/manager.py::start_itn_listener`(≈L1042).

---

# 3. 최초 기동 마이그레이션 & 자가치유

### 3.1 JSON(Lite) → SQL(Standard+) 1회 마이그레이션
- **무엇을**: 서버 최초 기동 시 `_run_sql_migrations`가 Redis 분산 락(`db:system_migration`, 120s) 안에서 계정·voice·설정을 legacy JSON(`voices.yaml`, `global_settings.json`)에서 SQL로 옮긴다.
- **왜/배경**: Edition을 1/3(JSON)에서 2/4(SQL)로 승급하는 배포 시 기존 데이터를 잃지 않아야 한다.
- **어디서**: `db/manager.py::_run_sql_migrations`(≈L213).
- **주의/함정**: voice 마이그레이션은 **"DB가 마스터"** 원칙 — 이미 DB에 데이터가 있으면 YAML 신규 항목만 추가하고, 메타데이터(name/description/descriptor/ref_text)는 DB 필드가 **빈 경우에만** YAML 값으로 채운다(blank-fill, overwrite 아님). YAML 값이 list/dict(손상)면 스킵 후 WARNING만 남기고 계속 진행. `ref_text`가 신규 시드되면 합성 캐시 무효화(`voice:version` INCR)까지 함께 트리거한다 — 캐시 오염 방지.

### 3.2 ITN 시드 드리프트 재동기화 (해시 게이트)
- **무엇을**: 배포로 `data/itn/*.json` 시드가 바뀌면(개선된 규칙 배포), 이미 DB에 값이 있어도 해시 비교로 변경분만 재적용한다.
- **왜/배경**: "seed-if-empty"(§3.1과 같은 blank-fill 방식)로는 "기존 DB에 이미 값이 있어 시드 개선이 반영 안 되는 드리프트"를 못 잡는다. ITN은 시드 JSON 자체가 소스 오브 트루스이므로 예외적으로 덮어쓴다.
- **어디서**: `db/manager.py::_resync_itn_seed_if_changed`(≈L426).
- **주의/함정**: 해시가 그대로면 no-op이라 **라이브 admin 편집은 보존**된다 — 시드 파일이 바뀐 배포에서만 동작. Lite(JSON) 에디션은 파일을 직접 읽으므로 이 로직 자체가 불필요.

### 3.3 `public_id` 결정적 1회 배정 (freeze)
- **무엇을**: `public_id`가 NULL인 voice에 결정적 규칙(`compute_public_id`)으로 공개 ID를 배정하고 저장 후 다시는 바뀌지 않는다(freeze).
- **왜/배경**: 예전엔 공개 ID를 런타임마다 파생(voice_alias 리빌드)했는데, enable/재시드/재정렬 시 충돌회피 로직 때문에 같은 voice의 공개 ID가 흔들릴 수 있었다. 부팅 시 1회 저장값으로 고정하면 이후 완전히 안정된다.
- **어디서**: `db/manager.py::_backfill_public_ids`(≈L381).
- **어떻게**: Redis 락으로 멀티워커 직렬화(첫 워커가 배정, 나머지는 NULL이 없어 no-op). 배정 순서는 `enabled` 우선 + 내부ID 정렬로 결정적.
- **주의/함정**: descriptor가 없어 공개 ID를 파생할 수 없는 voice는 NULL로 유지(내부 ID 노출) — 강제 배정하지 않는다.

### 3.4 `ref_audio` dangling 참조 자가치유
- **무엇을**: 부팅 시 모든 voice의 `ref_audio` 경로를 정규 경로(`samples/{voice_id}.{ext}`)와 대조해, 시드 재이관·편집 이력으로 끊긴 참조를 교정한다.
- **왜/배경**: 빈 DB에서 YAML로 재시드되면 옛(삭제된) 경로가 되살아나 미리듣기/합성이 404를 내는 버그가 있었다(관련: [[voice-preview-404-dangling-ref]] 메모리). 리터럴 파일이 살아있으면 무변경.
- **어디서**: `db/manager.py::_run_sql_migrations`(≈L322-339), `services/tts/sample_resolver.py::resolve_sample_ref`.

### 3.5 Voice 메타 조회 — TTL 캐시 + single-flight
- **무엇을**: `get_voice_metadata(vid)`는 TTL(30s) 캐시를 먼저 보고, miss 시 동일 `vid`에 대한 동시 호출들을 하나의 `asyncio.Future`로 묶어 DB는 1번만 조회한다.
- **왜/배경**: PG 일시 장애나 cache miss가 몰릴 때 N개 동시 요청이 각각 DB를 때리면(thundering herd) 장애가 N배로 증폭된다. single-flight로 N→1로 축소(M7 fix).
- **어디서**: `db/manager.py::get_voice_metadata`(≈L1069-1112).
- **주의/함정**: 실패 결과(`None`)는 캐시하지 않는다 — 마이그레이션 진행 중이거나 직전 삭제된 voice의 재조회를 보장하기 위해서다. in-flight `Future`가 완료 전에 취소되면(리더 코루틴이 `create_future()` 후 `set_result()` 전에 취소됨) 대기 중이던 다른 코루틴들이 영구 hang할 수 있어, `finally`에서 미완료 future를 명시적으로 `cancel()`해 깨운다.

---

# 4. 스키마 하이라이트 (`db/models.py`)

### 4.1 감사 로그 vs 설정 vs 음성 — 3대 핵심 테이블
- `User`: `api_key_hint`(마지막 4자리만 저장, 원본 키는 해시만 보관), `password_changed_at`/`role_changed_at`(JWT `iat` 비교용 세션 무효화 타임스탬프, [[로직_레퍼런스_코어]] JWT 검증 섹션과 연결).
- `AuditLog`: `queue_ms`/`inference_ms` 분리 저장(P3-3, SLA 분석용 latency breakdown), `status_code`/`error_message`(실패 요청 분석). 복합 인덱스 `(username, timestamp)`, `(action, timestamp)`.
- `GlobalSetting`: `key="global"` 단일 행에 전체 설정을 JSON으로 저장하는 방식으로 통일(Issue 215) — 키별 분산 저장 방식에서 일관성 문제로 전환됨.
- `Voice`: `engine` 필드(명시적 vllm/supertonic 구분, prefix 추론 대체), `public_id`(§3.3), `ref_audio_duration_ms`(ICL 응답 trim용, librosa 측정값).

---

# 5. Repository 패턴 (`db/repositories/`)

### 5.1 ABC 기반 5개 Repository, Sql/Json 듀얼 구현
- **무엇을**: `UserRepository`/`SettingsRepository`/`AuditRepository`/`VoiceRepository`/`FeedbackRepository` 추상 클래스를 정의하고, 각각 `Sql*Repository`(SQLAlchemy)와 `Json*Repository`(파일 기반) 두 구현체를 둔다.
- **어디서**: `db/repositories/base.py`(ABC 정의), `settings_repo.py`/`user_repo.py`/`voice_repo.py`/`audit_repo.py`/`feedback_repo.py`(Sql), `json_repos.py`(Json 4종 통합).
- **어떻게**: `SqlSettingsRepository.get_kv(key, raise_on_error=True)` 패턴 — DB 오류를 삼키지 않고 raise해 호출자가 "키 미존재(None)"와 "read 실패(예외)"를 구분할 수 있게 한다. 이 구분이 없으면 시드 재주입 로직이 read 실패를 "키 없음"으로 오인해 관리자 편집을 clobber(덮어쓰기)할 위험이 있다 — `load_itn_builtin` 등이 이 시그니처를 사용하는 이유.


---

## 증보 (2026-08-12) — `db/precache_repo.py` (175줄)

precache 문장 목록의 저장소. **SQL 이 아니라 JSON 파일**(`{db_dir}/precache_sentences.json`) — 소량·저빈도라 의도된 선택.

- 동시성: 모듈 전역 `asyncio.Lock` + 파일 IO 는 `to_thread`. 쓰기는 `mkstemp` → `os.replace` **원자적 교체**(부분 쓰기 파일 방지).
- 엔트리: `{id(uuid), text, voice_id, speed, language, created_at, cached_key}`.
- 중복 판정: `(text.strip, voice_id, round(speed,2), language)` 4-튜플. bulk 등록은 시그니처 set 으로 O(1)(대량 시 quadratic 방지)·단일 write.
- **`cached_key` 계약 (무TTL 영구화의 짝)**: `run_precache` 가 실제 Redis 키를 `update_cached_keys()` 로 되써 넣음(대량 중 hard-kill 대비 20건마다 flush). 삭제 API 3종(`delete_sentence/bulk/all`)은 **삭제된 엔트리를 반환** — 호출자(라우터)가 반환 엔트리의 `cached_key` 로 Redis 오디오를 evict 해야 orphan 이 영구 잔존하지 않음. 설정 드리프트로 키가 바뀐 경우의 옛 키 evict 는 `run_precache` 쪽이 담당.
- ⚠️ voice_id 는 등록 시 공개 별칭일 수 있음 — 해석은 저장소가 아니라 `run_precache` 실행 시점(`to_internal`)에 수행(2026-08 수정).
