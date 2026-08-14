# Qwen3-TTS API Key 관리 아키텍처 (v0.9.0 AutoLang & Polish)

본 문서는 Qwen3-TTS의 B2B 연동을 위한 영구적 API Key(`sk-...`)의 생애 주기(Lifecycle)와 보안 처리, 최적화 방식을 분석한 기술 문서임.

> **v0.8.0 변경**: 응답 헤더에 `X-Engine`, `X-Engine-Fallback`이 추가되어 B2B 클라이언트는 어떤 엔진이 응답했는지(vLLM 또는 Supertonic) 식별 가능.
>
> **v0.9.0 변경**: API Key 라이프사이클·인증 로직 자체는 변동 없음. 단, AIVOX 호환 라우터(`POST /v1/audio/speech`, `WS /ws/tts`)가 추가되어 동일한 `sk-` 키로 신규 OpenAI 호환 경로 사용 가능 (별도 키 발급 불필요).

## 1. API 키 발급 및 저장 (Issuance & Storage)
엔드포인트: `POST /v1/auth/api-key`

- **비대칭 저장 원칙**: API 키가 발급되면 사용자에게 원본 키(Raw Key)는 단 한 번만 반환되며, 시스템 내부(DB)에는 **SHA-256으로 단방향 암호화된 해시값**만 저장됨. 
- **식별 힌트 (Hint)**: 관리자나 사용자가 자신의 키를 식별할 수 있도록 `sk-...wxyz` (마지막 4자리) 형태의 힌트(`api_key_hint` 컬럼)만 평문으로 보관함.
- **발급 어뷰징 방지 (Rate Limit)**: 발급 엔드포인트는 악의적인 잦은 재발급을 막기 위해 Redis를 통해 **사용자당 1시간에 10회**로 발급을 제한함.

## 2. API 키 인증 및 캐싱 (Verification & Caching)
엔드포인트 연동 시 HTTP Header `Authorization: Bearer sk-...`를 통해 인증을 수행함.

- **Redis 초고속 캐싱**:
  - 요청이 들어오면 전달받은 API Key를 즉시 SHA-256으로 해싱함.
  - Redis 캐시 버킷(`auth:api_key:{key_hash}`)을 1순위로 조회하여 DB 병목을 차단함.
  - 캐시 미스(Miss) 시에만 DB를 조회하며, 확인된 유저 정보는 **60초 동안 Redis에 캐싱**됨.
- **Last Used 시각 추적 (비동기 Write-Throttling)**:
  - 보안 감사를 위해 키가 사용된 시각(`api_key_last_used_at`)을 기록함.
  - 단, 매 요청마다 DB Write가 발생하면 합성 API에 엄청난 지연(Latency)이 생기므로, **백그라운드 비동기 태스크(Fire-and-forget)**로 분리함.
  - 동시에 Redis의 `SETNX` (TTL 60초) 락을 활용해 1분에 단 1번만 DB를 업데이트하도록 쓰기(Write) 횟수를 획기적으로 스로틀링(Throttling)함.

## 3. API 키 무효화 및 실시간 블랙리스트 (Revocation & Cache Purge)
엔드포인트: `DELETE /v1/auth/api-key`

- **즉각 폐기**: 키 폐기 요청 시 DB의 해시값과 힌트를 즉시 `None`으로 덮어씀.
- **실시간 무효화 (Invalidation) 및 세션 즉시 만료**:
  - **비밀번호 변경(Password Change)**이나 **관리자의 키 무효화 리셋**이 발생하면, Redis의 API Key/JWT 인증 캐시 버킷(`auth:api_key:{key_hash}`)과 로컬 메모리 인스턴스 세션 캐시를 **그 즉시 강제로 Purge(삭제)** 처리합니다.
  - 다음 인입 요청 시, 서버는 Redis 캐시를 타지 못하고 DB 원본을 조회하게 되며, DB 레코드에 변경 시간 차이(`password_changed_at` 불일치)가 대조되어 탈취당한 구형 키 및 세션이 캐시 생명주기(TTL)와 무관하게 **1초의 유예도 없이 즉각 차단(Fail-Closed)** 처리됩니다.

---

## 4. DB 테이블 스키마 연관 컬럼
`users` 테이블 내 API Key 관련 컬럼 구성:

| 필드명 | 타입 | 설명 |
|---|---|---|
| `api_key` | String(100) | 외부 연동용 API Key의 **SHA-256 해시값** (고유 인덱스 설정) |
| `api_key_hint` | String(20) | 키 식별용 마지막 4자리 (예: `sk-...wxyz`) |
| `api_key_last_used_at` | DateTime | API Key 최종 사용 시각 (Redis Throttle 적용으로 분당 최대 1회 갱신) |
| `password_changed_at` | BigInteger | 비밀번호 최종 변경 시각. 세션 및 API Key 검증 시 실시간 무효화 트리거로 비교 활용 |

---

## 5. 외부 개발자 전달 절차 (Operational Workflow)

API Key는 **운영자가 평문을 본 적이 없도록** 클라이언트가 self-service로 발급하는 것을 전제로 설계됨. 운영자는 계정 생성 + 임시 비밀번호만 전달.

### 5.1 표준 절차 (권장)

```
운영자 (admin)                          외부 개발자
─────────────────                       ─────────────────
1) POST /v1/admin/users                 
   role: "user"  password: <임시>
                                        
   임시 비번 전달 (보안 채널) ─────────► 2) /v1/auth/token 로그인
                                        3) /v1/auth/change-password 임시→영구 변경
                                        4) POST /v1/auth/api-key
                                           → 응답에 sk-... 한 번만 반환
                                        5) sk-... 본인 비밀 저장소에 보관
                                           (.env / Vault / Secrets Manager)
                                        6) Authorization: Bearer sk-... 로 합성
```

**curl 한 줄 명령 (외부 개발자용)**:

```bash
# 임시 비번으로 토큰 발급 → 비번 변경 → 키 발급 → 즉시 사용
TOKEN=$(curl -s -X POST $URL/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=client-acme&password=<임시>" | jq -r .access_token)

curl -X POST $URL/v1/auth/change-password \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"old_password":"<임시>","new_password":"<강한_새비번>"}'

curl -X POST $URL/v1/auth/api-key \
  -H "Authorization: Bearer $TOKEN" | jq -r .api_key
# → sk-AbCdEfG...    (이 값을 자신의 .env에 저장)
```

### 5.2 임시 비밀번호 전달 채널

| 채널 | 권장 여부 |
|---|---|
| Signal · ProtonMail · 암호화 메일 (PGP) | ✅ 권장 |
| HashiCorp Vault · AWS Secrets Manager · 1Password Business | ✅ 가장 안전 |
| One-time secret 링크 (Yopass · onetimesecret) | ✅ 허용 |
| 대면 전달 + 즉시 변경 | ✅ 허용 |
| 일반 이메일 / 카카오톡 / 슬랙 일반 채널 | ❌ 금지 |
| GitHub PR / Issue / Jira 코멘트 | ❌ 금지 |
| 스크린샷 / 메모장 파일 첨부 | ❌ 금지 |

### 5.3 수신 확인 절차

1. 클라이언트가 "키 발급 + 첫 합성 1건 성공" 명시적 회신
2. 운영자: 전달 채널에서 임시 비밀번호 즉시 삭제
3. 30일 내 첫 사용 없으면 의심 (수신 실패 또는 누출) → 운영자 알림

---

## 6. 영속성 및 배포 영향 (Key Persistence)

### 6.1 일반 배포 — 키는 유지됨 ✅

API Key 해시는 Postgres `users.api_key` 컬럼에 저장되며, DB는 named volume(`tts-postgres-data`)에 영속화됨 ([docker-compose.prod.yml](../../../docker-compose.prod.yml)). 다음 작업은 키에 영향 없음:

- `./scripts/deploy.sh` 재실행 (정기 배포)
- `./scripts/deploy-roll.sh` (롤링 업데이트)
- 코드 변경 후 컨테이너 rebuild
- 호스트 재부팅 / docker daemon 재시작
- `docker compose down` (`-v` 미사용 시)
- 클라이언트 비밀번호 변경 (API Key는 독립 필드)

### 6.2 키 손실 시나리오 — 주의 🚨

| 작업 | 결과 |
|---|---|
| `docker compose down -v` | 모든 데이터 손실 (DB · 키 · audit logs) |
| `docker volume rm tts-postgres-data` | 동일 |
| `./scripts/deploy.sh down` → 메뉴에서 "볼륨 삭제" 선택 | 동일 |
| `users` 테이블 drop / 수동 wipe | 키 손실 |
| 호스트 디스크 손상 (백업 없는 경우) | 손실 |

### 6.3 운영자 안전 장치 (이미 구현됨)

- **자동 백업**: `scripts/deploy.sh` 배포 시 매일 새벽 3시 cron 자동 등록 (7일 보관)
- **수동 백업**: `./scripts/backup.sh` — Postgres dump + audio + config 포함
- **복원**: `./scripts/restore.sh <백업파일>` — 키 포함 전체 복구
- **금지 사항**: production에서는 절대 `docker compose down -v` 사용 금지

### 6.4 클라이언트한테 안내해야 할 한 줄

> **API Key 영속성**: 발급된 키는 서버 측 데이터 손실(디스크 장애, 운영자 수동 wipe 등)이 없는 한 영구 유효합니다. 정기 점검·코드 업데이트 등 일상적인 배포 작업에는 키가 영향받지 않습니다. 키 폐기·갱신이 필요한 경우(예: 노출 의심) 클라이언트가 `DELETE /v1/auth/api-key` 후 재발급하면 됩니다.

---

## 7. 키 회전 / 폐기 (Rotation & Revocation)

### 7.1 정기 회전 (계약상 권장)

90일 또는 180일 주기로:
```bash
# 클라이언트 측 — 직접 실행
TOKEN=$(curl -s -X POST $URL/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=client-acme&password=$PW" | jq -r .access_token)

# 1) 기존 키 폐기 (DB 즉시 무효화 + Redis 캐시 즉시 purge)
curl -X DELETE $URL/v1/auth/api-key -H "Authorization: Bearer $TOKEN"

# 2) 새 키 발급
NEW_KEY=$(curl -X POST $URL/v1/auth/api-key \
  -H "Authorization: Bearer $TOKEN" | jq -r .api_key)

# 3) 자신의 .env / Vault 업데이트 후 서비스 reload
```

발급은 사용자당 1시간에 10회 제한 (`auth:apikey_issue:{username}` Redis 카운터).

### 7.2 누출 의심 시 긴급 대응

| 주체 | 액션 |
|---|---|
| 클라이언트 | `DELETE /v1/auth/api-key` 즉시 (자체 가능) |
| 운영자 | `last_used_at` + audit log에서 이상 IP 패턴 확인, 필요 시 사용자 status를 `suspended`로 변경 |
| 운영자 | 클라이언트에게 새 임시 비밀번호 전달 (위 5.1 절차 반복) |

### 7.3 비정상 의심 신호

운영자가 정기 모니터링할 항목:
- `last_used_at`이 30일 이상 정지된 키 → 미사용 또는 누출 후 폐기됨, 운영자 확인 후 폐기
- 단시간에 다수의 다른 IP에서 호출 → 누출 가능성, audit log 추적
- 한 키에서 비정상적으로 높은 RPS / 실패율 → 봇/스크립트 의심
- 발급 후 한 번도 사용 안 된 키가 30일 경과 → 수신 실패 의심
