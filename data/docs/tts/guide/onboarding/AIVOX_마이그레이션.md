# AIVOX TTS → Qwen3-TTS Enterprise 마이그레이션 가이드

> 외부 개발자 대상. 기존 AIVOX TTS API 클라이언트를 신규 시스템으로 옮기는 방법.
> 본 문서는 외부 개발자에게 그대로 전달 가능합니다.

---

## 1줄 요약

**URL host, API key, voice 이름** — 이 3개만 바꾸면 기존 코드가 거의 그대로 동작합니다.

---

## 1. 변경 없음 ✅

다음 항목들은 AIVOX 시절 코드 그대로 사용하면 됩니다.

| 항목 | 동일 유지 |
|------|---------|
| URL 경로 | `POST /v1/audio/speech` |
| 인증 헤더 | `X-API-Key: sk-...` |
| 요청 필드명 | `input`, `voice`, `language`, `stream`, `response_format`, `task_type`, `speed` |
| 언어 키워드 | `"Korean"`, `"English"`, `"Auto"` 등 |
| `response_format` | `wav`, `mp3`, `flac`, `ogg`, `pcm` |
| 응답 Content-Type | 동일 |
| `task_type` | `"Base"`, `"CustomVoice"` |
| HTTP 스트리밍 동작 | `stream=true` 시 Chunked Transfer 동일 |

---

## 2. 한 번만 바꿔야 할 것 🔄

### 2-1. URL host

```diff
- https://aivox.example.com/v1/audio/speech
+ https://qwen3-tts.example.com/v1/audio/speech
```

### 2-2. API key

신규 시스템에서 발급받은 `sk-` 토큰으로 교체. 헤더 이름(`X-API-Key`)은 그대로.

```diff
- X-API-Key: sk-aivox-abc123...
+ X-API-Key: sk-qwen3-xyz789...
```

> 💡 `Authorization: Bearer sk-...` 형식도 허용됩니다. 기존 OpenAI SDK 사용자도 그대로 사용 가능.

### 2-3. Voice 이름

> ⚠️ **여기가 가장 큰 변경점입니다.** AIVOX의 프리셋 이름(Sohee, Dylan 등)은 더 이상 존재하지 않습니다.

신규 시스템은 더 많은 voice를 지원하지만 ID 체계가 다릅니다. **자동 매핑하지 않는 이유는, 같은 이름인데 목소리가 다르면 더 큰 혼란을 만들기 때문**입니다. (Sohee를 임의로 다른 한국어 여성 voice에 매핑하면 톤/억양이 달라 "음성 품질이 떨어졌다"는 클레임의 원인이 됩니다.)

> 💡 **v0.9.0 기준 voice 구성**: 총 **27개** 활성 voice (= vLLM 22개 + Supertonic 5개). vLLM은 24kHz 고음질 + ICL 음성 복제 지원, Supertonic은 44.1kHz + 21개 추가 언어 지원. `GET /v1/audio/voices` 응답의 `_meta.by_engine` 으로 실시간 카운트 확인 가능.

#### 사용 가능 voice 조회

```bash
# 전체 목록
curl -H "X-API-Key: sk-..." https://qwen3-tts.example.com/v1/audio/voices

# 엔진 필터
curl -H "X-API-Key: sk-..." "https://qwen3-tts.example.com/v1/audio/voices?engine=vllm"
curl -H "X-API-Key: sk-..." "https://qwen3-tts.example.com/v1/audio/voices?engine=supertonic"

# 언어 필터 (AIVOX 호환 키워드 지원)
curl -H "X-API-Key: sk-..." "https://qwen3-tts.example.com/v1/audio/voices?language=Korean"

# WebSocket /ws/tts 에서 쓸 수 있는 voice만 (Supertonic 제외)
curl -H "X-API-Key: sk-..." "https://qwen3-tts.example.com/v1/audio/voices?ws_only=true"
```

응답 예시:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen_kor_female_01_5s",
      "object": "voice",
      "name": "Korean Female 01 (5s)",
      "language": "ko",
      "gender": "female",
      "description": "아이유 음성 목소리 (5초 트림본)",
      "engine": "vllm",                          // ← 엔진 종류
      "sample_rate": 24000,                      // ← 24kHz (vLLM)
      "supports_icl": true,                      // ← use_icl 사용 가능
      "has_ref_audio": true,                     // ← Base task_type 사용 가능
      "ws_supported": true,                      // ← /ws/tts 에서 사용 가능
      "ready": true,                             // ← 실제 합성 가능 상태
      "sample_audio_url": "/samples/female/kor_IU_5s.wav",   // ← 🔊 미리듣기 URL
      "supports_all_languages": false
    },
    {
      "id": "supertonic_m1",
      "object": "voice",
      "name": "남성 1 (M1)",
      "language": "ko",
      "gender": "male",
      "engine": "supertonic",                    // ← Supertonic 엔진
      "sample_rate": 44100,                      // ← 44.1kHz (HTTP에서 처리 필요)
      "supports_icl": false,                     // ← Supertonic은 ICL 미지원
      "has_ref_audio": false,
      "ws_supported": false,                     // ← /ws/tts 에서 400 에러
      "ready": true,                             // ← 내장 스타일 (ref_audio 없어도 OK)
      "sample_audio_url": null,                  // ← 내장 스타일은 미리듣기 URL 없음
      "supports_all_languages": true             // ← 31개 언어 모두 합성 가능
    }
    // ...
  ],
  "_meta": {
    "total": 27,
    "ready_count": 25,                           // 준비 voice 수. 미준비 voice 는 이제 목록에서 제외돼 total ≈ ready_count (준비+활성만 반환)
    "icl_count": 22,                              // ICL 가능 vLLM voice 수
    "with_sample_count": 22,                     // 미리듣기 URL 있는 voice 수
    "by_engine": {"vllm": 22, "supertonic": 5},
    "ws_supported_count": 22,
    "filters_applied": {"engine": null, "language": null, "ws_only": false},
    "notes": [
      "engine='vllm': 24kHz, supports ICL/temperature/top_p/instructions-like options",
      "engine='supertonic': 44.1kHz CPU/ONNX. supports 21 extra languages (vi/ar/hi/...)",
      "ws_supported=false voices return 400 in /ws/tts (use HTTP /v1/audio/speech instead)",
      "sample_audio_url: GET that URL (no auth) to download a 3-15s ref clip and preview the voice"
    ]
  }
}
```

#### 각 voice 메타 필드 의미 (중요)

| 필드 | 의미 | 클라이언트 의사결정 |
|------|------|------------|
| `engine` | `vllm` (GPU, 한·영·중·일 외 6개 언어) / `supertonic` (CPU, 추가 21개 언어) | 외부에 노출할 voice 선택 |
| `sample_rate` | 24000 또는 44100 | PCM 디코딩 시 필수. 두 엔진 혼용 시 클라이언트가 sample rate 맞춰야 함 |
| `supports_icl` | `use_icl: true` 전달 가능 여부 | Base task_type / 음성 복제 사용 시 확인 |
| `has_ref_audio` | 사전 등록된 ref_audio 파일 존재 여부 | ICL 사용 가능 voice 선별 |
| `ws_supported` | WebSocket `/ws/tts` 에서 사용 가능 여부 | WS 사용 클라이언트는 `ws_only=true` 필터 권장 |
| `ready` | 실제 합성 가능 상태 (ref_audio 파일 존재 등) | **미준비(ready=false) voice 는 이제 서버가 목록에서 제외** — `/v1/audio/voices` 는 준비+활성 voice 만 반환(`/v1/voices` 와 동일 카탈로그). 따라서 응답 voice 는 사실상 항상 합성 가능 |
| **`sample_audio_url`** | **🔊 미리듣기 URL** (예: `/samples/female/kor_IU_5s.wav`) | 외부 개발자 UI에서 voice 선택 전 미리듣기 제공. `GET https://<host>{sample_audio_url}` (인증 불필요). null 이면 미리듣기 불가 (Supertonic builtin 등) |
| `supports_all_languages` | Supertonic 만 true. 31개 언어 자유 합성. | 다국어 입력 처리 시 라우팅 결정 |

#### 시나리오별 voice 조회 권장 패턴

```bash
# 한국어 vLLM voice만 (콜센터/안내 시스템에 가장 흔함)
GET /v1/audio/voices?engine=vllm&language=Korean

# 다국어 지원 필요 (베트남어/아랍어 등은 Supertonic만 가능)
GET /v1/audio/voices?engine=supertonic

# 실시간 WS 클라이언트 (Supertonic 자동 제외)
GET /v1/audio/voices?ws_only=true

# ICL 음성 복제 가능 voice (백엔드에서 has_ref_audio 필터링)
GET /v1/audio/voices?engine=vllm
# → 응답 받은 뒤 클라이언트에서 supports_icl=true 만 사용
```

#### 🔊 voice 미리듣기 (UI에서 활용)

각 voice의 `sample_audio_url` 을 그대로 GET 하면 3~15초 ref 클립 다운로드. 외부 개발자의 voice 선택 UI에 재생 버튼을 달 수 있음.

```javascript
// JavaScript 예시: voice 카드 + 미리듣기 버튼
const res = await fetch("https://qwen3-tts.example.com/v1/audio/voices?engine=vllm&language=Korean", {
  headers: {"X-API-Key": "sk-..."}
});
const { data: voices } = await res.json();

voices.forEach(v => {
  const card = document.createElement("div");
  card.innerHTML = `
    <h3>${v.name}</h3>
    <p>${v.description}</p>
    <p>language: ${v.language}, gender: ${v.gender}</p>
    ${v.sample_audio_url
      ? `<audio controls src="https://qwen3-tts.example.com${v.sample_audio_url}"></audio>`
      : `<small>(미리듣기 없음 — 내장 스타일)</small>`}
    <button onclick="selectVoice('${v.id}')">선택</button>
  `;
  document.body.appendChild(card);
});
```

```python
# Python 예시: voice 미리듣기 다운로드
import requests
voices = requests.get(
    "https://qwen3-tts.example.com/v1/audio/voices?engine=vllm",
    headers={"X-API-Key": "sk-..."}
).json()["data"]

for v in voices:
    if v["sample_audio_url"]:
        # 미리듣기 다운로드
        clip = requests.get(f"https://qwen3-tts.example.com{v['sample_audio_url']}")
        with open(f"preview_{v['id']}.wav", "wb") as f:
            f.write(clip.content)
        print(f"{v['name']} ({v['language']}/{v['gender']}) — preview saved")
```

#### 기존 voice → 신규 voice 추천 매핑 (참고용)

| AIVOX voice | 권장 신규 voice 후보 |
|------------|----------------|
| **Sohee** (한국어 여성) | `qwen_kor_female_01_5s` (아이유 5s), `qwen_kor_female_02_5s`, 또는 Qwen3-TTS 내장 `qwen_st_f1` ~ `f5` |
| **한국어 남성 (단일)** | `qwen_kor_male_01_5s`, 또는 Qwen3-TTS 내장 `qwen_st_m1` ~ `m5` |
| **Dylan / Eric / Ryan / Aiden** (영어 남성) | `supertonic_m1` ~ `m5` (31개 언어 지원) |
| **Vivian / Serena** (중국어 여성) | `supertonic_f1` ~ `f5` (다국어 통합) |
| **Uncle_Fu** (중국어 남성) | `supertonic_m1` ~ `m5` (다국어 통합) |
| **Ono_Anna** (일본어 여성) | `supertonic_f1` ~ `f5` (다국어 통합) |

> ⚠️ **운영 voice 카탈로그가 우선**: 위 표는 참고용이며, 실제 사용 가능한 voice는 항상 `GET /v1/voices?enabled_only=true` 응답을 기준으로 매핑하세요. 운영자가 비활성화(`enabled: false`)한 voice는 응답에 포함되지 않으며, 호출 시 422 에러로 거부됩니다.
>
> 데모 음성을 직접 들어보고(`GET /v1/voices/{voice_id}/preview`) 가장 가까운 것을 골라주세요.

#### 보너스: 추가 지원 언어

AIVOX는 4개 언어(한/영/중/일)만 지원했지만, 신규 시스템은 더 넓은 언어를 지원합니다. vLLM(고품질 ICL)은 **10개 언어**(ko/en/zh/ja/de/fr/es/it/pt/ru)를 네이티브 지원하고, Supertonic 엔진이 아래 **추가 21개 언어**를 더 커버합니다.

베트남어, 아랍어, 힌디어, 네덜란드어, 폴란드어, 스웨덴어, 터키어, 우크라이나어, 덴마크어, 핀란드어, 헝가리어, 체코어, 슬로바키아어, 루마니아어, 불가리아어, 크로아티아어, 슬로베니아어, 리투아니아어, 라트비아어, 에스토니아어, 그리스어.

해당 언어는 `supertonic_*` voice가 31개 언어를 통합 지원합니다. `language` 필드만 지정하면 자동 라우팅됩니다.

---

## 3. 동작이 바뀐 항목 ⚠️

응답 헤더 `X-Compat-Warning`으로 변환/무시된 항목을 알려줍니다.

| 입력 필드 | 신규 시스템 동작 |
|---------|------------|
| `instructions` (자유 텍스트 운율 지시) | **무시.** 응답 헤더 `X-Compat-Warning` 으로 안내. **대체 방안**: `temperature`/`top_p`/`speed`를 직접 지정 (아래 운율 제어 표 참조). ※ 신규 시스템은 `style_instruction` 필드도 받지만 **현재 엔진이 Qwen3-TTS Base 모델이라 자연어 감정 지시는 영향이 매우 적습니다** — Base 는 화자 임베딩 기반 복제에 특화된 변형으로 자연어 톤 제어는 설계상 거의 무시됩니다 (모델 카드 확인). 따라서 `instructions` 마이그레이션은 `style_instruction` 으로 옮기는 것이 아니라 아래 `temperature`/`top_p`/`speed` 프리셋으로 옮기는 것이 권장 경로입니다. |
| `seed` | **무시.** 재현성이 필요한 워크플로우는 캐시 활용 권장. |
| `n_candidates: 3` (Best-of-N) | **1로 강제.** UTMOS 기반 자동 선택 미지원. |

---

## 3-2. 신규 고급 옵션 (선택 사용) ✨

신규 시스템은 AIVOX보다 훨씬 풍부한 합성 제어 옵션을 제공합니다. **기존 AIVOX 코드는 그대로 두어도 동작**하지만, 다음 필드를 추가하면 합성 품질·안전성을 더 정밀하게 제어할 수 있습니다.

### 운율 제어 (vLLM 전용)

```json
{
  "input": "안녕하세요",
  "voice": "qwen_kor_female_01_5s",
  "temperature": 0.65,   // 0.1~2.0 - 낮을수록 안정, 높을수록 표현 풍부. platform default 0.65
  "top_p": 0.85,         // 0.1~1.0 - 안정성 우선. platform default 0.85
  "speed": 1.0           // 0.5~2.0
}
```

### 감정 권장 프리셋 (`temperature`/`top_p`/`speed` 조합)

> ⚠️ **`emotion` 필드는 현재 신규 시스템에 구현되어 있지 않습니다.** 아래 표의 권장값을 `temperature`/`top_p`/`speed` 3개 필드로 명시적으로 전달하세요. (단축 alias 도입 여부는 추후 로드맵 참조.)

자유 텍스트 `instructions` 대신 다음 권장 조합을 직접 합성 body에 포함하면 비슷한 감정 톤을 얻을 수 있습니다.

```json
{
  "input": "안녕하세요. 차분하게 안내드립니다",
  "voice": "qwen_kor_female_01_5s",
  "temperature": 0.7,
  "top_p": 0.85,
  "speed": 0.9
}
```

| 감정 톤 | temperature | top_p | speed | 비고 |
|------|------------|-------|-------|-----|
| default | 1.0 | 1.0 | 1.0 | AIVOX 기본 — 환각 빈도↑ |
| **권장 기본** | **0.65** | **0.85** | **1.0** | platform default |
| energetic (활기) | 1.3 | 0.95 | 1.1 | 다채로운 운율 |
| soft (부드러움) | 0.75 | 0.88 | 0.92 | 안정 + 약간 느림 |
| calm (차분) | 0.7 | 0.85 | 0.9 | 안내·상담 |
| whisper (속삭임) | 0.5 | 0.8 | 0.85 | 정적·조용한 톤 |

### 품질·안전 토글 (vLLM)

| 필드 | 효과 |
|------|------|
| `use_silence_guard` | 후미 환각(반복/노이즈) 차단 |
| `use_silence_lookback` | 초성 잘림 방지 (단어 첫 자음 보존) |
| `high_fidelity` | 24→48kHz 업샘플 (배치 합성 시 음질↑) |
| `smart_prosody` | 구두점·문맥 기반 자동 운율 정규화 |

### 스트리밍 튜닝 (저지연 최적화)

| 필드 | 효과 |
|------|------|
| `chunk_size` | 스트리밍 청크 크기 (bytes, 1KB~512KB). 클라이언트 buffer 크기에 맞춰 조정. stream=true 일 때만 의미. |
| `initial_codec_chunk_frames` | 첫 코덱 청크 프레임 수 (4~64). 작을수록 TTFB 단축, 클수록 안정성↑. 기본 4 (TTFB 우선). |

### 공통 후처리

| 필드 | 효과 |
|------|------|
| `use_micro_fade` | 청크 경계 'tick' 노이즈 제거 |
| `use_normalize` | 피크 정규화 (loudness 평준화) |
| `use_split` | 긴 텍스트를 문장 단위로 자동 분할하여 병렬 합성 |
| `silence_padding_ms` | 합성 시작 전 무음 패딩 (ms, cold-start glitch 완화) |
| `max_silence_ms` | 합성 종료 판정 무음 임계 (ms) |
| `hardware_profile` | `standard`/`phone_line`/`small_speaker`/`studio` DSP 프리셋 |

### 한국어 ITN (숫자/단위 정규화)

```json
{
  "input": "결제 금액은 1,250원이고 USD 변환 시 0.85달러입니다.",
  "voice": "qwen_kor_female_01_5s",
  "use_itn": true,
  "itn_domain": "finance"   // standard/finance/commerce/healthcare/tech/education/media/aicc/mobility/game/casual (11종)
}
```

### Supertonic 엔진 전용

```json
{
  "input": "Hello world",
  "voice": "supertonic_m1",       // Supertonic 다국어 voice (m1~m5 / f1~f5)
  "supertonic_total_steps": 6,    // 1~24 - 기본 6 (속도 우선), 8 균형, 10~12 고품질
  "supertonic_style": "M3"        // M1~M5 / F1~F5
}
```

### 시스템 기본값 (서버가 자동 적용)

외부 개발자가 **고급 옵션을 안 보내면 적용되는 시스템 기본값**입니다. 신규 솔루션은 운영 분석 결과를 반영해 균형 잡힌 값으로 튜닝되어 있어, 대부분의 사용자는 **그대로 두는 것이 베스트**입니다.

| 필드 | 기본값 | 의미 |
|------|------|------|
| `temperature` | **0.65** | 환각 방지 + 자연성 균형 (AIVOX 기본 1.0보다 안전) |
| `top_p` | **0.85** | 안정성 우선 (AIVOX 기본 1.0보다 명확) |
| `speed` | 1.0 | 표준 속도 |
| `use_icl` | **true** | 음성 클로닝 ON — ref_audio 있는 voice는 자동 클로닝, ref_audio 없는 voice는 no-op |
| `use_silence_guard` | **true** (v0.9.8 ON) | 환각/noise loop 자동 차단. `silence_hangover_ms=300` + `max_silence_ms=700` 조합으로 false-positive 방지 |
| `use_silence_lookback` | **false** | 초성 보존 OFF (필요 시 `true` 명시) |
| `use_micro_fade` | **true** | 청크 노이즈 제거 ON |
| `high_fidelity` | true | 고품질 DSP ON (배치 시 24→48kHz 업샘플) |
| `smart_prosody` | false | 자동 운율 정규화 OFF (필요 시 `true` 명시) |
| `use_itn` | true | 한국어 숫자 정규화 ON (Korean만 적용) |
| `use_normalize` | false | 피크 정규화 OFF (음질 손실 방지) |
| `use_split` | true | 자동 분할 ON (platform default; 단문은 분할 없이 통과) |
| `itn_domain` | `standard` | 일반 도메인 |
| `hardware_profile` | `standard` | 표준 DSP |
| `silence_padding_ms` | 200 | 시작 무음 200ms |
| `max_silence_ms` | **700** (v0.9.8 단축 3000→700) | silence_guard 종료 임계 — 빠른 stop |
| `silence_hangover_ms` | **300** | silence_guard 카운트 전 유예 — vocal pause 보호 |
| `max_sentence_audio_ms` | **30000** | hallucination guard — 1문장 최대 30초 |
| `trailing_silence_ms` | **250** (v0.9.8 단축 1500→250) | trailing padding — silence_guard ON 환경 짧게 |
| `inter_sentence_padding_ms` | **120** (v0.9.8 단축 500→120) | 문장 간 포즈 (구두점 기반 자동 스케일) |
| `default_use_silence_guard` | **true** (v0.9.8 ON) | 무음 가드 기본 ON — hallucination 자동 차단 |
| `audio_format` | wav | 손실 없는 PCM 컨테이너 |
| `chunk_size` | 4096 | 4KB 스트리밍 청크 |
| `initial_codec_chunk_frames` | **4** | 첫 패킷 4 프레임 (TTFB 우선) |
| `supertonic_total_steps` | **6** | 속도 우선 (8 균형, 10~12 고품질) |
| `supertonic_style` | M1 | 남성 1번 |

> 💡 **AIVOX보다 더 안전한 기본값**: AIVOX는 `temperature=1.0, top_p=1.0`이라 가끔 발음 꼬임/환각 발생. 신규 시스템은 `0.65/0.85`로 미리 안정화. 외부 개발자가 1.0/1.0을 명시적으로 보내면 AIVOX와 동일한 동작 가능.
>
> 🛠️ **운영자가 변경 가능**: 위 기본값은 platform 운영자(super_admin)가 Admin 패널 또는 `PATCH /v1/admin/settings`로 언제든지 변경할 수 있습니다. 통합 시점의 실제 default가 궁금하면 `GET /v1/capabilities` 또는 `GET /v1/admin/settings`로 조회하세요. 통합 코드가 특정 동작을 가정한다면 해당 옵션을 요청 body에 **명시적으로** 전달하는 것을 권장합니다.

### 시나리오별 권장 프리셋 (구체적 값)

#### 🎙️ 콜센터 ARS / 짧은 안내 응답 (TTFB 우선)
```json
{
  "input": "주문번호 12345로 접수되었습니다.",
  "voice": "qwen_kor_female_01_5s",
  "temperature": 0.7,                    // 차분한 안내 톤
  "top_p": 0.85,
  "speed": 0.9,
  "use_silence_guard": true,
  "use_silence_lookback": true,
  "silence_padding_ms": 100,             // 시작 지연 최소화
  "use_normalize": false,
  "hardware_profile": "phone_line",      // 좁은 대역폭 최적화
  "initial_codec_chunk_frames": 4        // TTFB 단축
}
```
**이유**: ARS는 즉시 응답이 최우선. `phone_line` 프로필이 통화 음질에 맞춰 저주파 강조. `temperature=0.7`이 차분한 안내 톤. 스트리밍이 필요하면 `/v1/tts/synthesize/stream` 엔드포인트 또는 WebSocket 사용.

#### 📢 안내 방송 / 매장 멘트 (품질 우선)
```json
{
  "input": "잠시 후 폐점 시간입니다. 계산대를 이용해 주세요.",
  "voice": "qwen_kor_female_02_5s",
  "temperature": 0.85,
  "top_p": 0.92,
  "high_fidelity": true,                // 24→48kHz 업샘플
  "use_normalize": true,                // 라우드니스 평준화 (방송 표준)
  "hardware_profile": "studio",
  "use_split": true,                    // 긴 멘트는 자동 분할
  "stream": false                        // 한 번에 받아서 재생
}
```
**이유**: 매장 스피커는 음질 차이가 크게 들림. `studio` 프로필이 풍부한 주파수 범위 유지. `use_normalize`로 매번 같은 음량.

#### 📚 오디오북 / 긴 콘텐츠 (자연성 우선)
```json
{
  "input": "(긴 책 내용 5000자)",
  "voice": "qwen_kor_male_03_5s",
  "temperature": 1.0,                   // 다채로운 운율
  "top_p": 0.95,
  "high_fidelity": true,
  "use_split": true,                    // 문장 단위 분할 + 병렬 합성
  "use_silence_lookback": true,         // 단어 첫 자음 보존 (오디오북 중요)
  "smart_prosody": true,
  "silence_padding_ms": 500             // 챕터 시작 여유
}
```
**이유**: 오디오북은 단조로움 피해야 함. temperature 높여 운율 다양성. 5000자 입력은 `use_split: true` 필수 (병렬 합성으로 5배 빠름).

#### 🎮 게임 NPC 대사 (캐릭터 표현)
```json
{
  "input": "안녕, 모험가여! 어떤 일을 맡길까?",
  "voice": "qwen_kor_male_02_5s",
  "temperature": 1.3,                   // 표현력↑
  "top_p": 0.95,
  "speed": 1.1,                          // 활기찬 톤
  "use_silence_guard": true,
  "supertonic_style": "M3",              // (Supertonic 사용 시) 활달한 남성 톤
  "hardware_profile": "small_speaker"
}
```
**이유**: NPC는 생동감 필요. `temperature=1.3`이 표현력 높임. `small_speaker` 프로필이 게임기/노트북 스피커에 최적.

#### ⚡ 저지연 실시간 (WebSocket 라이브)
```javascript
ws.send(JSON.stringify({
  type: "tts.request",
  text: "...",
  speaker: "qwen_kor_female_01_5s",
  silence_padding_ms: 0,                // 무음 패딩 제거
  use_normalize: false,                  // 후처리 스킵
  initial_codec_chunk_frames: 4,         // 최소 프레임 (TTFB ~200ms)
  hardware_profile: "standard"
}));
```
**이유**: 실시간 합성은 TTFB가 모든 것. `silence_padding_ms: 0`으로 시작 지연 제거.

#### 💰 금융 / 결제 안내 (정확한 발음)
```json
{
  "input": "결제 금액 1,250원이 승인되었습니다. 카드번호 끝 4자리 5678입니다.",
  "voice": "qwen_kor_female_01_5s",
  "temperature": 0.7,                   // 차분한 안내
  "top_p": 0.85,
  "speed": 0.95,
  "use_itn": true,
  "itn_domain": "finance",              // 금융 도메인 사전
  "use_silence_guard": true
}
```
**결과**: "천이백오십원이 승인되었습니다. 카드번호 끝 사자리 오육칠팔입니다." (도메인 사전이 숫자/통화 정확히 발화)

#### 🏥 의료 안내
```json
{
  "input": "혈압 120/80은 정상 범위입니다.",
  "voice": "qwen_kor_female_02_5s",
  "temperature": 0.7,
  "top_p": 0.85,
  "speed": 0.95,
  "use_itn": true,
  "itn_domain": "healthcare"            // 의료 도메인 (혈압/단위 등)
}
```

---

### 🔧 트러블슈팅 — 증상별 파라미터 조정

| 증상 | 조정 |
|------|------|
| 발음이 꼬이거나 이상한 단어 생성 | `temperature: 0.7` 또는 0.6 으로 낮춤 |
| 끝 부분에 이상한 소리/숨소리 | v0.9.8 기본 ON (`use_silence_guard: true`) + `max_silence_ms: 700` (단축 3000→700) + `silence_hangover_ms: 300` + hallucination guard `max_sentence_audio_ms: 30000` 조합으로 자동 차단 |
| 첫 자음 잘림 ("녕하세요" 같은) | `use_silence_lookback: true` (기본 ON) + `silence_padding_ms: 500` 증가 |
| 청크 경계 "틱" 노이즈 | `use_micro_fade: true` (기본 ON) |
| 같은 입력에 매번 다른 결과 | 신규 시스템은 seed 미지원 — 캐시를 활용 (동일 입력 = 동일 출력) |
| 너무 단조로움 | `temperature: 1.1`, `top_p: 0.95`, `speed: 1.1` |
| 너무 빠름/느림 | `speed: 0.9` (느림) 또는 `speed: 1.15` (빠름) |
| 합성이 너무 오래 걸림 | `high_fidelity: false`, `supertonic_total_steps: 5` (Supertonic 시) |
| 통화 음질에 적합 안 함 | `hardware_profile: "phone_line"` |
| 한국어 숫자가 영어로 읽힘 | `use_itn: true`, `language: "Korean"` (또는 "ko") |
| ITN이 의도와 다른 발음 | `itn_domain` 명시: standard/finance/commerce/healthcare/tech/... |

---

### 📐 Quick Start — 최소 권장 호출

처음 마이그레이션할 때 **가장 안전한 시작값** (이 한 가지만 외워도 됨):

```json
{
  "input": "합성할 텍스트",
  "voice": "qwen_kor_female_01_5s",
  "language": "Korean",
  "response_format": "wav"
}
```

이것만으로 시스템이 균형 잡힌 기본값을 모두 자동 적용 (temperature 0.65, top_p 0.85 등). 90% 사용 사례에 충분.

운율 제어 필요할 때만 추가:
```json
{
  ...
  "temperature": 0.7,   // 차분 톤
  "top_p": 0.85,
  "speed": 0.9
}
```

> 🎯 **권장**: 첫 마이그레이션 시 위 최소 호출로 테스트 → 결과가 만족스러우면 그대로 사용. 특정 시나리오(콜센터/오디오북 등)에 맞춰 추가 옵션은 위 시나리오 표 참조.

> Swagger UI(`GET /docs`)에서 각 필드의 상세 설명과 유효 범위 확인 가능.

---

## 4. 거절되는 호출 🚫

다음 호출은 명확한 400 에러를 반환합니다.

### 4-1. inline `ref_audio`

```jsonc
// 기존 AIVOX 호출 — 신규 시스템에서는 거절됨
{
  "input": "안녕하세요",
  "task_type": "Base",
  "ref_audio": "data:audio/wav;base64,UklGRiQ..."  // ❌ 400 에러
}
```

**대체 방법**: 음성을 사전 등록 후 voice ID로 호출.

```bash
# 1. 한 번만 등록 (관리자 권한 필요)
curl -X POST -H "X-API-Key: sk-admin-..." \
  -F "name=내목소리" -F "language=Korean" -F "gender=female" \
  -F "ref_text=안녕하세요 저는..." \
  -F "audio_file=@my_voice.wav" \
  https://qwen3-tts.example.com/v1/admin/voices

# 2. 이후엔 voice ID로 호출
curl -X POST -H "X-API-Key: sk-..." \
  -d '{"input":"...","voice":"my_voice_id"}' \
  https://qwen3-tts.example.com/v1/audio/speech
```

### 4-2. `response_format: "opus"` 또는 `"aac"`

미지원. `wav`, `mp3`, `flac`, `ogg`, `pcm` 중 사용.

### 4-3. `task_type: "VoiceDesign"`

자연어 음성 디자인은 현재 미지원. `Base` (ICL 음성 복제) 또는 `CustomVoice` (프리셋) 사용.

---

## 5. WebSocket 실시간 합성 (`/ws/tts`)

기존 AIVOX의 WebSocket 프로토콜이 그대로 지원됩니다. **호스트만 바꾸면 작동**합니다.

### 5-1. 접속

```javascript
// Before
const ws = new WebSocket("wss://aivox.example.com/ws/tts?api_key=sk-aivox-old");

// After
const ws = new WebSocket("wss://qwen3-tts.example.com/ws/tts?api_key=sk-new");
```

API key는 `?api_key=` 또는 `?token=` 둘 다 허용. Sec-WebSocket-Protocol 두 번째 값도 지원.

### 5-2. 메시지 포맷 (동일 유지)

**Client → Server**:
```javascript
// 합성 요청
ws.send(JSON.stringify({
  type: "tts.request",
  text: "안녕하세요",
  language: "Korean",
  speaker: "qwen_kor_female_01_5s",   // ← voice ID만 신규로
  volume: 1.5,                          // 동일
  style: "neutral"                      // 무시됨 (5순위 작업에서 매핑 예정)
}));

// 중단
ws.send(JSON.stringify({ type: "tts.stop" }));

// keep-alive
ws.send(JSON.stringify({ type: "ping" }));
```

**Server → Client**:
```javascript
// 합성 시작
{ "type": "tts.start" }

// PCM 청크 (binary, 정확히 9,600 bytes = 200ms @ 24kHz 16-bit mono)
<Binary data>

// 합성 완료
{
  "type": "tts.end",
  "duration": 3.456,    // 초
  "cached": false,
  "elapsed": 2150,      // 밀리초
  "trace_id": "..."
}

// keep-alive 응답
{ "type": "pong" }

// 에러
{ "type": "error", "message": "...", "code": 429 }
```

### 5-3. 변경된 동작

| 항목 | 처리 |
|------|------|
| `style` (neutral/conversational/expressive/formal) | **무시됨**. 대신 `temperature`/`top_p`/`speed`를 직접 지정 (3-2 절 권장 톤 표 참조) |
| Supertonic voice (`supertonic_*`) | **400 에러**. 44.1kHz sample rate가 AIVOX의 24kHz 가정과 불일치하여 사전 차단 |
| 동일 세션 내 새 `tts.request` | **이전 요청 자동 취소** 후 새 요청 시작 (AIVOX와 동일) |

### 5-4. 신규 고급 옵션 (선택 사용)

HTTP API (`POST /v1/audio/speech`) 와 동일한 16개 고급 옵션을 `tts.request` 메시지에 함께 보낼 수 있습니다.

```javascript
ws.send(JSON.stringify({
  type: "tts.request",
  text: "차분하게 안내드립니다.",
  speaker: "qwen_kor_female_01_5s",
  language: "Korean",

  // 신규 고급 옵션 (모두 선택)
  temperature: 0.7,             // 0.1~2.0 (calm 톤)
  top_p: 0.85,                  // 0.1~1.0
  speed: 0.9,                   // 0.5~2.0
  use_silence_guard: true,      // 환각 차단
  use_silence_lookback: true,   // 초성 보존
  use_itn: true,                // 한국어 숫자 정규화
  itn_domain: "finance",        // 도메인 사전
  silence_padding_ms: 100,      // 시작 무음 ms
  hardware_profile: "phone_line",
  initial_codec_chunk_frames: 4,  // TTFB 최소화
  // ... 등
}));
```

**제외된 옵션 (WS에서 의미 없음)**: `audio_format`, `chunk_size`, `response_format`, `task_type`, `seed`, `n_candidates`, `ref_audio`, `supertonic_*` (모두 WS PCM 24kHz 고정 또는 호환 불가).

전체 옵션 의미는 **섹션 3-2** (HTTP 고급 옵션) 와 동일합니다.

### 5-4. 코드 예시 (Python `websockets`)

```python
import asyncio
import websockets
import json

async def synthesize_streaming():
    uri = "wss://qwen3-tts.example.com/ws/tts?api_key=sk-new"
    async with websockets.connect(uri) as ws:
        # 합성 요청
        await ws.send(json.dumps({
            "type": "tts.request",
            "text": "실시간 합성 테스트입니다.",
            "language": "Korean",
            "speaker": "qwen_kor_female_01_5s",
            "volume": 1.0,
        }))

        pcm_chunks = []
        async for msg in ws:
            if isinstance(msg, bytes):
                # 200ms PCM chunk (9,600 bytes)
                pcm_chunks.append(msg)
            else:
                data = json.loads(msg)
                if data["type"] == "tts.start":
                    print("합성 시작")
                elif data["type"] == "tts.end":
                    print(f"완료: {data['duration']}s ({data['elapsed']}ms)")
                    break
                elif data["type"] == "error":
                    print(f"에러: {data['message']}")
                    break

        # PCM → WAV 저장 등 후처리
        # (24kHz, 16-bit signed PCM, little-endian, mono)

asyncio.run(synthesize_streaming())
```

---

## 6. 품질 평가 API (`/v1/audio/speech/evaluate`)

> ⛔ **신규 시스템은 자체 품질 평가(UTMOS/CER/WER) 기능을 내장하지 않습니다.**

AIVOX의 `evaluate` 엔드포인트로 UTMOS/CER/WER 점수를 자동 측정하던 QA 파이프라인은 신규 시스템에서 **동작하지 않습니다**. 이유:
- 신규 시스템엔 UTMOS22, ASR 모델이 통합되어 있지 않음
- 합성 자체는 정상 작동하나 점수 산출 로직이 부재

**대안**:
- **합성만 필요한 경우**: 일반 `POST /v1/audio/speech` 사용 (음성은 정상 반환)
- **점수 측정이 꼭 필요한 경우**:
  - 외부 ASR/UTMOS 서버를 별도 구축하여 클라이언트에서 직접 평가
  - 또는 운영팀에 요청 → 향후 UTMOS22 + Qwen3-ASR 통합 검토 (별도 모델 추가 작업 필요)

---

## 7. 코드 예시

### Python (`requests`)

```python
import requests

# Before (AIVOX)
# response = requests.post(
#     "https://aivox.example.com/v1/audio/speech",
#     headers={"X-API-Key": "sk-aivox-old"},
#     json={"input": "안녕하세요", "voice": "Sohee", "language": "Korean"}
# )

# After (신규)
response = requests.post(
    "https://qwen3-tts.example.com/v1/audio/speech",   # ← URL host
    headers={"X-API-Key": "sk-qwen3-new"},              # ← key
    json={
        "input": "안녕하세요",
        "voice": "qwen_kor_female_01_5s",               # ← voice ID
        "language": "Korean",
    },
)
with open("out.wav", "wb") as f:
    f.write(response.content)

# 호환 경고 확인
if "X-Compat-Warning" in response.headers:
    print("Warning:", response.headers["X-Compat-Warning"])
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-qwen3-new",
    base_url="https://qwen3-tts.example.com/v1",
)

response = client.audio.speech.create(
    model="qwen3-tts",
    voice="qwen_kor_female_01_5s",
    input="안녕하세요",
    response_format="wav",
)
response.stream_to_file("out.wav")
```

### Node.js (`fetch`)

```javascript
const res = await fetch("https://qwen3-tts.example.com/v1/audio/speech", {
  method: "POST",
  headers: {
    "X-API-Key": "sk-qwen3-new",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    input: "안녕하세요",
    voice: "qwen_kor_female_01_5s",
    language: "Korean",
    stream: true,
  }),
});

// 스트리밍 처리
const writer = fs.createWriteStream("out.wav");
const reader = res.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  writer.write(value);
}
```

### cURL

```bash
curl -X POST https://qwen3-tts.example.com/v1/audio/speech \
  -H "X-API-Key: sk-qwen3-new" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요",
    "voice": "qwen_kor_female_01_5s",
    "language": "Korean",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

---

## 8. 마이그레이션 체크리스트

- [ ] 신규 시스템 host URL 확보
- [ ] 신규 `sk-...` API key 발급
- [ ] `GET /v1/audio/voices` 로 사용 가능 voice 목록 확인
- [ ] 기존 코드의 voice 이름을 신규 voice ID로 일괄 변경 (보통 1~3개)
- [ ] `instructions` / `seed` / `n_candidates` 사용 여부 확인 → 응답 헤더 `X-Compat-Warning` 모니터링
- [ ] `ref_audio` inline 사용 중이면 → 음성 사전 등록 워크플로우로 전환
- [ ] `evaluate` API 사용 중이면 → **사용 중단** (신규 시스템 미지원. 자체 ASR/UTMOS 평가가 필요하면 별도 평가 인프라 구축)
- [ ] 트래픽 일부(예: 10%)로 카나리아 검증 → 100% 전환

---

## 9. 신규 시스템에서만 가능한 추가 기능 (참고)

마이그레이션 이후 활용 가능한 신기능:

- **31개 언어**: 한/영/중/일 외 21개 추가 언어
- **자동 fallback**: vLLM 장애 시 Supertonic 엔진으로 자동 전환 (503 발생 빈도 ↓, 응답 헤더 `X-Engine-Fallback`)
- **WebSocket 양방향**: `WS /v1/tts/synthesize/ws` — 토큰 query/subprotocol 양쪽 모두 인증 지원
- **ITN 11개 도메인**: standard/finance/commerce/healthcare/tech 등 도메인 특화 발음 정규화
- **상세 모니터링**: `GET /v2/me/usage/daily`, `GET /v2/me/traces/{id}` 등 사용자 자체 통계 API

전체 API는 `GET /docs` (Swagger UI)에서 확인 가능합니다.

---

## 10. 문의

마이그레이션 중 막히는 부분이 있으면 응답 헤더 `X-Synthesis-Trace-ID` 값과 함께 운영팀에 문의해주세요. 해당 trace ID로 서버 측 단계별 지연/에러를 정확히 추적할 수 있습니다.
