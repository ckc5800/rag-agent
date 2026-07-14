# 이윤선 포트폴리오

![KakaoTalk_20240928_081641201.jpg](KakaoTalk_20240928_081641201.jpg)

### $\color{3399ff}{Lee}$ $\color{3399ff}{Yoon Seon}$

> Seoul, South Korea
E. yoon7829@gmail.com
> 

---

### $\color{3399ff}{Navigator }$

①   [Experience](%EC%9D%B4%EC%9C%A4%EC%84%A0%20%ED%8F%AC%ED%8A%B8%ED%8F%B4%EB%A6%AC%EC%98%A4%202050330b274180dda2dcc1fbd465f112.md) 

②   [Projects](%EC%9D%B4%EC%9C%A4%EC%84%A0%20%ED%8F%AC%ED%8A%B8%ED%8F%B4%EB%A6%AC%EC%98%A4%202050330b274180dda2dcc1fbd465f112.md) 

③   [Skills](%EC%9D%B4%EC%9C%A4%EC%84%A0%20%ED%8F%AC%ED%8A%B8%ED%8F%B4%EB%A6%AC%EC%98%A4%202050330b274180dda2dcc1fbd465f112.md) 

④   [Social Media](%EC%9D%B4%EC%9C%A4%EC%84%A0%20%ED%8F%AC%ED%8A%B8%ED%8F%B4%EB%A6%AC%EC%98%A4%202050330b274180dda2dcc1fbd465f112.md) 

⑤   [Others](%EC%9D%B4%EC%9C%A4%EC%84%A0%20%ED%8F%AC%ED%8A%B8%ED%8F%B4%EB%A6%AC%EC%98%A4%202050330b274180dda2dcc1fbd465f112.md)  

### $\color{3399ff}{About}$ $\color{3399ff}{Me}$

AI 및 kubernetes 경험을 지닌 6년 차 엔지니어입니다.

현재 미코그룹 AICESS 인공지능 연구소에서 음성 관련 업무를 담당하고 있으며, AI 관련 석사 학위를 보유하고 있습니다.

### $\color{3399ff}{Experience}$ $\color{3399ff}{Overview}$

- AICESS
    
    2025.04.23 ~ (현재)
    
- 인피닉
    
    2022.08.08 - 2025.04.22(2년 9개월)
    
- 이든티앤에스
    
    2022.04.04 - 2022.08.05(5개월)
    
- 큐헷지
    
    2020.09 - 2021.09 (1년 1개월)
    

### $\color{3399ff}{Launguage}$

- 영어 (중급) - IM1

# $\color{3399ff}{Experience}$

# AICESS (2025.04~ 재직중)

AI언어개발팀 (2025.09~)

- TTS(2025.09~)
    - QA 업무
    - TTS 데이터 정제
    - ZipVoice 모델 학습
        - [음성 샘플(1)](https://drive.google.com/file/d/1ZyhsduB2POS8jQTAKQE57XlmH8-fzt4L/view?usp=drive_link)
        - [음성 샘플(2)](https://drive.google.com/file/d/15gcy1TERrs6M9-ovSs_ExBAsNtJ9JPi2/view?usp=drive_link)
    - 사내 운영 테스트
    - 
    - 아키텍처
        
        
        | Browser ↔ FastAPI (스트리밍) | WebSocket | 8000 | binary PCM chunks + JSON done |
        | --- | --- | --- | --- |
        | Browser ↔ FastAPI (배치/REST) | HTTP | 8000 | JSON + WAV blob URL |
        | FastAPI → Qwen3 Engine | gRPC | 9104 | `StreamingSynthesizeRequest` |
        | FastAPI → Metrics API | HTTP | 9001 | WAV bytes → CER JSON |
        | FastAPI ↔ Redis | Redis protocol | 6379 | PCM/WAV bytes, stats |
        
        **캐시 키 구분** (배치/스트리밍 충돌 방지)
        
        ```markdown
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                         Browser (React + Vite)                           │
        │                                                                          │
        │  ┌──────────────┐ ┌─────────────┐ ┌────────────┐ ┌────────┐ ┌────────┐  │
        │  │SynthesisView │ │MeasureView  │ │  Settings  │ │ Admin  │ │ Stats  │  │
        │  │              │ │             │ │    View    │ │  View  │ │  View  │  │
        │  │ • WS stream  │ │ • HTTP synth│ │• User Mgmt │ │• Metrics│ │• Usage │  │
        │  │ • ITN/CER/   │ │ • playBlob()│ │• Voice CRUD│ │• Logs  │ │• CER   │  │
        │  │   Split 토글 │ │             │ │• 유지보수   │ │• Warmup│ │• Trend │  │
        │  │ • CER/REF/   │ │             │ │  모드      │ │• Cache │ │        │  │
        │  │   HYP 표시   │ │             │ │            │ │  Flush │ │        │  │
        │  └──────┬───────┘ └──────┬──────┘ └─────┬──────┘ └───┬────┘ └───┬────┘  │
        │  AudioPlayer (audio.ts)  │              │            │          │       │
        │  • scheduleWavChunk()    │         HTTP /v1/admin/*──┴──────────┘       │
        │  • Web Audio API         │                                              │
        └──────┬───────────────────┴──────────────────────────────────────────────┘
               │ WebSocket                   │ HTTP
               │ /v1/tts/synthesize/ws       │ /v1/*
               ▼                             ▼
        ┌──────────────────────────────────────────────────────────────────────────┐
        │                  [Nginx :80]  →  FastAPI :11100                          │
        │                                                                          │
        │  ws_synthesize.py          synthesize.py         admin.py               │
        │  ─────────────────         ─────────────         ────────               │
        │  1. JWT 인증               1. JWT 인증            • POST /warmup         │
        │  2. 유지보수 모드 체크      2. Redis 캐시 조회     • DELETE /cache        │
        │  3. ITN 정규화             3. vLLM 배치 합성       • GET /metrics         │
        │  4. 문장 분할              4. WAV 반환             • GET /sys-info        │
        │  5. 문장별 Redis 캐시 조회  5. Redis 캐시 저장      • GET /logs            │
        │  6. vLLM PCM 스트리밍                             • GET /audit-logs      │
        │  7. PCM → frontend (binary)                      • CRUD /users          │
        │  8. 복합 Redis 캐시 저장                           • CRUD /voices         │
        │  9. CER/WER 측정 (옵션)                           • GET/PATCH /settings  │
        │  10. done JSON 전송                                                      │
        │       {cer, wer, ref, hyp, deletions ...}                               │
        │                                                                          │
        │  pipeline/                                                               │
        │  ├── text_processor.py    ← ITN (숫자/단위 정규화)                        │
        │  └── sentence_splitter.py ← 문장 경계 분할 (ko:150자, en:300자)           │
        └──────────┬──────────────────────────┬────────────────────────────────────┘
                   │                          │
                   │ HTTP :8400               │ HTTP :8080
                   │ POST /v1/audio/speech    │ POST /measure
                   │ POST /v1/audio/voices    │ (WAV + ref_text)
                   │ GET  /health             │
                   ▼                          ▼
        ┌───────────────────────┐   ┌──────────────────────────────┐
        │   vLLM-omni (Docker)  │   │   Metrics API                │
        │        :8400          │   │   10.50.2.132:8080           │
        │                       │   │                              │
        │  • Qwen3-TTS-1.7B-Base│   │  • faster-whisper large-v3   │
        │  • Voice Registration │   │  • WAV → STT → hypothesis    │
        │    (ICL ref audio)    │   │  • jiwer → CER/WER           │
        │  • PCM 스트리밍 출력   │   │  • deletions/insertions/     │
        │  • max-num-seqs: 32   │   │    substitutions             │
        │  • bfloat16           │   │  • ref / hyp 텍스트 반환      │
        │  • CUDA Graph 최적화   │   └──────────────────────────────┘
        └───────────────────────┘
                   │
                   │ (API 서버 시작 시)
                   │ health 폴링 → 보이스 등록 → GPU 워밍업
                   │
        ┌──────────────────────────────┐
        │   Redis (Docker)  :9108      │
        │                              │
        │  문장별 캐시:                 │
        │  qwen3tts:cache:{hash}:pcm   │
        │  (raw PCM, TTL 1h)           │
        │                              │
        │  전체 복합 캐시:               │
        │  qwen3tts:cache:{hash}:pcm   │
        │  (모든 문장 합산, TTL 1h)     │
        │                              │
        │  통계 카운터:                  │
        │  tts_stats:requests          │
        │  tts_stats:cache_hits        │
        └──────────────────────────────┘
        
        ```
        
    - 기술 명세서
        
        ```markdown
          1. 기술 스택 (Tech Stack)
        
          ┌──────────────────┬─────────────────────────────────────────┬──────────────────────────────────────┐
          │ 분류             │ 기술 및 라이브러리                      │ 비고                                 │
          ├──────────────────┼─────────────────────────────────────────┼──────────────────────────────────────┤
          │ Backend          │ Python 3.11+, FastAPI, Uvicorn          │ 고성능 비동기 API 프레임워크         │
          │ Frontend         │ React (TypeScript), Vite, CSS3          │ 실시간 스트리밍 UI 대응              │
          │ Inference Engine │ vLLM-omni (AR/Generation 2-Stage)       │ Qwen3-TTS 전용 추론 엔진             │
          │ Storage & Cache  │ Redis 7.0+, JSON (File-based DB)        │ 오디오 캐싱 및 메타데이터 저장       │
          │ AI Models        │ Qwen3-TTS-1.7B-Base, Whisper-large-v3   │ 핵심 TTS 모델 및 품질 측정(STT) 모델 │
          │ Protocol         │ HTTP/1.1 (JSON), WebSocket (Binary/PCM) │ 배치 및 실시간 스트리밍 지원         │
          │ DevOps/Tools     │ Docker, Nginx, Loguru, Pydantic         │ 컨테이너화 및 구조화된 로깅          │
          └──────────────────┴─────────────────────────────────────────┴──────────────────────────────────────┘
        
          ---
        
          2. 파이프라인 및 아키텍처 (Pipeline Architecture)
        
          프로젝트는 "텍스트 전처리 → 엔진 전달 → 오디오 생성 → 사후 처리"의 4단계 파이프라인으로 구성됩니다.
        
          [Step 1] Text Processing (전처리)
           * ITN (Inverse Text Normalization): 숫자, 단위, 기호를 음성 언어(예: 123 → 백이십삼)로 변환.
           * Sentence Splitting: 마침표, 줄바꿈 기준으로 문장을 자동 분할.
               * 이유: 긴 문장을 한 번에 생성할 때 발생하는 Latency(대기 시간)를 줄이고, 문장 단위 캐시 히트율을 높이기 위함.
               * 제한: 한국어 기준 150자 이내로 분할하여 추론 안정성 확보.
        
          [Step 2] Synthesis Engine (vLLM-omni)
           * 2-Stage Inference:
               1. Stage 0 (Talker): 텍스트와 참조 음성(Reference Audio)을 분석하여 오디오 잠재 변수(Latent) 생성.
               2. Stage 1 (Code2Wav): 생성된 잠재 변수를 실제 들을 수 있는 오디오 파형(WAV/PCM)으로 변환.
           * ICL (In-Context Learning): 5초 분량의 참조 음성과 텍스트를 프롬프트로 주어 특정 화자의 목소리를 복제(Zero-shot Voice Cloning).
        
          [Step 3] Post-Processing (사후 처리)
           * Caching: 생성된 오디오 바이너리를 Redis에 저장. 키(Key)는 (엔진+텍스트+보이스ID+설정값)의 SHA-256 해시.
           * Metrics (CER/WER): 생성된 오디오를 Whisper 모델(Metrics Server)로 보내 텍스트로 복원한 뒤, 원본과 비교하여 음절 오류율(CER) 측정.
        
          ---
        
          3. 동시성 및 큐 관리 (Concurrency & Queue)
        
           * API Semaphore 제어:
               * asyncio.Semaphore(4)를 사용하여 vLLM 엔진으로 동시에 들어가는 요청을 최대 4개로 제한.
               * 목적: GPU 메모리(OOM) 방지 및 단일 요청에 대한 추론 속도 보장.
           * vLLM 내부 큐:
               * 엔진 레벨에서 max-num-seqs: 32 설정을 통해 배치(Batching) 처리.
           * 우선순위(Priority) 정책:
               1. Admin 요청: 유지보수 모드에서도 우회 권한 부여.
               2. WebSocket 스트리밍: 일반 HTTP 요청보다 TTFA(첫 소리가 들리는 시간)를 우선적으로 처리하도록 비동기 제네레이터 설계.
        
          ---
        
          4. 에러 분류 및 처리 전략 (Error Handling)
        
          ┌─────────────────────┬────────────────────┬──────────────────────────────────────────────────────────────────────────┐
          │ 에러 유형           │ 감지 방식          │ 대응 전략                                                                │
          ├─────────────────────┼────────────────────┼──────────────────────────────────────────────────────────────────────────┤
          │ Engine Timeout      │ httpx.ReadTimeout  │ 60초 대기 후 실패 응답 (504 Gateway Timeout)                             │
          │ Connection Fail     │ httpx.ConnectError │ 0.5초 대기 후 1회 자동 재시도(Retry)                                     │
          │ Model Hallucination │ CER 측정 루프      │ CER이 일정 임계치(예: 0.3)를 넘으면 로그에 경고 기록 및 감사 로그에 남김 │
          │ Concurrency Limit   │ Semaphore 대기     │ 큐에서 대기시키거나, 설정된 타임아웃 초과 시 503 에러 반환               │
          │ File DB Conflict    │ asyncio.Lock       │ 단일 프로세스 내에서 파일 쓰기 시 락(Lock)을 걸어 데이터 손상 방지       │
          └─────────────────────┴────────────────────┴──────────────────────────────────────────────────────────────────────────┘
        
          ---
        
          5. 핵심 성능 기술 (Optimization)
        
           * CUDA Graph Warmup: 서버 시작 시 '안녕하세요' 등의 텍스트로 미리 추론을 수행하여 GPU 연산 그래프를 컴파일. (첫 요청의 Latency 80% 감소)
           * Redis Binary Caching: WAV 파일 전체를 Redis에 바이너리로 저장하여 동일 문장 재요청 시 0ms 대기 시간 실현.
           * Streaming Delivery: WebSocket을 통해 PCM 청크가 생성되는 즉시 클라이언트로 전송. 사용자는 전체 문장이 완성되기 전부터 음성을 들을 수 있음.
        
          ---
        ```
        
    - 기술명세서(2) 상세
        
        ```markdown
          1. 상세 통신 규약 (Communication Protocols)
        
          시스템은 총 4개의 계층에서 서로 다른 프로토콜과 규격으로 통신합니다.
        
          ┌───────────────────┬───────────┬──────────────────────────┬──────────────────────────────────────────────┐
          │ 구간              │ 프로토콜  │ 데이터 포맷              │ 비고                                         │
          ├───────────────────┼───────────┼──────────────────────────┼──────────────────────────────────────────────┤
          │ Client ↔ API      │ HTTP/1.1  │ JSON                     │ 인증(JWT), 설정 조회, 일반 배치 합성         │
          │ Client ↔ API      │ WebSocket │ JSON (In) / Binary (Out) │ 실시간 스트리밍 (PCM 16-bit Mono)            │
          │ API ↔ vLLM Engine │ HTTP/1.1  │ JSON / Streaming Body    │ vLLM-omni REST API (OpenAI Speech Spec 호환) │
          │ API ↔ Redis       │ RESP      │ Binary (WAV/PCM)         │ Redis 전용 바이너리 직렬화 프로토콜          │
          │ API ↔ Metrics     │ HTTP/1.1  │ Multipart/form-data      │ 오디오 파일 전송 및 CER 결과 수신            │
          └───────────────────┴───────────┴──────────────────────────┴──────────────────────────────────────────────┘
        
          ---
        
          2. 스트리밍 및 데이터 처리 규격 (Chunk & Audio Specs)
        
          실시간성을 보장하기 위해 정의된 세부 파라미터입니다.
        
           * 오디오 규격 (Audio Format):
               * Sample Rate: 24,000 Hz (Qwen3-TTS 12Hz 모델 기본값)
               * Bit Depth: 16-bit (Signed Integer, Little-endian)
               * Channels: Mono (1채널)
           * 스트리밍 청크 크기 (Chunk Size):
               * WebSocket 전송 단위: 32,768 Bytes (약 0.68초 분량의 오디오 데이터)
               * 전략: 너무 작으면 네트워크 오버헤드가 크고, 너무 크면 대기 시간이 길어지므로 32KB로 최적화.
           * 초기 무음 패딩 (Silence Padding):
               * 스트림 시작 시 9,600 Bytes (200ms)의 무음(b"\x00")을 강제 삽입.
               * 목적: 브라우저 오디오 컨텍스트가 초기화되는 동안 실제 음성이 잘리는 현상(Clicking/Clipping) 방지.
        
          ---
        
          3. 배치 및 병렬 처리 정책 (Batching & Concurrency)
        
          GPU 자원 효율과 응답 속도 사이의 균형을 맞춘 설정입니다.
        
           * 엔진 레벨 배치 (vLLM Batching):
               * max-num-seqs (최대 동시 시퀀스): 32
               * max-model-len: 2,048 Tokens
               * 방식: Continuous Batching (여러 사용자의 요청을 실시간으로 묶어서 GPU 연산 효율 극대화)
           * API 서버 병렬 제어 (Semaphore):
               * TTS_VLLM_MAX_CONCURRENT: 4
               * 전략: API 서버 수준에서 동시에 엔진에 요청을 보내는 프로세스를 4개로 제한하여 엔진 과부하 및 타임아웃 방지.
           * 연결 풀링 (Connection Pooling):
               * max_connections: 20
               * keepalive_expiry: 10 seconds
               * 방식: 매 요청마다 TCP 연결을 맺지 않고 유지하여 핸드셰이크 시간(Latency) 절약.
        
          ---
        
          4. 텍스트 파이프라인 세부 수치 (Text Processing)
        
           * 문장 분할 임계값 (Splitting Threshold):
               * 한국어 (KO): 150자 (공백 포함)
               * 영어 (EN): 300자
               * 중국어/일본어 (ZH/JA): 120자
           * ITN (정규화) 엔진:
               * 자체 제작된 itn.py의 정규식(Regex) 기반 엔진 사용.
               * 전화번호, 날짜, 화폐 단위, 소수점 등을 한글 읽기 방식으로 변환.
        
          ---
        
          5. 에러 분류 및 리턴 코드 (Error Classification)
        
          프로덕션 환경을 고려하여 에러를 세밀하게 정의했습니다.
        
          ┌─────────────────────────┬─────────────┬─────────────────────────────────────┬──────────────────────────────────┐
          │ HTTP Code               │ 에러 명칭   │ 발생 원인                           │ 대응 로직                        │
          ├─────────────────────────┼─────────────┼─────────────────────────────────────┼──────────────────────────────────┤
          │ 200 OK                  │ Success     │ 합성 성공                           │ 바이너리 데이터 또는 스트림 반환 │
          │ 401 Unauthorized        │ Auth Fail   │ JWT 토큰 누락 또는 만료             │ 클라이언트 재로그인 유도         │
          │ 403 Forbidden           │ Maint Mode  │ 유지보수 모드 활성화 중 유저 요청   │ UI에 점검 모달 표시              │
          │ 502 Bad Gateway         │ Engine Fail │ vLLM 서버 응답 에러 (HTTP 5xx)      │ 로그 기록 및 1회 재시도(Retry)   │
          │ 503 Service Unavailable │ Overloaded  │ 세마포어 대기열 초과 또는 연결 실패 │ 잠시 후 다시 시도 안내           │
          │ 504 Gateway Timeout     │ Timeout     │ 60초 내 오디오 미생성               │ 요청 취소 및 리소스 회수         │
          └─────────────────────────┴─────────────┴─────────────────────────────────────┴──────────────────────────────────┘
        
          ---
        
          6. 보고를 위한 핵심 성능 지표 (KPIs for Report)
        
          이 시스템의 가치를 증명할 수 있는 주요 수치들입니다.
        
           1. TTFA (Time To First Audio): 약 100~300ms (WebSocket 스트리밍 시, 캐시 히트 시 10ms 미만)                                                               
           2. RTF (Real-Time Factor): 0.1 이하 (10초 분량 음성 생성에 약 1초 소요)                                                                                   
           3. Cache Efficiency: 동일 문장 반복 요청 시 GPU 사용량 0%, 대기 시간 0ms.                                                                                 
           4. Accuracy (CER): Whisper-large-v3 기준 평균 3~5% 미만 (한국어 기준).                                                                                    
        ```
        
    - 운영 페이지 제작
        
        
        ![image.png](image.png)
        
        ![image.png](image%201.png)
        
        ![image.png](image%202.png)
        
        ![image.png](image%203.png)
        
    - 스트리밍 api 리팩토링
        
        [Demo 페이지 ](Demo%20%ED%8E%98%EC%9D%B4%EC%A7%80%203420330b274180bf8c0de5c30d075bd0.md)
        
        [보고- 메일(개인 소장용)](%EB%B3%B4%EA%B3%A0-%20%EB%A9%94%EC%9D%BC(%EA%B0%9C%EC%9D%B8%20%EC%86%8C%EC%9E%A5%EC%9A%A9)%2032c0330b274180fd8321ecfc234f9c6b.md)
        
        **버전 (deprecated)**
        
        - 성과:  응답 시간이 약 **85.4%** 단축 ( TTFB 약 6.85배 빨라짐)
        
        ```
        POST /api/v2/tts-engine/synthesize/sse
        
        (/api/v1/tts-engine/synthesize/sse 도 동일 방식 — 둘 다 deprecated)
        ```
        
        - 전체 합성이 끝날 때까지 첫 청크가 오지 않습니다
        - TTFB: **2292ms**
        - 총 청크: **342개**
        - 포맷: **WAV**
        - 샘플레이트: **24000 Hz**
        - 길이: **57.86s**
        - 완료 (342개 청크)
        - [오후 1:14:24] POST http://10.50.1.43:9102/api/v2/tts-engine/synthesize/sse
        [오후 1:14:26] 첫 청크 수신 → TTFB: 2292ms
        [오후 1:14:27] 완료: 총 342개 청크
        
        **신버전**
        
        ```
        POST /api/v2/tts-engine/synthesize/sse-realtime
        ```
        
        - 첫 문장이 완성되는 즉시 재생 시작합니다
        - TTFB: **334ms**
        - 포맷: **WAV**
        - 샘플레이트: **24000 Hz**
        - 길이: **-**
        - 중지됨
        - [오후 1:14:33] POST http://10.50.1.43:9102/api/v2/tts-engine/synthesize/sse-realtime
        [오후 1:14:34] 첫 청크 수신 → TTFB: 334ms
    
    [Qwen3-TTS Enterprise Gateway: 프로젝트 딥다이브 리포트](Qwen3-TTS%20Enterprise%20Gateway%20%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%20%EB%94%A5%EB%8B%A4%EC%9D%B4%EB%B8%8C%20%EB%A6%AC%ED%8F%AC%ED%8A%B8%203440330b27418016a3bdebfea63c9d49.md)
    

음성팀 (2025.04~09)

- 화자 분할(2025.04~07)
    - PPT - [pyannote.pdf](https://app.notion.com/p/pyannote-pdf-1fb0330b2741802aaa52dfc399ba3cda?pvs=21)
    - Pyannote fine-tuning 진행
- STT(2025.07~09)
    - AIG 상담 전화 STT
        - AIG STT 구조 개선
            - log 및 docker 구조 개선
            - 재사용성 위한 코드 모듈화 및 구성 리팩토링
            - 성능 및 속도 향상을 위한 전처리/병렬 코드 추가
            - 컨테이너 재시작 모니터링 등 안전성 관련 설정 추가
        - 기존 코드 리팩토링, 엔진 수정 진행
        - [AIG-STT](https://app.notion.com/p/AIG-STT-2770330b274180bdbe91f6cf8c19d361?pvs=21)
    - 사내 회의록 기능 내  STT 탑재
        - 구조 개선
            - log 및 docker 구조 개선
            - 디렉토리 구성 개선
            - whisper X 최적의 파라미터 찾기
            - whisper 계열 모델별 성능 조사, 정리
            - QA 업무 (오류 찾기, 결과 확인하기)
            - 엔진-브로커(api-http) 통신 관련 에러 수정
        - 기능
            1. 파일 기반 STT 처리
            2. [스트리밍 기반 STT 처리](https://app.notion.com/p/STT-2480330b2741807c862ec04125ba1010?pvs=21)
        - 역할
            1. STT, 화자분리 연동 
            2. STT 고도화 작업

# 인피닉 (2022.8- 2025.4)

개발팀

---

- 쿠버네티스(24년 8월 ~ 25년 4월)
    
    사내 쿠버네티스 망 구축
    
    [PPSX](PPSX%202050330b2741803a9aa1d85075cf403b.md)
    
- 3차원 시멘틱세그멘테이션 (25년 1월 ~ 2월)
    - annotation tool 유지보수
    
- 텍스트 요약 (24년 1월 ~ 7월)
    - BART 기반 대화 요약
    - T5 기반 문서 요약
    
    [텍스트 요약](%ED%85%8D%EC%8A%A4%ED%8A%B8%20%EC%9A%94%EC%95%BD%202050330b27418044a6baee9f74c15183.md)
    
- 탱크 생성모델(23년 9월 ~12월)
    - 성과
        - 논문
            1. [**국방 데이터 확보를 위한 생성 모델 Lattent Diffusion**](https://www.kidet.or.kr/index.php?hCode=BOARD&page=view&idx=1219&bo_idx=1&hCode=BOARD&bo_idx=1&sfl=&stx=)
                
                [국방 데이터 확보를 위한 생성 모델 Lattent Diffusion](%EA%B5%AD%EB%B0%A9%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%ED%99%95%EB%B3%B4%EB%A5%BC%20%EC%9C%84%ED%95%9C%20%EC%83%9D%EC%84%B1%20%EB%AA%A8%EB%8D%B8%20Lattent%20Diffusion%202050330b2741809b9302fbf700a95ce8.md)
                
                - 우수 논문 수상
            2. [GAN을 활용한 데이터 생성 연구 동향](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11622970&nodeId=NODE11622970&mobileYN=N&medaTypeCode=185004&isPDFSizeAllowed=true&locale=ko&foreignIpYn=N&articleTitle=GAN%EC%9D%84+%ED%99%9C%EC%9A%A9%ED%95%9C+%EB%8D%B0%EC%9D%B4%ED%84%B0+%EC%83%9D%EC%84%B1+%EC%97%B0%EA%B5%AC+%EB%8F%99%ED%96%A5&articleTitleEn=Data+generation+research+trend+using+GAN&voisId=VOIS00738913&voisName=%ED%95%9C%EA%B5%AD%ED%95%AD%EA%B3%B5%EC%9A%B0%EC%A3%BC%ED%95%99%ED%9A%8C+2023+%EC%A0%9C1%ED%9A%8C+%EC%9A%B0%EC%A3%BC%ED%95%99%EC%88%A0%EB%8C%80%ED%9A%8C&voisCnt=346&language=ko_KR&hasTopBanner=true)
    
- 탱크 탐지 모델 (23년 1월 ~8월)
    
    
- 2D, 3D 센서 퓨전 세그멘테이션 모델(22년 8월 ~ 22년 12월)
    - 성과
        - **논문**
            1. [자율 주행 도메인의 3차원 시맨틱 세그멘테이션을 위한 센서 퓨전 기반의 Trans-Unet](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11220500)
            2. [센서 퓨전 기반의 Trans-Unet을 활용한 2차원 시맨틱 세그멘테이션의 3차원적 해석](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11232100)
        - **특허(1 발명자)**
            1. [센서 퓨전 기반의 시맨틱 세그멘테이션 방법 및 이를 실행하기 위하여 기록매체에 기록된 컴퓨터 프로그램](https://doi.org/10.8080/1020230030194)
                1. 등록번호: **10-2538225-0000**
                    
                    [KIPRIS 지식재산정보 검색 서비스.pdf](KIPRIS_%EC%A7%80%EC%8B%9D%EC%9E%AC%EC%82%B0%EC%A0%95%EB%B3%B4_%EA%B2%80%EC%83%89_%EC%84%9C%EB%B9%84%EC%8A%A4.pdf)
                    
            2. [시맨틱 세그멘테이션의 3차원 해석 방법 및 이를 실행하기 위하여 기록매체에 기록된 컴퓨터 프로그램](https://doi.org/10.8080/1020230030193)
                1. 등록번호 : **10-2538231-0000**
                    
                    [KIPRIS 지식재산정보 검색 서비스.pdf](KIPRIS_%EC%A7%80%EC%8B%9D%EC%9E%AC%EC%82%B0%EC%A0%95%EB%B3%B4_%EA%B2%80%EC%83%89_%EC%84%9C%EB%B9%84%EC%8A%A4%201.pdf)
                    

# 이든티앤에스 (2022.4- 2022.8)

AI 팀 

---

- table detection(OCR)
    - pdf table detection 모델
    - GUI tool 제작
    - 성과
        - 사내 프로젝트 축하 성과금

# 큐헷지 (2020.9- 2021.9)

---

- 데이터 수집, 전처리

# $\color{3399ff}{Projects}$

---

### 프로젝트 (1)

[제목 없음](%EC%A0%9C%EB%AA%A9%20%EC%97%86%EC%9D%8C%202050330b2741802ab6e7c224457f7e6e.csv)

# $\color{3399ff}{Skills}$

### Collaboration

---

<aside>
<img src="notion.png" alt="notion.png" width="40px" /> Notion                           🟪🟪🟪⬜️⬜️

</aside>

<aside>
<img src="git.png" alt="git.png" width="40px" /> Git                          🟪🟪🟪⬜️⬜️

</aside>

# $\color{3399ff}{Social}$  $\color{3399ff}{Media}$

---

<aside>
<img src="git%201.png" alt="git%201.png" width="40px" /> Git

[https://github.com/ckc5800?tab=repositories](https://github.com/ckc5800?tab=repositories)

</aside>

<aside>
<img src="Jz0MFNR7_7LHx1Yda0Hy6929g3BD5fWmaARdUMMFPkFIAVC_ewY7BEcoIGhepmTKRBKmSxRSUBjI7pklIZLaAA.svg" alt="Jz0MFNR7_7LHx1Yda0Hy6929g3BD5fWmaARdUMMFPkFIAVC_ewY7BEcoIGhepmTKRBKmSxRSUBjI7pklIZLaAA.svg" width="40px" /> tistory                           
https://taepseon.tistory.com/

</aside>

<aside>
<img src="%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.png" alt="%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.png" width="40px" /> linkedIn 
[www.linkedin.com](http://www.linkedin.com/in/yoonseon-lee-91981416b)

</aside>

<aside>
<img src="notion%201.png" alt="notion%201.png" width="40px" /> notion
[notion](%EC%9D%B4%EC%9C%A4%EC%84%A0%20%ED%8F%AC%ED%8A%B8%ED%8F%B4%EB%A6%AC%EC%98%A4%202050330b274180dda2dcc1fbd465f112.md)

</aside>

# $\color{3399ff}{Others}$

---

이런 경험을 해봤습니다. 

[연구 수행 실적 (학위 논문 외)](%EC%97%B0%EA%B5%AC%20%EC%88%98%ED%96%89%20%EC%8B%A4%EC%A0%81%20(%ED%95%99%EC%9C%84%20%EB%85%BC%EB%AC%B8%20%EC%99%B8)%2020a0330b274180b1b2a5d1d63687fbab.md)

[사내 풋살 동호회 운영 (인피닉)](%EC%82%AC%EB%82%B4%20%ED%92%8B%EC%82%B4%20%EB%8F%99%ED%98%B8%ED%9A%8C%20%EC%9A%B4%EC%98%81%20(%EC%9D%B8%ED%94%BC%EB%8B%89)%2020a0330b274180eb85f1e6ddb537e3a7.md)

[수영장 강사](%EC%88%98%EC%98%81%EC%9E%A5%20%EA%B0%95%EC%82%AC%2020a0330b274180b987d7ee2b7473b029.md)

[포스텍 MOOC 교육 프로그램 수료](%ED%8F%AC%EC%8A%A4%ED%85%8D%20MOOC%20%EA%B5%90%EC%9C%A1%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8%20%EC%88%98%EB%A3%8C%2020a0330b274180e5b1dfc16e213ee4eb.md)

[[**한국과학기술정보연구원](https://namu.wiki/w/%ED%95%9C%EA%B5%AD%EA%B3%BC%ED%95%99%EA%B8%B0%EC%88%A0%EC%A0%95%EB%B3%B4%EC%97%B0%EA%B5%AC%EC%9B%90)(**kisti) ](%ED%95%9C%EA%B5%AD%EA%B3%BC%ED%95%99%EA%B8%B0%EC%88%A0%EC%A0%95%EB%B3%B4%EC%97%B0%EA%B5%AC%EC%9B%90(kisti)%2020a0330b2741805c823ae9d474ec169c.md)

[봉사 활동](%EB%B4%89%EC%82%AC%20%ED%99%9C%EB%8F%99%202130330b2741803da385f68a8f6989d5.md)

[특허](%ED%8A%B9%ED%97%88%2032d0330b27418053b103f8b137a2c5c3.md)

[203 특공여단](203%20%ED%8A%B9%EA%B3%B5%EC%97%AC%EB%8B%A8%203420330b274180878fe6dbbdbe6a2e6a.md)