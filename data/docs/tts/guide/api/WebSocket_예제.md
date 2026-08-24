# WebSocket Streaming Example

가장 낮은 TTFB로 음성을 받기 위한 WebSocket 사용 가이드. SDK 없이 raw WebSocket으로 구현.

## 프로토콜 요약

**Endpoint**: `wss://your-domain.com/v1/tts/synthesize/ws?token=<API_KEY_or_JWT>`

또는 `Sec-WebSocket-Protocol: <token>` subprotocol 헤더로도 전달 가능.

### 메시지 종류

**Client → Server (JSON text frame)**:

| `command` | 페이로드 | 설명 |
|---|---|---|
| `synthesize` | `{...SynthesizeBody}` | 합성 시작 |
| `stop` | — | 진행 중 합성 취소 |
| `ping` | — | heartbeat |
| `client_metrics` | `{client_ttfb_ms: <int>}` | 클라이언트 측 TTFB 통보 (audit 합산용) |

**Server → Client**:

- **JSON text frame** — 필드명이 메시지 종류마다 다릅니다 (코드 호환성을 위해):

| 페이로드 | 분기 키 | 값 |
|---|---|---|
| 합성 시작 | `status` | `started` (+ `sample_rate: int`) |
| 문장 시작 | `type` | `sentence_start` (+ `idx`, `text`) |
| 장시간 합성 keep-alive | `status` | `heartbeat` |
| 청크 모두 송신 완료 | `status` | `audio_complete` |
| CER/WER 메트릭 | `status` | `metrics` (+ `metrics: {...}`) |
| 세션 정상 종료 | `status` | `done` (+ `trace_id`, `total_ms`, `engine_fallback`, ...) |
| `stop` 응답 | `status` | `interrupted` |
| `ping` 응답 | `status` | `pong` |
| 문장 단위 실패 (세션 유지) | `warning` | `"sentence[N] synthesis failed — skipped"` |
| 합성 실패 (세션 종료 가능성) | `error` | `"..."` (+ 선택적 `code`, `trace_id`) |

권장 분기 코드: `key = msg.get('status') or msg.get('type') or ('error' if 'error' in msg else 'warning' if 'warning' in msg else 'unknown')`.

- **Binary frame**: raw PCM (16-bit little-endian, mono). 첫 청크에 44바이트 WAV 헤더 포함 — 단순 concat으로 WAV 파일 생성 가능. Sample rate는 `started` 메시지의 `sample_rate` 필드 참조.

## Python 예제 (websockets 라이브러리)

```python
import asyncio
import json
import struct
import websockets

URL = "wss://tts.your-domain.com/v1/tts/synthesize/ws"
TOKEN = "sk-xxxxx"

async def synthesize_stream(text: str, voice_id: str, output_path: str):
    pcm_chunks = []
    sample_rate = 24000  # started 메시지에서 갱신
    start = asyncio.get_event_loop().time()
    first_chunk_at = None

    async with websockets.connect(f"{URL}?token={TOKEN}") as ws:
        await ws.send(json.dumps({
            "command": "synthesize",
            "text": text,
            "voice_id": voice_id,
            "language": "ko",
            "speed": 1.0,
        }))

        async for msg in ws:
            if isinstance(msg, bytes):
                if first_chunk_at is None:
                    first_chunk_at = asyncio.get_event_loop().time()
                    ttfb_ms = int((first_chunk_at - start) * 1000)
                    print(f"TTFB: {ttfb_ms}ms")
                    # 종단 TTFB를 서버에 통보 → audit에 합산됨
                    await ws.send(json.dumps({"command": "client_metrics", "client_ttfb_ms": ttfb_ms}))
                pcm_chunks.append(msg)
            else:
                data = json.loads(msg)
                # 분기 키는 메시지 종류마다 다름 (status / type / error / warning)
                if "error" in data:
                    raise RuntimeError(f"server error: {data['error']} (code={data.get('code')})")
                if "warning" in data:
                    print(f"  warning: {data['warning']}")
                    continue
                kind = data.get("status") or data.get("type")
                if kind == "started":
                    sample_rate = data.get("sample_rate", sample_rate)
                elif kind == "sentence_start":
                    print(f"  sentence[{data['idx']}]: {data['text']}")
                elif kind == "audio_complete":
                    print("  audio complete")
                elif kind == "done":
                    print(f"done: total {data.get('total_ms')}ms trace={data.get('trace_id')}")
                    break
                elif kind == "interrupted":
                    print("  stop acknowledged")
                    break
                # heartbeat / pong / metrics 는 무시 또는 별도 처리

    # PCM → WAV 변환 후 저장
    pcm = b"".join(pcm_chunks)
    write_wav(output_path, pcm, sample_rate)

def write_wav(path: str, pcm: bytes, sample_rate: int):
    """16-bit mono PCM → WAV file."""
    with open(path, "wb") as f:
        data_len = len(pcm)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_len))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_len))
        f.write(pcm)

asyncio.run(synthesize_stream("안녕하세요. 첫 문장입니다. 두 번째 문장입니다.",
                              "qwen_kor_female_01_5s", "out.wav"))
```

## JavaScript 예제 (브라우저 + Web Audio)

청크별 즉시 재생 패턴. AudioContext로 gapless playback.

```js
async function synthesizeWS(text, voiceId, token) {
  const ws = new WebSocket(`wss://tts.your-domain.com/v1/tts/synthesize/ws?token=${token}`);
  ws.binaryType = "arraybuffer";
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  let sampleRate = 24000;
  let nextStartTime = 0;
  const start = performance.now();
  let firstChunkSent = false;

  ws.onopen = () => {
    ws.send(JSON.stringify({
      command: "synthesize",
      text,
      voice_id: voiceId,
      language: "ko",
      speed: 1.0,
    }));
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.error) { console.error("server error:", msg.error, msg.code); ws.close(); return; }
      if (msg.warning) { console.warn("warning:", msg.warning); return; }
      const kind = msg.status || msg.type;
      if (kind === "started") sampleRate = msg.sample_rate || sampleRate;
      else if (kind === "done") ws.close();
      // heartbeat / sentence_start / audio_complete / metrics / interrupted / pong 처리 추가 가능
    } else {
      const buf = event.data;  // ArrayBuffer
      if (!firstChunkSent) {
        const ttfb = Math.round(performance.now() - start);
        console.log(`TTFB: ${ttfb}ms`);
        // 종단 TTFB 송신
        ws.send(JSON.stringify({ command: "client_metrics", client_ttfb_ms: ttfb }));
        firstChunkSent = true;
      }
      schedulePcmChunk(audioCtx, buf, sampleRate, () => nextStartTime, (t) => { nextStartTime = t; });
    }
  };

  ws.onerror = (err) => console.error("ws error:", err);
  ws.onclose = () => audioCtx.close();
}

function schedulePcmChunk(ctx, arrayBuffer, sampleRate, getNext, setNext) {
  // Int16 PCM → Float32
  const view = new DataView(arrayBuffer);
  const sampleCount = arrayBuffer.byteLength / 2;
  const audioBuffer = ctx.createBuffer(1, sampleCount, sampleRate);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < sampleCount; i++) {
    channel[i] = view.getInt16(i * 2, true) / 32768.0;
  }
  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  let startAt = getNext();
  if (startAt < ctx.currentTime + 0.05) startAt = ctx.currentTime + 0.05;  // underrun 방지
  source.start(startAt);
  setNext(startAt + audioBuffer.duration);
}

synthesizeWS("안녕하세요. 첫 번째 문장. 두 번째 문장.",
             "qwen_kor_female_01_5s", "sk-xxxxx");
```

## 에러 처리

WebSocket close code 의미:

| Code | 의미 | 재시도 가능? |
|---|---|---|
| 1000 | 정상 종료 | — |
| 1006 | abnormal — 네트워크 끊김 | ✅ 가능 |
| 1008 | policy violation (인증 실패 등) | ❌ |
| 1011 | server error / overload | ✅ (백오프 후) |
| 1012 | 점검 모드 | ✅ (잠시 후) |
| 4001 | unauthorized | ❌ 토큰 갱신 후 |
| 4401 | 세션 무효화 (비밀번호 변경 등) | ❌ 재로그인 |

`error` JSON 메시지의 `code` 필드는 [오류_코드.md](오류_코드.md) 참조.

## 권장 패턴

1. **재연결 backoff**: 1006/1011 close 시 지수 백오프 (1s, 2s, 4s, 최대 3회).
2. **Heartbeat**: 30초마다 서버가 `{"status":"heartbeat"}` 송신. 60초 무수신 시 연결 끊김 판단.
3. **client_metrics 송신**: 첫 청크 수신 직후 종단 TTFB 송신 → 서버 audit에 합쳐져 운영자가 종단 SLA 추적 가능.
4. **stop 명령**: 사용자가 합성 도중 취소하려면 `{"command":"stop"}`. 서버가 1초 내 합성 task cancel + `{"status":"interrupted"}` 응답.
5. **재사용 세션**: 한 WebSocket으로 여러 합성 가능 (persistent). 매 합성마다 새로 연결할 필요 없음.
