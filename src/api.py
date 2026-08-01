"""FastAPI 서빙 레이어: POST /ask 로 RAG Agent 호출."""
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import cache
import tracelog
from graph import build_graph, warmup
from preflight import check_all

graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    check_all()            # 인덱스·Ollama 확인 — 첫 요청 때 500으로 알지 않도록
    graph = build_graph()  # 그래프는 시작 시 1회만 컴파일
    # 인덱스도 기동 시 적재한다. 지연 초기화에 맡기면 첫 요청이 인덱스를
    # 만드는 동안 들어온 두 번째 요청과 경합하고(graph._load_indexes 주석),
    # 첫 응답만 수 초 느려진다.
    warmup()
    yield


app = FastAPI(title="Portfolio RAG Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    # 상한이 없으면 임의 길이 입력이 그대로 임베딩·LLM으로 간다
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    rewrites: int
    cached: bool = False
    # VERIFY_GROUNDING이 꺼져 있으면(기본) 둘 다 None — 아직 안 잰 것과
    # "확인했더니 근거 있음"을 구분하기 위해 False가 아니라 None을 쓴다.
    grounded: bool | None = None
    unsupported_claim: str | None = None


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()
    # 캐시 키에 현재 런타임 설정 지문이 들어 있어(cache.py), 설정을 바꾸면
    # 자동으로 새 항목이 된다 — 옛 설정의 답을 잘못 돌려줄 일이 없다.
    hit = cache.get(req.question)
    if hit is not None:
        tracelog.log(req.question, hit, time.time() - t0, cached=True)
        return AskResponse(**hit, cached=True)

    try:
        result = graph.invoke(
            {"question": req.question, "query": req.question, "rewrites": 0}
        )
    except Exception as e:                                   # noqa: BLE001
        # 대부분 Ollama 다운·모델 언로드다. 스택트레이스 대신 원인을 준다.
        raise HTTPException(
            status_code=503,
            detail=f"추론 백엔드 오류: {type(e).__name__}: {e}") from e

    payload = {
        "answer": result["answer"],
        "sources": result["sources"],
        "rewrites": result["rewrites"],
        "grounded": result.get("grounded"),
        "unsupported_claim": result.get("unsupported_claim"),
    }
    cache.put(req.question, payload)
    tracelog.log(req.question, payload, time.time() - t0, cached=False)
    return AskResponse(**payload, cached=False)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """토큰 스트리밍 버전. 60~150초(README 실측) 무응답 대기 대신 generate
    단계의 토큰을 SSE로 흘려보낸다.

    retrieve·grade·rewrite는 원래대로 블로킹으로 돈다 — 어차피 짧고
    (grade 구조화 출력은 한 번에 완성돼야 하는 JSON이라 스트리밍이 의미
    없다), 사용자가 오래 기다리는 구간은 generate 하나다. LangGraph의
    stream_mode=["messages","values"]가 그래프 노드별로 태깅된 토큰
    청크와 최종 상태 스냅샷을 함께 준다 — generate 노드 태그가 붙은
    청크만 골라 내보낸다.
    """
    def event_gen():
        t0 = time.time()
        # 캐시 히트면 스트리밍할 것도 없이 그대로 한 번에 낸다
        hit = cache.get(req.question)
        if hit is not None:
            tracelog.log(req.question, hit, time.time() - t0, cached=True,
                        endpoint="stream")
            yield f"data: {json.dumps({'token': hit['answer']}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True, **hit, 'cached': True}, ensure_ascii=False)}\n\n"
            return

        final_state = {}
        try:
            for mode, chunk in graph.stream(
                {"question": req.question, "query": req.question, "rewrites": 0},
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    msg_chunk, metadata = chunk
                    if (metadata.get("langgraph_node") == "generate"
                            and msg_chunk.content):
                        yield (f"data: {json.dumps({'token': msg_chunk.content}, ensure_ascii=False)}\n\n")
                elif mode == "values":
                    final_state = chunk
        except Exception as e:                               # noqa: BLE001
            yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
            return

        payload = {
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "rewrites": final_state.get("rewrites", 0),
            "grounded": final_state.get("grounded"),
            "unsupported_claim": final_state.get("unsupported_claim"),
        }
        cache.put(req.question, payload)
        tracelog.log(req.question, payload, time.time() - t0, cached=False,
                    endpoint="stream")
        yield f"data: {json.dumps({'done': True, **payload}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/health")
def health():
    # 예전엔 무조건 {"status": "ok"}였다 — Ollama가 죽었거나 인덱스가
    # 어긋나도 헬스체크는 통과했다. preflight의 같은 점검을 재사용해
    # (strict=False라 예외 대신 문제 목록을 받는다) 실제 상태를 본다.
    problems = check_all(strict=False)
    return {"status": "ok" if not problems else "degraded", "problems": problems}
