"""FastAPI 서빙 레이어: POST /ask 로 RAG Agent 호출."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = graph.invoke(
            {"question": req.question, "query": req.question, "rewrites": 0}
        )
    except Exception as e:                                   # noqa: BLE001
        # 대부분 Ollama 다운·모델 언로드다. 스택트레이스 대신 원인을 준다.
        raise HTTPException(
            status_code=503,
            detail=f"추론 백엔드 오류: {type(e).__name__}: {e}") from e
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        rewrites=result["rewrites"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
