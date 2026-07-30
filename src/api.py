"""FastAPI 서빙 레이어: POST /ask 로 RAG Agent 호출."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from graph import build_graph, warmup

graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    graph = build_graph()  # 그래프는 시작 시 1회만 컴파일
    # 인덱스도 기동 시 적재한다. 지연 초기화에 맡기면 첫 요청이 인덱스를
    # 만드는 동안 들어온 두 번째 요청과 경합하고(graph._load_indexes 주석),
    # 첫 응답만 수 초 느려진다.
    warmup()
    yield


app = FastAPI(title="Portfolio RAG Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    rewrites: int


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = graph.invoke(
        {"question": req.question, "query": req.question, "rewrites": 0}
    )
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        rewrites=result["rewrites"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
