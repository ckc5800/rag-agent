"""FastAPI 서빙 레이어: POST /ask 로 RAG Agent 호출."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from graph import build_graph

graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    graph = build_graph()  # 그래프는 시작 시 1회만 컴파일
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
