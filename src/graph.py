"""LangGraph 기반 Corrective-RAG Agent.

실행 흐름:
    retrieve → grade ─(충분)→ generate
                     └(부족)→ rewrite → retrieve  (최대 MAX_REWRITES회)

State로 질문/문서/재작성 횟수를 관리하며, 검색 품질이 낮으면
질문을 재작성해 재검색하는 self-corrective 루프를 구성한다.
"""
import re
from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

import config


class AgentState(TypedDict):
    question: str          # 사용자 원본 질문
    query: str             # 실제 검색에 사용하는 질의 (재작성될 수 있음)
    documents: list[Document]
    rewrites: int          # 질문 재작성 횟수
    grade: str             # 검색 품질 판정 (sufficient / insufficient)
    answer: str
    sources: list[str]


_vectorstore = None
_bm25 = None


def bm25_tokenize(text: str) -> list[str]:
    """공백 분리 + 한글 어절엔 문자 bigram 추가.

    BM25의 공백 토큰화는 '회사들'과 '회사'를 다른 토큰으로 취급해
    조사가 붙은 한국어 질의에 약하다. 어절을 유지한 채 문자 bigram을
    함께 넣으면 '회사들' ↔ '회사'가 bigram(회사)으로 겹친다.
    """
    grams = []
    for t in text.lower().split():
        grams.append(t)
        if len(t) >= 2 and any("가" <= ch <= "힣" for ch in t):
            grams += [t[i:i + 2] for i in range(len(t) - 1)]
    return grams


def _load_indexes():
    global _vectorstore, _bm25
    if _vectorstore is None:
        import json

        from langchain_community.retrievers import BM25Retriever

        _vectorstore = FAISS.load_local(
            config.DB_DIR,
            OllamaEmbeddings(model=config.EMBED_MODEL),
            allow_dangerous_deserialization=True,  # 로컬에서 직접 생성한 인덱스만 로드
        )
        chunks = []
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                chunks.append(Document(page_content=d["page_content"],
                                       metadata=d["metadata"]))
        _bm25 = BM25Retriever.from_documents(
            chunks, preprocess_func=bm25_tokenize)
        _bm25.k = config.TOP_K
    return _vectorstore, _bm25


def hybrid_search(query: str) -> list[Document]:
    """FAISS(의미) + BM25(키워드) 결과를 RRF(Reciprocal Rank Fusion)로 융합.

    'Jenkins', 'Pyannote' 같은 고유명사/키워드 질문은 벡터 검색이 놓치기 쉬워
    BM25를 결합해 검색 재현율을 보완한다. RRF score = Σ 1 / (k + rank).
    """
    import hashlib

    vectorstore, bm25 = _load_indexes()
    vec_docs = vectorstore.similarity_search(query, k=config.TOP_K)
    kw_docs = bm25.invoke(query)

    K = 60  # RRF 완충 상수 (표준값)
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in (vec_docs, kw_docs):
        for rank, doc in enumerate(docs):
            # 전체 내용 해시로 중복 판별 (접두어가 같은 서로 다른 청크의 충돌 방지)
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_key[k] for k in ranked[: config.TOP_K]]


_llm_cache: dict[float, ChatOllama] = {}


# 판정(grade)·생성(generate)은 재현성이 중요하므로 temperature 0으로 고정.
# 질문 재작성(rewrite)만 다양성이 필요해 0.3을 사용한다.
def get_llm(temperature: float = 0.0) -> ChatOllama:
    if temperature not in _llm_cache:
        _llm_cache[temperature] = ChatOllama(
            model=config.LLM_MODEL, temperature=temperature
        )
    return _llm_cache[temperature]


# ── Nodes ──────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    return {"documents": hybrid_search(state["query"])}


GRADE_PROMPT = ChatPromptTemplate.from_template(
    "당신은 검색 품질 평가자입니다.\n"
    "질문: {question}\n\n"
    "검색된 문서:\n{context}\n\n"
    "문서에 질문과 관련된 정보가 일부라도 포함되어 있으면 'yes', "
    "전혀 관련 없는 내용뿐이면 'no'만 출력하세요."
)


def grade(state: AgentState) -> dict:
    context = "\n---\n".join(d.page_content[:500] for d in state["documents"])
    chain = GRADE_PROMPT | get_llm()
    verdict = chain.invoke(
        {"question": state["question"], "context": context}
    ).content.strip().lower()
    # 한국어 응답(예/아니오) 포함 견고한 파싱 — 명시적 부정일 때만 재검색 (fail-open)
    negative = ("no" in verdict.split() or verdict.startswith("no")
                or "아니" in verdict) and "yes" not in verdict
    return {"grade": "insufficient" if negative else "sufficient"}


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "다음 질문으로 문서 검색을 했지만 관련 문서를 찾지 못했습니다.\n"
    "검색이 잘 되도록 핵심 키워드 중심으로 질문을 한 문장으로 재작성하세요.\n"
    "재작성된 질문만 출력하세요.\n\n질문: {question}"
)


def rewrite(state: AgentState) -> dict:
    chain = REWRITE_PROMPT | get_llm(temperature=0.3)
    new_query = chain.invoke({"question": state["question"]}).content.strip()
    return {"query": new_query, "rewrites": state["rewrites"] + 1}


GENERATE_PROMPT = ChatPromptTemplate.from_template(
    "당신은 AI 엔지니어 이윤선의 포트폴리오를 안내하는 어시스턴트입니다.\n"
    "아래 문서를 주의 깊게 읽고, 관련 내용이 있으면 그것을 근거로 한국어로 답하세요.\n"
    "참고: 항목 옆 (YYYY.MM ~ YYYY.MM) 또는 (YYYY.MM~) 표기는 그 항목의 "
    "수행 기간이며 왼쪽 날짜가 시작 시점입니다.\n"
    "관련 정보가 정말로 전혀 없을 때만 '문서에서 찾을 수 없습니다'라고 답하고, "
    "문서에 없는 내용을 지어내지 마세요.\n\n"
    "문서:\n{context}\n\n질문: {question}\n\n답변:"
)


def generate(state: AgentState) -> dict:
    # 소형 모델은 긴 컨텍스트에서 근거를 놓치기 쉬우므로 생성에는 상위 3개
    # 청크만 사용하고, 끝부분 주의집중이 강한 특성에 맞춰 랭크 역순으로 배치해
    # 최상위 청크가 질문 바로 앞에 오게 한다
    docs = list(reversed(state["documents"][:3]))
    context = "\n---\n".join(d.page_content for d in docs)
    chain = GENERATE_PROMPT | get_llm()
    answer = chain.invoke(
        {"question": state["question"], "context": context}
    ).content.strip()
    sources = sorted({d.metadata.get("source", "?") for d in state["documents"]})
    return {"answer": answer, "sources": sources}


# ── Graph ──────────────────────────────────────────────

def decide_next(state: AgentState) -> str:
    """검색 품질과 재작성 횟수에 따라 다음 노드 결정."""
    if state.get("grade") == "sufficient" or state["rewrites"] >= config.MAX_REWRITES:
        return "generate"
    return "rewrite"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", decide_next, ["rewrite", "generate"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def ask(question: str) -> dict:
    graph = build_graph()
    result = graph.invoke(
        {"question": question, "query": question, "rewrites": 0}
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "rewrites": result["rewrites"],
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "TTS 프로젝트의 TTFB 개선 수치는?"
    out = ask(q)
    print("답변:", out["answer"])
    print("출처:", out["sources"])
