"""LangGraph 기반 Corrective-RAG Agent.

실행 흐름:
    retrieve → grade ─(충분)→ generate
                     └(부족)→ rewrite → retrieve  (최대 MAX_REWRITES회)

State로 질문/문서/재작성 횟수를 관리하며, 검색 품질이 낮으면
질문을 재작성해 재검색하는 self-corrective 루프를 구성한다.
"""
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


def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = FAISS.load_local(
            config.DB_DIR,
            OllamaEmbeddings(model=config.EMBED_MODEL),
            allow_dangerous_deserialization=True,  # 로컬에서 직접 생성한 인덱스만 로드
        )
    return _vectorstore


def get_llm(temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(model=config.LLM_MODEL, temperature=temperature)


# ── Nodes ──────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    docs = get_vectorstore().similarity_search(state["query"], k=config.TOP_K)
    return {"documents": docs}


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
    return {"grade": "sufficient" if "yes" in verdict else "insufficient"}


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
    "아래 문서 내용에 근거해서만 한국어로 답변하세요.\n"
    "문서에 없는 내용은 추측하지 말고 '문서에서 찾을 수 없습니다'라고 답하세요.\n\n"
    "문서:\n{context}\n\n질문: {question}\n\n답변:"
)


def generate(state: AgentState) -> dict:
    context = "\n---\n".join(d.page_content for d in state["documents"])
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
