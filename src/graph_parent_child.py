"""Parent-Child RAG — child로 검색·판정하고 parent로 생성한다.

기본 graph.py와 흐름은 동일하다(retrieve → grade → generate, 부족하면
rewrite). 다른 건 generate 직전 한 지점뿐이다:

    retrieve → grade → [child를 parent로 확장] → generate

grade까지 parent로 넘기면 child 검색의 정밀도 이점(작은 청크 = 주제 하나만
담겨 매칭이 명확함)이 판정 단계에서부터 흐려진다. 그래서 판정은 child로
하고, "이걸로 답을 써도 되겠다"고 확정된 다음에만 맥락을 부풀린다.

의존: python src/ingest_parent_child.py 로 만든 인덱스가 있어야 한다.
"""
import json
from pathlib import Path
from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

import config
from graph import (
    GRADE_NO,
    GRADE_UNPARSED,
    REWRITE_PROMPT,
    bm25_tokenize,
    clean_rewrite,
    context_docs,
    context_text,
    generate_prompt,
    get_llm,
    judge_relevance,
    order_for_prompt,
)


class ParentChildState(TypedDict):
    question: str
    query: str
    documents: list[Document]     # child 문서 (검색·판정용)
    rewrites: int
    grade: str
    rewrite_failed: bool
    answer: str
    sources: list[str]
    contexts: list[str]


_vectorstore = None
_bm25 = None
_parent_by_id: dict[str, str] = {}   # parent_id → parent 원문


def _load_parent_store():
    """parent_id → 원문. vectorstore/BM25와 독립적으로 로드한다 —
    expand_to_parents()는 이것만 있으면 되고, 이렇게 나눠야
    단위 테스트에서 _parent_by_id를 미리 주입해 격리할 수 있다."""
    global _parent_by_id
    if not _parent_by_id:
        with open(config.PARENT_STORE_PATH, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                _parent_by_id[d["parent_id"]] = d["page_content"]
    return _parent_by_id


def _load_indexes():
    global _vectorstore, _bm25
    if _vectorstore is None:
        from langchain_community.retrievers import BM25Retriever

        _vectorstore = FAISS.load_local(
            config.PARENT_DB_DIR,
            OllamaEmbeddings(model=config.EMBED_MODEL),
            allow_dangerous_deserialization=True,
        )
        children = []
        with open(config.PARENT_CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                children.append(Document(page_content=d["page_content"],
                                         metadata=d["metadata"]))
        _bm25 = BM25Retriever.from_documents(
            children, preprocess_func=bm25_tokenize)
        _bm25.k = config.TOP_K
    _load_parent_store()
    return _vectorstore, _bm25


def hybrid_search_child(query: str) -> list[Document]:
    """graph.hybrid_search와 완전히 동일한 RRF 융합, 대상만 child 인덱스."""
    import hashlib

    vectorstore, bm25 = _load_indexes()
    vec_docs = vectorstore.similarity_search(query, k=config.TOP_K)
    kw_docs = bm25.invoke(query)

    K = config.RRF_K
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in (vec_docs, kw_docs):
        for rank, doc in enumerate(docs):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_key[k] for k in ranked[: config.TOP_K]]


def expand_to_parents(children: list[Document]) -> list[Document]:
    """child 목록을 parent 목록으로 바꾼다. 같은 parent에 속한 child가
    여러 개 뽑혀도 parent는 한 번만 넣는다(중복 맥락 방지), 순서는 유지."""
    _load_parent_store()
    seen, parents = set(), []
    for child in children:
        pid = child.metadata.get("parent_id")
        if pid in seen or pid not in _parent_by_id:
            continue
        seen.add(pid)
        parents.append(Document(
            page_content=_parent_by_id[pid], metadata=child.metadata))
    return parents


# ── Nodes — retrieve/grade/rewrite는 child 기준, generate만 parent로 확장 ──

def retrieve(state: ParentChildState) -> dict:
    return {"documents": hybrid_search_child(state["query"])}


def grade(state: ParentChildState) -> dict:
    # base graph.py의 3값 판정기를 그대로 재사용한다 — child 문서만 다르다.
    verdict, raw = judge_relevance(
        state["question"], context_docs(state["documents"]))
    if verdict == GRADE_UNPARSED:
        print(f"[grade] 판정 파싱 실패 → sufficient(fail-open): "
              f"{raw.strip()[:60]!r}")
    return {"grade": "insufficient" if verdict == GRADE_NO else "sufficient"}


def rewrite(state: ParentChildState) -> dict:
    chain = REWRITE_PROMPT | get_llm(temperature=0.3)
    raw = chain.invoke({"question": state["question"]}).content
    new_query = clean_rewrite(raw, state["question"])
    if new_query is None:
        print(f"[rewrite] 재작성 실패 → 재검색 생략: {raw.strip()[:60]!r}")
        return {"rewrites": state["rewrites"] + 1, "rewrite_failed": True}
    return {"query": new_query, "rewrites": state["rewrites"] + 1,
            "rewrite_failed": False}


# parent를 몇 개까지 생성에 넘길지 — base의 GENERATE_TOP_N과 맞춰서
# child·parent 두 파이프라인이 같은 "몇 번의 기회"를 받게 한다.
#
# 예전엔 3으로 고정해 뒀었다(당시 base의 GENERATE_TOP_N도 3). 그 사이 base가
# 51문항 스윕으로 top-N을 5로 올렸으므로(config.py 주석 참고) 여기도 같이
# 올린다 — 그러지 않으면 "child recall은 낮은데 정답률은 base와 동률"이라는
# 비교 자체가 서로 다른 예산(3개 vs 5개)을 쓰는 불공정 비교가 된다.
#
# top-N 자체를 둘러싼 과거 실험(2로 줄이면 오답 유입, 1로 줄이면 다른 질문
# 실패 증가)은 config.GENERATE_TOP_N=3 시절 결과라 5 기준으로 재검증했다.
#
# 재검증 결과 (top-5, 새 3값 grade, 개선된 BM25 토크나이저 반영 후):
#   - "논문 몇 편?" — parent-child가 풀던 문제였는데, **base도 top-5에서는
#     5/5로 통과한다.** base의 top-N을 3→5로 올린 것 자체가 이 문제를
#     해결해 버려서, parent-child가 더 이상 유일한 해법이 아니다.
#   - TTS TTFB 질문 — 여전히 base·parent-child 둘 다 5/5 (동률 유지).
#   - child 검색 recall@1: 78%(BM25 토크나이저 공유 개선으로 56%→78%
#     올랐지만) vs base 89% — 여전히 base보다 낮다.
#   - 9문항 비교(compare_parent_child.py)에서 생성 답변 품질도 두 파이프
#     라인이 사실상 동등했다.
#
# 결론: 지금 base 설정(top-5)에서는 parent-child가 base 대비 뚜렷한 이점을
# 못 준다. 애초에 풀려던 문제가 더 단순한 수정(top-N 확장)으로 해결됐기
# 때문이다. 코드·테스트는 유지하되(그 자체로 유효한 청킹 실험이고 정보
# 희석·recall-정답률 괴리 등 배운 것은 남는다), base를 여전히 기본
# 파이프라인으로 둔다. child 인덱스 recall이 base를 넘어서거나, base의
# top-5로도 못 푸는 새로운 실패 사례가 나오면 다시 볼 만하다.
PARENT_TOP_N = config.GENERATE_TOP_N


def generate(state: ParentChildState) -> dict:
    # 여기가 기본 graph.py와의 유일한 실질적 차이다: context_docs로 뽑은
    # 상위 child들을 parent로 확장해서 생성 프롬프트에 넣는다. parent가
    # child보다 훨씬 크므로(800자 vs 300자) 개수는 PARENT_TOP_N으로 별도 관리.
    # 배치 순서·프롬프트 변형은 base와 동일하게 config를 그대로 따른다.
    top_children = context_docs(state["documents"])[:PARENT_TOP_N]
    parents = expand_to_parents(top_children)
    context = context_text(order_for_prompt(parents))
    chain = generate_prompt() | get_llm()
    answer = chain.invoke(
        {"question": state["question"], "context": context}
    ).content.strip()
    # 출처는 실제로 프롬프트에 들어간 parent 청크에서만 뽑는다 (base graph.py의
    # generate와 같은 이유 — state["documents"]는 top_children 자르기 전의
    # 검색 결과 전체라, 여기서 뽑으면 생성에 쓰이지도 않은 문서가 근거로
    # 표시된다).
    sources = sorted({d.metadata.get("source", "?") for d in parents})
    return {"answer": answer, "sources": sources,
            "contexts": [d.page_content for d in parents]}


def needs_grading(state: ParentChildState) -> str:
    """base graph.py와 동일한 이유로 재작성 여력이 없으면 grade를 건너뛴다."""
    return "generate" if state["rewrites"] >= config.MAX_REWRITES else "grade"


def decide_next(state: ParentChildState) -> str:
    if state.get("grade") == "sufficient" or state["rewrites"] >= config.MAX_REWRITES:
        return "generate"
    return "rewrite"


def after_rewrite(state: ParentChildState) -> str:
    return "generate" if state.get("rewrite_failed") else "retrieve"


def build_graph():
    g = StateGraph(ParentChildState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", needs_grading, ["grade", "generate"])
    g.add_conditional_edges("grade", decide_next, ["rewrite", "generate"])
    g.add_conditional_edges("rewrite", after_rewrite, ["retrieve", "generate"])
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

    if not Path(config.PARENT_DB_DIR).exists():
        raise SystemExit(
            "parent-child 인덱스가 없다. 먼저 실행: "
            "python src/ingest_parent_child.py")

    q = sys.argv[1] if len(sys.argv) > 1 else "TTS 프로젝트의 TTFB 개선 수치는?"
    out = ask(q)
    print("답변:", out["answer"])
    print("출처:", out["sources"])
