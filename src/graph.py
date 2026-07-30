"""LangGraph 기반 Corrective-RAG Agent.

실행 흐름:
    retrieve → grade ─(충분)→ generate
                     └(부족)→ rewrite → retrieve  (최대 MAX_REWRITES회)

State로 질문/문서/재작성 횟수를 관리하며, 검색 품질이 낮으면
질문을 재작성해 재검색하는 self-corrective 루프를 구성한다.
"""
import hashlib
import re
import threading
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

import config


class AgentState(TypedDict):
    question: str          # 사용자 원본 질문
    query: str             # 실제 검색에 사용하는 질의 (재작성될 수 있음)
    documents: list[Document]
    rewrites: int          # 질문 재작성 횟수
    grade: str             # 검색 품질 판정 (sufficient / insufficient)
    rewrite_failed: bool   # 재작성이 쓸 만한 질의를 못 만들었는가
    answer: str
    sources: list[str]


_vectorstore = None
_bm25 = None
_index_lock = threading.Lock()


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
    """FAISS·BM25 인덱스를 한 번만 만들어 재사용한다 (스레드 안전).

    api.py의 `/ask`는 동기 핸들러라 FastAPI가 스레드풀에서 돌린다. 락 없이
    지연 초기화하면, 첫 요청이 BM25를 만드는 동안(수백 ms) 들어온 두 번째
    요청이 '_vectorstore가 이미 있으니 준비됐다'고 보고 아직 None인 _bm25를
    받아 간다 → bm25.invoke()에서 AttributeError. 그래서 (1) 락으로 감싸고
    (2) 두 인덱스가 **모두** 준비된 뒤에 전역에 공개한다.
    """
    global _vectorstore, _bm25
    if _bm25 is not None:                 # 준비 완료 후엔 락 없이 통과
        return _vectorstore, _bm25

    with _index_lock:
        if _bm25 is None:                 # 락을 기다리는 동안 남이 채웠을 수 있다
            import json

            from langchain_community.retrievers import BM25Retriever

            import vectorstore as vs

            store = vs.load()             # config.VECTOR_STORE로 FAISS/Qdrant 선택
            chunks = []
            with open(config.CHUNKS_PATH, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    chunks.append(Document(page_content=d["page_content"],
                                           metadata=d["metadata"]))
            bm25 = BM25Retriever.from_documents(
                chunks, preprocess_func=bm25_tokenize)
            bm25.k = config.TOP_K

            # 마지막에 세팅되는 _bm25가 '준비 완료' 신호다 — 순서를 바꾸지 말 것
            _vectorstore = store
            _bm25 = bm25
    return _vectorstore, _bm25


def warmup() -> None:
    """인덱스를 미리 적재한다 (서버 기동 시 1회 — 첫 요청 지연·경합 제거)."""
    _load_indexes()


def hybrid_search(query: str) -> list[Document]:
    """FAISS(의미) + BM25(키워드) 결과를 RRF(Reciprocal Rank Fusion)로 융합.

    'Jenkins', 'Pyannote' 같은 고유명사/키워드 질문은 벡터 검색이 놓치기 쉬워
    BM25를 결합해 검색 재현율을 보완한다. RRF score = Σ 1 / (k + rank).
    """
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
            # 같은 사실이 resume.md·portfolio.md 양쪽에 중복돼 내용까지 같으면
            # 두 Document가 같은 키로 합쳐진다. 나중 것으로 덮어쓰면 출처가
            # 검색 순서에 따라 임의로 바뀌므로, 먼저 본 것(상위 랭크)을 유지한다.
            by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_key[k] for k in ranked[: config.TOP_K]]


_llm_cache: dict[tuple[str, float], ChatOllama] = {}


# 판정(grade)·생성(generate)은 재현성이 중요하므로 temperature 0으로 고정.
# 질문 재작성(rewrite)만 다양성이 필요해 0.3을 사용한다.
def get_llm(temperature: float = 0.0) -> ChatOllama:
    # 캐시 키에 모델명을 포함한다. eval_tool_chain.py처럼 런타임에
    # config.LLM_MODEL을 바꿔 모델 간 비교를 하는 스크립트가 있어서,
    # temperature만 키로 쓰면 3b용 인스턴스가 7b 실행에 재사용된다.
    key = (config.LLM_MODEL, temperature)
    if key not in _llm_cache:
        _llm_cache[key] = ChatOllama(
            model=config.LLM_MODEL, temperature=temperature
        )
    return _llm_cache[key]


# ── Nodes ──────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    return {"documents": hybrid_search(state["query"])}


# generate가 실제로 받는 청크 수. grade와 generate가 이 상수를 공유해야
# "grade는 통과시켰는데 generate는 그 근거를 못 받는" 불일치가 생기지 않는다
# ("근무한 회사들" 질문이 실제로 그 상태였다 — README v6 기록).
GENERATE_TOP_N = 3


def context_docs(documents: list[Document]) -> list[Document]:
    """grade·generate가 공통으로 봐야 할 상위 N개."""
    return documents[:GENERATE_TOP_N]


def context_text(docs: list[Document]) -> str:
    """청크 목록을 프롬프트에 넣을 하나의 문자열로.

    grade와 generate가 **같은 텍스트**를 보게 하려고 있는 함수다. 예전에는
    grade만 page_content[:500]으로 잘라 봤는데, 청크 중앙값이 711자이고
    다이어그램 청크는 4천 자가 넘어서 grade가 근거의 상당 부분을 못 봤다.
    개수(GENERATE_TOP_N)를 맞춰 놓고 길이는 안 맞춘 상태였던 셈이라,
    "grade는 통과시켰는데 generate는 근거를 못 받는"(그리고 그 반대인)
    불일치가 그대로 남아 있었다.
    """
    return "\n---\n".join(d.page_content for d in docs)


GRADE_PROMPT = ChatPromptTemplate.from_template(
    "당신은 검색 품질 평가자입니다.\n"
    "질문: {question}\n\n"
    "검색된 문서:\n{context}\n\n"
    "문서에 질문과 관련된 정보가 일부라도 포함되어 있으면 'yes', "
    "전혀 관련 없는 내용뿐이면 'no'만 출력하세요."
)


# 판정 3값. 예전엔 yes/no 두 값이라 **파싱 실패가 sufficient에 흡수됐다** —
# fail-open 정책은 맞지만, 그래서 실패가 몇 %인지 아무도 모르는 상태였다.
# 세 번째 값을 만들어 세는 것이 이 파서의 핵심 변경이다.
GRADE_YES, GRADE_NO, GRADE_UNPARSED = "yes", "no", "unparsed"

# 2단 판정. ① 지시대로 답했다면 판정어가 **맨 앞**에 온다 → 앞부분에 명시적
# 판정어가 있으면 그게 이긴다(뒤에 붙는 건 사족이다). ② 없으면 서술문에서
# 찾아본다. 한 번에 뭉쳐서 보면 "yes. 다만 관련 없는 부분도…"가 yes/no 양쪽에
# 걸려 판정 실패로 떨어진다.
_VERDICT_LEAD = 16      # 판정어를 찾는 선두 구간
_VERDICT_HEAD = 80      # 서술문을 훑는 구간

# ^ 앵커 + \b 단어경계라 'no.' 'no,'가 특례 없이 잡힌다.
_LEAD_NO = re.compile(
    # '아닙니다'는 아+닙+니+다라서 '아니' 부분매칭으로 **안 잡힌다**
    # (옛 파서가 못 잡던 케이스다. 음절 단위로 열거해야 한다.)
    r"^\W*(no|n)\b|^\W*(아니(오|요|다|에요|었)|아닙니다|아녜요|아님)")
_LEAD_YES = re.compile(r"^\W*(yes|y|예|네)\b")

# 서술문 패턴. '있는지/없는지'는 의문·확인 어미이므로 판정이 아니다 —
# "관련 있는지 아니면…"을 긍정으로 읽던 것이 실제 오탐이었다.
_PROSE_NO = re.compile(r"관련\s*(정보|내용|성)?\s*(이|가)?\s*없(?!는지|은지)"
                       r"|찾을\s*수\s*없|no relevant|not relevant")
_PROSE_YES = re.compile(r"관련\s*(정보|내용|성)?\s*(이|가)?\s*있(?!는지|은지)"
                        r"|포함되어\s*있(?!는지|은지)")


def parse_verdict(raw: str) -> str:
    """grade 응답 → GRADE_YES / GRADE_NO / GRADE_UNPARSED."""
    text = raw.strip().lower()

    lead = text[:_VERDICT_LEAD]                 # ① 명시적 판정어 우선
    if _LEAD_NO.search(lead):
        return GRADE_NO
    if _LEAD_YES.search(lead):
        return GRADE_YES

    head = text[:_VERDICT_HEAD]                 # ② 서술문 폴백
    yes, no = bool(_PROSE_YES.search(head)), bool(_PROSE_NO.search(head))
    if yes == no:                               # 둘 다거나 둘 다 아니면 실패
        return GRADE_UNPARSED
    return GRADE_YES if yes else GRADE_NO


def judge_relevance(question: str, docs: list[Document]) -> tuple[str, str]:
    """(3값 판정, 모델 원문). 평가 스크립트가 파싱 실패까지 세도록 분리해 둔다."""
    raw = (GRADE_PROMPT | get_llm()).invoke(
        {"question": question, "context": context_text(docs)}).content
    return parse_verdict(raw), raw


def grade(state: AgentState) -> dict:
    verdict, raw = judge_relevance(
        state["question"], context_docs(state["documents"]))
    if verdict == GRADE_UNPARSED:
        # 정책은 fail-open 유지(애매하면 통과) — 단 조용히 흡수하지 않는다.
        # 빈도를 모르면 고칠 수 없다. eval/eval_grade.py가 이 비율을 잰다.
        print(f"[grade] 판정 파싱 실패 → sufficient(fail-open): "
              f"{raw.strip()[:60]!r}")
    return {"grade": "insufficient" if verdict == GRADE_NO else "sufficient"}


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "다음 질문으로 문서 검색을 했지만 관련 문서를 찾지 못했습니다.\n"
    "검색이 잘 되도록 핵심 키워드 중심으로 질문을 한 문장으로 재작성하세요.\n"
    "재작성된 질문만 출력하세요.\n\n질문: {question}"
)


# "재작성된 질문:" 처럼 모델이 붙이는 접두어. 그대로 두면 이 라벨까지
# BM25 토큰으로 들어가서 검색을 흐린다.
_REWRITE_PREFIX = re.compile(
    r"^\s*(재작성(된)?\s*질문|질문|검색\s*질의|답변)\s*[:：]\s*")
# 콜론으로 끝나는 줄은 안내문이다. 실측에서 3B가 이런 걸 뱉었다:
#   "…설명하는 문장을 만드는데 도움이 되겠습니다. 재작성된 질문은 다음과 같습니다:"
# 실제 질문이 그 다음 줄에 오는 경우가 있어, 첫 줄만 보면 질문을 통째로 잃는다.
_PREAMBLE_LINE = re.compile(r"[:：]\s*$")
_MAX_QUERY_CHARS = 200


def clean_rewrite(raw: str, original: str) -> str | None:
    """재작성 출력 정리. 쓸 수 없으면 None.

    grade 파싱에는 공을 들였는데 여기는 출력 전체를 그대로 검색어로 쓰고
    있었다. 첫 줄만 취하고(모델이 설명을 덧붙인다), 접두어·인용부호를
    떼고, 빈 출력이나 원 질문과 똑같은 결과는 실패로 본다 — 같은 질의로
    재검색하면 같은 결과가 나올 뿐이다.
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    candidates = [ln for ln in lines if not _PREAMBLE_LINE.search(ln)]
    if not candidates:
        return None          # 안내문뿐 — 검색어로 쓸 게 없다
    line = _REWRITE_PREFIX.sub("", candidates[0]).strip().strip("\"'`“”‘’")
    if len(line) < 2 or line == original.strip():
        return None
    return line[:_MAX_QUERY_CHARS]


def rewrite(state: AgentState) -> dict:
    chain = REWRITE_PROMPT | get_llm(temperature=0.3)
    raw = chain.invoke({"question": state["question"]}).content
    new_query = clean_rewrite(raw, state["question"])
    if new_query is None:
        # 재작성 실패 — 재검색해도 같은 결과다. 있는 근거로 답하게 보낸다.
        # (rewrite_failed는 AgentState에 **선언돼 있어야** 한다. 선언 안 된
        #  키를 반환하면 LangGraph가 조용히 버린다 — 이 프로젝트 첫 버그.)
        print(f"[rewrite] 재작성 실패 → 재검색 생략: {raw.strip()[:60]!r}")
        return {"rewrites": state["rewrites"] + 1, "rewrite_failed": True}
    return {"query": new_query, "rewrites": state["rewrites"] + 1,
            "rewrite_failed": False}


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
    # 소형 모델은 긴 컨텍스트에서 근거를 놓치기 쉬우므로 상위 N개만 쓰고
    # (N은 grade와 공유 — context_docs), 끝부분 주의집중이 강한 특성에 맞춰
    # 랭크 역순으로 배치해 최상위 청크가 질문 바로 앞에 오게 한다
    used = context_docs(state["documents"])
    context = context_text(list(reversed(used)))
    chain = GENERATE_PROMPT | get_llm()
    answer = chain.invoke(
        {"question": state["question"], "context": context}
    ).content.strip()
    # 출처는 **실제로 프롬프트에 들어간 청크**에서만 뽑는다. 예전에는 검색된
    # TOP_K(6개) 전부에서 뽑아, 답변 생성에 쓰이지도 않은 문서가 근거로
    # 표시됐다 — 개수는 context_docs로 맞춰 놓고 인용은 안 맞춘 상태였다.
    sources = sorted({d.metadata.get("source", "?") for d in used})
    return {"answer": answer, "sources": sources}


# ── Graph ──────────────────────────────────────────────

def needs_grading(state: AgentState) -> str:
    """재작성 여력이 없으면 판정을 아예 묻지 않는다.

    decide_next가 `rewrites >= MAX_REWRITES`면 무조건 generate로 보내므로,
    그 상태에서 실행되는 grade의 판정 결과는 **라우팅을 바꿀 수 없다** —
    3B 호출 한 번이 통째로 버려진다. 두 군데서 발생했다:
      · 메인 경로의 2회차(재작성 후 재검색) grade
      · team.py의 worker 전부 (rewrites=MAX_REWRITES로 시작해 재작성을 끈다)
    멀티홉 sub-질문 3개면 무의미한 호출 3회다. 여기서 잘라낸다.

    MAX_REWRITES=0이면 grade·rewrite가 통째로 빠져 순수 RAG가 된다 —
    corrective 루프의 A/B(eval/ab_rewrite.py)가 이걸 이용한다.
    """
    return "generate" if state["rewrites"] >= config.MAX_REWRITES else "grade"


def decide_next(state: AgentState) -> str:
    """검색 품질과 재작성 횟수에 따라 다음 노드 결정."""
    if state.get("grade") == "sufficient" or state["rewrites"] >= config.MAX_REWRITES:
        return "generate"
    return "rewrite"


def after_rewrite(state: AgentState) -> str:
    """재작성이 실패했으면 재검색을 건너뛴다 (같은 질의 = 같은 결과)."""
    return "generate" if state.get("rewrite_failed") else "retrieve"


def build_graph():
    g = StateGraph(AgentState)
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

    q = sys.argv[1] if len(sys.argv) > 1 else "TTS 프로젝트의 TTFB 개선 수치는?"
    out = ask(q)
    print("답변:", out["answer"])
    print("출처:", out["sources"])
