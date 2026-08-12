"""LangGraph 기반 Corrective-RAG Agent.

실행 흐름:
    retrieve → grade ─(충분)→ generate
                     └(부족)→ rewrite → retrieve  (최대 MAX_REWRITES회)

State로 질문/문서/재작성 횟수를 관리하며, 검색 품질이 낮으면
질문을 재작성해 재검색하는 self-corrective 루프를 구성한다.
"""
import functools
import hashlib
import re
import threading
import time
import unicodedata
from typing import Annotated, TypedDict

from kiwipiepy import Kiwi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config


def _merge_timings(old: dict | None, new: dict | None) -> dict:
    """노드별 시간을 합친다.

    LangGraph는 노드가 돌려준 dict를 State에 병합하는데, 리듀서가 없으면
    같은 키를 **덮어쓴다** — timings를 그냥 dict로 두면 마지막 노드의
    기록만 남는다. 그리고 retrieve는 재작성 후 다시 도므로 누적해야
    "재작성이 발동하면 검색에 시간을 두 배 쓴다"가 보인다.
    """
    merged = dict(old or {})
    for node, sec in (new or {}).items():
        merged[node] = merged.get(node, 0.0) + sec
    return merged


def timed(fn):
    """노드를 감싸 소요 시간을 State에 남긴다.

    end-to-end 지연은 tracelog가 원래 기록했지만 **배분은 없었다** — 이
    저장소는 recall·정확도·판정기 정확도까지 다 재면서 정작 자기 파이프라인이
    시간을 어디에 쓰는지는 안 쟀다. "grade가 87% 정확하다"는 알아도 "grade가
    전체의 몇 %를 먹는다"는 답할 수 없었다.

    배선에서만 감싼다(build_graph) — 노드 함수 자체는 그대로라 단위 테스트와
    직접 호출(eval 스크립트)이 영향을 안 받는다.
    """
    @functools.wraps(fn)
    def wrapper(state):
        t0 = time.perf_counter()
        out = fn(state)
        return {**out, "timings": {fn.__name__: time.perf_counter() - t0}}

    return wrapper


class AgentState(TypedDict):
    question: str          # 사용자 원본 질문
    query: str             # 실제 검색에 사용하는 질의 (재작성될 수 있음)
    documents: list[Document]
    rewrites: int          # 질문 재작성 횟수
    grade: str             # 검색 품질 판정 (sufficient / insufficient)
    rewrite_failed: bool   # 재작성이 쓸 만한 질의를 못 만들었는가
    answer: str
    sources: list[str]
    contexts: list[str]     # 실제로 프롬프트에 들어간 청크 본문 (평가·심판용)
    strategy: dict         # 유형 라우팅이 정한 컨텍스트 전략 ({}면 전역 기본값)
    timings: Annotated[dict, _merge_timings]   # 노드별 소요 시간(초), 재방문은 누적
    grounded: bool | None          # verify 노드 결과 (VERIFY_GROUNDING 꺼지면 None)
    unsupported_claim: str | None  # grounded=False일 때 근거 없는 주장 요약


_vectorstore = None
_bm25 = None
_index_lock = threading.Lock()


# 천단위 콤마 제거. Kiwi는 숫자를 SN 태그로 인식하지만 콤마를 그대로
# 형태(form)에 남긴다("2,292" ≠ "2292") — 지우지 않으면 겹치는 토큰이
# 0개. 반대로 "Jenkins, ArgoCD" 같은 열거 콤마는 사이에 공백이 있어
# \d{3}에 걸리지 않는다 — 코퍼스 전수 검사로 확인(portfolio.md 24,000·
# 32,768·9,600·2,048 등 전부 천단위였고 목록 용도의 무공백 콤마 나열은
# 0건이었다).
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

_kiwi = None
_kiwi_lock = threading.Lock()

# 체언(N*)·용언 어간(V*)·부사(MA*)·어근(XR)·외국어(SL)·한자(SH)·숫자(SN)·
# URL·이메일·일련번호(W_*, 예: "10-2538225-0000"→W_SERIAL)만 남긴다.
# 조사(JK*/JX/JC)·어미(E*)·순수 기호(SF/SP/SS/SE/SO/SW)는 버린다 — 이게 곧
# '회사들'에서 '들'(XSN)을 떼고 '회사'만 남기는 지점이라 별도 bigram
# 없이도 조사가 붙은 질의가 겹친다.
_KEEP_TAG_PREFIXES = ("N", "V", "MA", "XR", "W")
_KEEP_TAGS = frozenset({"SL", "SH", "SN"})


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        with _kiwi_lock:
            if _kiwi is None:
                _kiwi = Kiwi()
    return _kiwi


def bm25_tokenize(text: str) -> list[str]:
    """Kiwi 형태소 분석으로 어간·명사·숫자·외국어 토큰만 추출한다.

    이전엔 문자 종류별 런(run) 분해 + 한글 bigram으로 조사 문제를
    우회했다("회사들" ↔ "회사"가 bigram '회사'로 겹침). 하지만 bigram은
    형태소 경계를 모르는 근사치라 노이즈 토큰이 섞이고('회사들을' →
    '사들'·'들을' 같은 의미 없는 조각까지 토큰화됨), 진짜 복합명사
    분해("화자분할" → "화자"+"분할")는 못 했다.

    Kiwi는 형태소 경계를 실제로 알아서 이 둘을 한 번에 해결한다 —
    "회사들을" → 어간 '회사'만 남고(들=XSN 접미사, 을=JKO 조사는 버림),
    "화자분할"·"화자 분할" 둘 다 ['화자','분할']로 동일하게 분해된다.
    영문·숫자도 SL/SN 태그로 그대로 보존되어 "Throughput은"(SL+JX)과
    "Throughput:"(SL+SP)이 똑같이 'throughput' 하나로 겹친다 — 예전엔
    구두점·조사가 토큰에 눌어붙어 겹치는 토큰이 0개였던 자리다.

    다만 Kiwi도 두 가지는 직접 안 해준다:
      1. 천단위 콤마를 형태(form)에 그대로 남긴다 — 지우지 않으면
         "2,292"(SN)과 "2292"(SN)가 다른 토큰이 된다. 토큰화 전에 지운다.
      2. 수학 이탤릭 유니코드("𝐼𝑜𝑈")·원문자 글머리("①②③")·위첨자("N²")는
         PDF 수식 추출(pypdf)이 그대로 뽑아내는데, NFKC로 정규화하지
         않으면 "IoU"와 별개 문자로 인식한다. 토큰화 전에 NFKC를 한 번
         통과시킨다(완성형 한글 음절은 NFKC에서도 NFC와 동일해 영향 없음).

    "password_changed_at"처럼 밑줄로 이어붙인 식별자는 Kiwi가
    'password'·'changed'·'at' 세 개의 SL 토큰(+ SW 기호, 버려짐)으로
    쪼갠다 — 예전엔 통짜 토큰으로 유지했지만, 질의도 똑같이 쪼개지므로
    BM25 매칭 자체는 깨지지 않는다(eval_retrieval.py로 회귀 없음 확인).
    "10-2538225-0000" 같은 일련번호는 Kiwi가 W_SERIAL 태그로 통째로
    인식해 오히려 더 정확히 보존된다.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _THOUSANDS_COMMA.sub("", text)
    kiwi = _get_kiwi()
    tokens = []
    for t in kiwi.tokenize(text):
        if t.tag in _KEEP_TAGS or t.tag.startswith(_KEEP_TAG_PREFIXES):
            tokens.append(t.form.lower())
    return tokens


class IndexError_(RuntimeError):
    """인덱스가 없거나 chunks.jsonl과 어긋날 때."""


def check_index_consistency() -> None:
    """벡터 인덱스와 chunks.jsonl이 같은 인제스트 산출물인지 확인한다.

    벡터 검색은 인덱스를, BM25는 chunks.jsonl을 각각 읽는다. 둘이 어긋나면
    **서로 다른 청킹 두 개를 RRF로 섞게 되는데 아무 증상이 없다** — 검색이
    조금 이상해질 뿐이라 원인을 찾을 단서가 없다. sweep_chunk_size.py가 두
    파일을 매 스텝 덮어쓰므로 중간에 중단되면 실제로 이 상태가 된다.

    인제스트가 남긴 매니페스트의 청크 지문과 대조해 조기에 실패시킨다.
    """
    import json
    from pathlib import Path

    if not Path(config.CHUNKS_PATH).exists():
        raise IndexError_(
            f"청크 파일이 없습니다: {config.CHUNKS_PATH}\n"
            "먼저 `python src/ingest.py`로 인덱스를 구축하세요.")

    manifest_path = Path(config.INDEX_MANIFEST)
    if not manifest_path.exists():
        # 매니페스트 도입 이전에 만든 인덱스 — 막지는 않고 알린다.
        print("[warn] 인덱스 매니페스트가 없습니다. chunks.jsonl과 벡터 인덱스가"
              " 같은 인제스트 산출물인지 확인할 수 없습니다."
              " `python src/ingest.py`로 재구축을 권장합니다.")
        return

    import ingest

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = ingest.chunk_fingerprint(config.CHUNKS_PATH)
    if manifest.get("chunks_md5") != actual:
        raise IndexError_(
            "벡터 인덱스와 chunks.jsonl이 어긋났습니다.\n"
            f"  매니페스트 기록: {manifest.get('chunks_md5')} "
            f"({manifest.get('n_chunks')}청크, "
            f"chunk_size={manifest.get('chunk_size')})\n"
            f"  현재 chunks.jsonl: {actual}\n"
            "서로 다른 청킹을 벡터·BM25가 각각 쓰게 되므로 검색 결과를 믿을 수 "
            "없습니다. `python src/ingest.py`로 재구축하세요.\n"
            "(sweep_chunk_size.py를 중단했다면 이 상태가 됩니다.)")


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

            check_index_consistency()
            store = vs.load()             # config.VECTOR_STORE로 FAISS/Qdrant 선택
            chunks = []
            with open(config.CHUNKS_PATH, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    chunks.append(Document(page_content=d["page_content"],
                                           metadata=d["metadata"]))
            bm25 = BM25Retriever.from_documents(
                chunks, preprocess_func=bm25_tokenize)

            # 마지막에 세팅되는 _bm25가 '준비 완료' 신호다 — 순서를 바꾸지 말 것
            _vectorstore = store
            _bm25 = bm25
            global _by_index_cache
            _by_index_cache = None      # 인덱스가 바뀌면 조회표도 무효
    return _vectorstore, _bm25


def warmup() -> None:
    """인덱스를 미리 적재한다 (서버 기동 시 1회 — 첫 요청 지연·경합 제거)."""
    _load_indexes()


def rrf_fuse(doc_lists: list[list[Document]]) -> list[Document]:
    """순위 목록 여러 개를 RRF(Reciprocal Rank Fusion)로 융합.

    RRF score = Σ 1 / (K + rank). hybrid_search에서 분리한 이유는
    HyDE가 목록을 2개 더 얹으면서 융합 자체를 단위 테스트할 수 있어야
    했기 때문이다(kg.fused_search와 같은 계산이지만 그쪽은 코퍼스 조회표
    주입이 얽혀 있어 재사용하지 않았다).
    """
    K = config.RRF_K
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in doc_lists:
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


def hybrid_search(query: str) -> list[Document]:
    """FAISS(의미) + BM25(키워드) 결과를 RRF로 융합.

    'Jenkins', 'Pyannote' 같은 고유명사/키워드 질문은 벡터 검색이 놓치기 쉬워
    BM25를 결합해 검색 재현율을 보완한다.

    HYDE가 켜져 있으면 가상 답변 단락(hypothetical_doc)으로 검색한 순위
    목록 2개(벡터·BM25)를 **추가**해 4개를 융합한다. 질의 목록을 대체하지
    않는 게 요점이다 — 질의·가상 단락 양쪽에서 잡히는 청크는 표가 겹쳐
    올라가고, 이미 질의만으로 잘 찾던 문항의 신호는 그대로 남는다.
    """
    vectorstore, bm25 = _load_indexes()
    # bm25.k를 질의 시점에 맞춘다. 예전엔 인덱스 빌드 때 한 번만 넣어서,
    # TOP_K를 런타임에 바꾸면(sweep_top_k.py 등) 벡터는 새 k로 BM25는 옛 k로
    # 뽑아 RRF가 비대칭이 됐다 — 증상 없이 순위만 틀어지는 종류다.
    bm25.k = config.TOP_K
    lists = [vectorstore.similarity_search(query, k=config.TOP_K),
             bm25.invoke(query)]
    if config.HYDE:
        hypo = hypothetical_doc(query)
        if hypo is not None:
            if config.HYDE_MODE == "terms":
                # BM25만 쓴다 — 틀린 추측(코퍼스에 없는 명칭)은 BM25에서
                # 아무것도 매치하지 않아 무해하지만, 임베딩은 그 추측을
                # 실제 신호로 받아 순위를 오염시킨다(1차 실측: GPU 문항
                # 5위 → MISS). config.HYDE_MODE 주석 참고.
                tq = hyde_term_query(query, hypo)
                if tq:
                    lists.append(bm25.invoke(tq))
            else:
                lists.append(vectorstore.similarity_search(hypo, k=config.TOP_K))
                lists.append(bm25.invoke(hypo))
    return rrf_fuse(lists)


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


class RerankOrder(BaseModel):
    """RRF 순위를 LLM이 다시 매긴다 — listwise, 한 번의 호출로.

    diagnose.py가 지목한 병목(검색O·정답X 10건)에 대한 시도다. RRF는
    등수만 합치는 결정적 융합이라 의미상 진짜 관련도는 못 본다. 문서마다
    따로 점수를 매기는 pointwise 리랭킹은 TOP_K개만큼 LLM을 더 불러야
    해서 이미 느린 CPU 추론에 안 맞고, cross-encoder 모델을 새로 추가하면
    이 프로젝트가 지켜온 "가벼운 의존성" 원칙(MCP 서버 의존성 2개 등)과
    어긋난다. 대신 grade와 같은 구조화 출력 패턴으로 **한 번에** 순서만
    다시 받는다.
    """
    ranked_indices: list[int] = Field(
        description="검색된 문서를 질문과의 관련도가 높은 순서로 나열한 "
                    "인덱스 목록(0부터 시작). 모든 인덱스를 정확히 한 번씩 포함")


RERANK_PROMPT = ChatPromptTemplate.from_template(
    "질문에 가장 관련 있는 문서 순서를 다시 매기려 합니다.\n"
    "질문: {question}\n\n"
    "문서 목록:\n{numbered_context}\n\n"
    "질문과의 관련도가 높은 순서대로 문서 인덱스를 정렬하세요."
)

_structured_rerank_cache: dict[str, object] = {}


def _structured_rerank_llm():
    model = config.LLM_MODEL
    if model not in _structured_rerank_cache:
        _structured_rerank_cache[model] = get_llm().with_structured_output(
            RerankOrder, include_raw=True)
    return _structured_rerank_cache[model]


def rerank(question: str, docs: list[Document]) -> list[Document]:
    """관련도 순으로 재정렬. 파싱 실패나 불완전한 순열이면 원래 순서를
    그대로 반환한다(fail-open — RRF 순위보다 나쁘게 만들 이유는 없다)."""
    if len(docs) <= 1:
        return docs
    numbered = "\n---\n".join(
        f"[{i}] {d.page_content[:400]}" for i, d in enumerate(docs))
    chain = RERANK_PROMPT | _structured_rerank_llm()
    result = chain.invoke({"question": question, "numbered_context": numbered})
    parsed = result.get("parsed")
    if parsed is None or set(parsed.ranked_indices) != set(range(len(docs))):
        return docs
    return [docs[i] for i in parsed.ranked_indices]


# ── Nodes ──────────────────────────────────────────────

def retrieve(state: AgentState) -> dict:
    docs = hybrid_search(state["query"])
    # 재정렬은 원 질문 기준 — grade가 query(재작성 가능)가 아니라 question을
    # 보는 것과 같은 이유다. 재작성된 검색어로 재정렬하면 정작 사용자가
    # 물은 것과는 다른 기준으로 순서가 매겨진다.
    if config.RERANK:
        docs = rerank(state["question"], docs)
    return {"documents": docs}


# generate가 실제로 받는 청크 수. grade와 generate가 이 상수를 공유해야
# "grade는 통과시켰는데 generate는 그 근거를 못 받는" 불일치가 생기지 않는다
# ("근무한 회사들" 질문이 실제로 그 상태였다 — README v6 기록).
def context_docs(documents: list[Document],
                 strategy: dict | None = None) -> list[Document]:
    """grade·generate가 공통으로 봐야 할 상위 N개.

    호출 시점에 config를 읽는다 — A/B 스크립트가 런타임에 바꿀 수 있어야 한다.
    이웃 확장(NEIGHBOR_WINDOW)도 여기서 적용해 grade와 generate가 끝까지
    **같은 텍스트**를 보게 한다.
    """
    docs = documents
    if config.EXCLUDE_DIAGRAMS:
        docs = [d for d in docs if d.metadata.get("kind") != "diagram"]
    st = strategy or {}
    top_n = st.get("GENERATE_TOP_N", config.GENERATE_TOP_N)
    return expand_with_neighbors(docs[:top_n], st.get("NEIGHBOR_WINDOW"))


def order_for_prompt(docs: list[Document]) -> list[Document]:
    """generate 프롬프트에 넣을 순서. grade는 순위 그대로 본다.

    기존은 무조건 역순이었다 — "소형 모델은 끝부분 주의가 강하니 1위를 질문
    바로 앞에" 라는 직관인데 잰 적이 없었다. 오늘 세 실험(top-6, TOP_K=15,
    이웃 확장)이 전부 위치 문제로 귀결됐으므로 이 규칙 자체를 잰다.
    """
    order = config.CONTEXT_ORDER
    if order == "ranked":
        return list(docs)
    if order == "sandwich" and len(docs) >= 3:
        # 1위를 양 끝에 둔다 — 가운데가 약하다는 가정을 시험
        top, rest = docs[0], list(docs[1:])
        return [top] + list(reversed(rest)) + [top]
    return list(reversed(docs))


def expand_with_neighbors(docs: list[Document],
                          window: int | None = None) -> list[Document]:
    """각 청크를 인덱스상 이웃과 합쳐 문맥을 넓힌다 (순위·개수는 유지).

    `NEIGHBOR_WINDOW=0`이면 아무것도 하지 않는다(기본값).

    핵심은 **개수를 안 늘린다**는 것이다. 하위 랭크 청크를 더 넣는 방식은
    두 번 실패했다 — generate가 랭크 역순 배치라 새로 들어온 근거가 프롬프트
    맨 앞에 놓여 모델이 못 본다. 여기서는 1위는 계속 1위 자리(질문 바로 앞)에
    있고, 그 청크가 담는 내용만 앞뒤로 넓어진다.

    청크가 800자 상한에서 잘리며 문장이 끊긴 경우를 이어 붙이는 효과도
    있다. 통짜 청크(다이어그램·표)는 이미 완결돼 있어 확장하지 않는다.
    """
    w = config.NEIGHBOR_WINDOW if window is None else window
    if w <= 0 or not docs:
        return docs

    # _chunks_by_index()는 chunks.jsonl만 읽는다 — 벡터 인덱스가 필요 없다.
    # 예전엔 여기서 _load_indexes()를 불러 FAISS 적재와 BM25 구축까지 하고
    # 있었는데 쓰지도 않는 작업이었고, 인덱스 없는 환경(새 클론·CI)에서
    # 이 경로만으로 RuntimeError가 났다.
    by_index = _chunks_by_index()
    out = []
    for d in docs:
        i = d.metadata.get("chunk_index")
        # kind가 있으면 통짜 청크다(다이어그램·표). 이미 완결된 블록이라
        # 앞뒤를 붙이면 문서의 다른 부분이 딸려 들어가 오히려 오염된다.
        if i is None or d.metadata.get("kind"):
            out.append(d)
            continue
        src = d.metadata.get("source")
        parts = []
        for j in range(i - w, i + w + 1):
            n = by_index.get(j)
            # 문서 경계를 넘지 않는다 — 다른 문서의 텍스트를 붙이면 노이즈다
            if n is not None and n.metadata.get("source") == src:
                parts.append(n.page_content)
        out.append(Document(page_content="\n".join(parts), metadata=d.metadata))
    return out


_by_index_cache: dict[int, Document] | None = None


def _chunks_by_index() -> dict[int, Document]:
    """chunk_index → Document. 이웃 확장용 조회표(인덱스와 같은 소스에서 만든다)."""
    global _by_index_cache
    if _by_index_cache is None:
        import json

        table = {}
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = d["metadata"].get("chunk_index")
                if idx is not None:
                    table[idx] = Document(page_content=d["page_content"],
                                          metadata=d["metadata"])
        _by_index_cache = table
    return _by_index_cache


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
    "문서에 질문과 관련된 정보가 일부라도 포함되어 있는지 판단하세요."
)


# 판정 3값. 예전엔 yes/no 두 값이라 **파싱 실패가 sufficient에 흡수됐다** —
# fail-open 정책은 맞지만, 그래서 실패가 몇 %인지 아무도 모르는 상태였다.
GRADE_YES, GRADE_NO, GRADE_UNPARSED = "yes", "no", "unparsed"


class GradeVerdict(BaseModel):
    """grade 판정을 자유 문장이 아니라 이 스키마로 강제한다.

    예전엔 "yes/no만 출력하라"고 지시한 뒤 정규식(앵커+서술문 2단)으로
    자유 문장을 역파싱했다. 그 파서를 eval_grade.py로 직접 재 보니
    오탐(충분한데 재검색 보냄) 20%가 나왔고, corrective 루프 A/B
    재검증(51문항 층화 표본)에서 이 오탐이 실제로 정답률을 깎는 것까지
    확인됐다(OFF 81%→ON 69%, 전부 grade가 잘못 재작성을 건 케이스).
    with_structured_output()으로 모델 출력 자체를 이 스키마에 강제하면
    "아닙니다가 왜 안 잡히나" 같은 문장 파싱 버그 클래스 자체가 없어진다.
    """
    relevant: bool = Field(
        description="문서에 질문과 관련된 정보가 일부라도 포함되어 있으면 "
                    "true, 전혀 관련 없는 내용뿐이면 false")


_structured_grade_cache: dict[str, object] = {}


def _structured_grade_llm():
    # get_llm()의 일반 캐시와 별도 — with_structured_output()으로 감싼
    # 버전을 모델별로 재사용한다. include_raw=True라야 파싱 실패 시에도
    # 예외 대신 parsed=None으로 받아 grade의 fail-open 정책을 그대로 쓸 수 있다.
    model = config.LLM_MODEL
    if model not in _structured_grade_cache:
        _structured_grade_cache[model] = get_llm().with_structured_output(
            GradeVerdict, include_raw=True)
    return _structured_grade_cache[model]


def judge_relevance(question: str, docs: list[Document]) -> tuple[str, str]:
    """(3값 판정, 모델 원문). 평가 스크립트가 파싱 실패까지 세도록 분리해 둔다."""
    chain = GRADE_PROMPT | _structured_grade_llm()
    result = chain.invoke({"question": question, "context": context_text(docs)})
    raw_msg = result.get("raw")
    raw = raw_msg.content if raw_msg is not None else ""
    parsed = result.get("parsed")
    if parsed is None:
        err = result.get("parsing_error")
        return GRADE_UNPARSED, raw or (str(err) if err else "")
    return (GRADE_YES if parsed.relevant else GRADE_NO), raw


def grade(state: AgentState) -> dict:
    verdict, raw = judge_relevance(
        state["question"],
        context_docs(state["documents"], state.get("strategy")))
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


HYDE_PROMPT = ChatPromptTemplate.from_template(
    "다음 질문에 대한 답변이 실릴 법한 문서 단락을 한국어 2~3문장으로 "
    "작성하세요. 사실 여부는 중요하지 않습니다 — 검색용 가상 단락입니다. "
    "단락만 출력하세요.\n\n질문: {question}"
)

# 가상 단락이 이보다 길면 자른다 — BM25 질의가 길어질수록 노이즈 토큰이
# 늘고, 임베딩도 앞부분이 지배적이다. 3B가 지시(2~3문장)를 넘겨 장황하게
# 쓰는 경우의 안전판.
_MAX_HYDE_CHARS = 500

_hyde_cache: dict[tuple[str, str], str | None] = {}


def clean_hypothetical(raw: str) -> str | None:
    """HyDE 가상 단락 출력 정리. 쓸 수 없으면 None(질의만으로 검색).

    clean_rewrite와 같은 이유로 존재한다 — 3B는 "다음과 같습니다:" 류
    안내문과 접두어를 붙인다. 다른 점 둘: (1) 단락이 목적이라 첫 줄만
    취하지 않고 남은 줄을 전부 합친다, (2) 원 질의와 같아도 실패가
    아니다 — 질의를 대체하는 게 아니라 순위 목록을 추가할 뿐이라
    같은 텍스트면 결과도 같은 순위일 뿐 해가 없다.
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not _PREAMBLE_LINE.search(ln)]
    text = " ".join(_REWRITE_PREFIX.sub("", ln) for ln in lines)
    text = text.strip().strip("\"'`“”‘’")
    if len(text) < 2:
        return None
    return text[:_MAX_HYDE_CHARS]


def hyde_term_query(query: str, hypo: str) -> str | None:
    """가상 단락에서 **질의에 없는 영숫자 토큰**만 추린 BM25 확장 질의.

    어휘 격차의 실체는 "프레임워크"(질의)와 "Tensorflow"(문서)처럼
    한국어 일반명사와 영문 기술 명칭 사이의 골이다 — 다리가 되는 토큰은
    가상 단락 속 영숫자 용어이고, 나머지(세그멘테이션·모델·구현 같은
    질의 중복·일반 어휘)는 1차 실측에서 그 다리를 희석시킨 주범이었다.
    질의에 이미 있는 토큰은 원래 목록 2개가 처리하므로 중복시키지 않는다.

    한국어 동의어 격차("회사"↔"기업")는 이 필터가 못 다룬다 — 영숫자만
    남기는 대신 틀린 추측의 폭발 반경을 좁힌 트레이드오프다.
    """
    q_tokens = set(bm25_tokenize(query))
    novel = [t for t in dict.fromkeys(bm25_tokenize(hypo))    # 순서 보존 중복 제거
             if t not in q_tokens and t.isascii() and t.isalnum()]
    return " ".join(novel) if novel else None


def hypothetical_doc(query: str) -> str | None:
    """질의에 대한 가상 답변 단락. 같은 (모델, 질의)는 재사용한다 —
    temperature 0이라 같은 출력이 나올 자리에 LLM을 다시 부를 이유가
    없고, corrective 루프의 재검색이나 평가 반복에서 호출이 겹친다."""
    key = (config.LLM_MODEL, query)
    if key not in _hyde_cache:
        raw = (HYDE_PROMPT | get_llm()).invoke({"question": query}).content
        _hyde_cache[key] = clean_hypothetical(raw)
    return _hyde_cache[key]


_BASE_RULES = (
    "당신은 AI 엔지니어 이윤선의 포트폴리오를 안내하는 어시스턴트입니다.\n"
    "아래 문서를 주의 깊게 읽고, 관련 내용이 있으면 그것을 근거로 한국어로 답하세요.\n"
    "참고: 항목 옆 (YYYY.MM ~ YYYY.MM) 또는 (YYYY.MM~) 표기는 그 항목의 "
    "수행 기간이며 왼쪽 날짜가 시작 시점입니다.\n"
    "관련 정보가 정말로 전혀 없을 때만 '문서에서 찾을 수 없습니다'라고 답하고, "
    "문서에 없는 내용을 지어내지 마세요.\n"
    # 위 줄의 "한국어로 답하세요"만으로는 안 잡힌다. 100문항 실측에서 답변
    # 10개에 한자가 섞였고 그중 3개는 내용이 맞는데 숫자를 중국어로 써서
    # 틀렸다(共有2项 / 论文有1篇 / 共计1年1个月). 이탈 방식이 일정하다 —
    # 한글 단어 중간에 한자 한 글자가 나오고(특許·언语) 거기서부터 문장
    # 전체가 중국어로 넘어간다. temperature 0.0이라 재생성해도 같은 자리에서
    # 똑같이 이탈하고, 온도를 0.5·0.8로 올려도 5문항 중 4개가 재발했다.
    # 글자 단위를 직접 금지하는 이 한 줄만 다섯 문항 전부를 되돌렸다.
    "답변은 한자를 쓰지 말고 한글과 숫자로만 작성하세요.\n"
)

# 실측된 두 실패 유형을 겨냥한 추가 지시.
#
#  열거: 정답 청크가 "Kubernetes, Docker, Helm, GitLab CI/CD, Jenkins,
#        ArgoCD, Prometheus/..." 처럼 7개 목록인데, 모델이 CI/CD처럼 보이는
#        **첫 항목(GitLab CI/CD)만 집고 멈춘다.** 여러 실행에서 같은 방식으로
#        틀렸다.
#  집계: 코퍼스에 "논문 7편(제1저자)"이 1회, "논문 2편 게재(제1저자)"가 2회
#        나온다(후자는 개별 연구의 편수다). 모델이 다수인 쪽을 집어 "2편"이라
#        답한다. 질문이 묻는 범위를 먼저 확인하게 만드는 것이 목표다.
#
# 프롬프트를 늘리는 건 위험하다 — 이 프로젝트는 시스템 프롬프트를 늘렸다가
# 3B의 tool calling이 죽는 것을 이미 겪었다. 그래서 켜고 끌 수 있게 두고 잰다.
_TARGETED_RULES = (
    "여러 항목을 묻는 질문(어떤 도구들, 무엇무엇)이면 문서에서 해당하는 항목을 "
    "빠짐없이 나열하세요. 목록의 첫 항목만 답하지 마세요.\n"
    "같은 대상에 대한 숫자가 문서에 여러 개 보이면, 질문이 묻는 범위(전체 합계인지 "
    "개별 프로젝트인지)를 먼저 확인하고 그 범위에 맞는 숫자를 쓰세요.\n"
)

_TAIL = "\n문서:\n{context}\n\n질문: {question}\n\n답변:"

_PROMPTS = {
    "base": ChatPromptTemplate.from_template(_BASE_RULES + _TAIL),
    "targeted": ChatPromptTemplate.from_template(
        _BASE_RULES + _TARGETED_RULES + _TAIL),
}


def generate_prompt() -> ChatPromptTemplate:
    """config.GENERATE_PROMPT_VARIANT 에 따라 프롬프트를 고른다."""
    return _PROMPTS[config.GENERATE_PROMPT_VARIANT]


GENERATE_PROMPT = _PROMPTS["base"]        # 하위호환 (기존 참조)


_HANJA = re.compile(r"[一-鿿]")

# 재시도 온도. 0.0이면 같은 자리에서 똑같이 이탈하므로(실측) 재생성 자체가
# 무의미하다 — 이탈을 깨려면 샘플링이 필요하다. 다만 온도만 올리는 것도
# 안 통해서(t0.5·t0.8에서 5문항 중 4개 재발) 프롬프트의 한자 금지 줄과
# **함께**여야 한다. 둘을 같이 준 조건이 실측에서 5/5 깨끗했다.
_RETRY_TEMPERATURE = 0.5


def _drop_hanja(answer: str, question: str, context: str) -> str:
    """답변에 한자가 섞이면 1회만 다시 생성한다.

    프롬프트 지시로 100문항 중 한자 답변이 10 → 6으로 줄었지만 남은 6건은
    지시를 무시한다. 전부 채점은 통과하므로 **정답률로는 안 보이는 결함**이다
    — 한국어 포트폴리오 안내가 "특許是文档中没..."라고 답하는 것이라 사용자
    눈에는 먼저 띈다.

    재시도도 한자면 원본을 쓴다. 코퍼스에 한자가 있는 청크가 하나 있어
    (resume.md의 古語) 정당한 인용까지 지우지 않기 위해서다.
    """
    if not _HANJA.search(answer):
        return answer
    retry = (generate_prompt() | get_llm(_RETRY_TEMPERATURE)).invoke(
        {"question": question, "context": context}).content.strip()
    return retry if not _HANJA.search(retry) else answer


def generate(state: AgentState) -> dict:
    # 소형 모델은 긴 컨텍스트에서 근거를 놓치기 쉬우므로 상위 N개만 쓰고
    # (N은 grade와 공유 — context_docs), 끝부분 주의집중이 강한 특성에 맞춰
    # 랭크 역순으로 배치해 최상위 청크가 질문 바로 앞에 오게 한다
    used = context_docs(state["documents"], state.get("strategy"))
    context = context_text(order_for_prompt(used))
    chain = generate_prompt() | get_llm()
    answer = chain.invoke(
        {"question": state["question"], "context": context}
    ).content.strip()
    answer = _drop_hanja(answer, state["question"], context)
    # 출처는 **실제로 프롬프트에 들어간 청크**에서만 뽑는다. 예전에는 검색된
    # TOP_K(6개) 전부에서 뽑아, 답변 생성에 쓰이지도 않은 문서가 근거로
    # 표시됐다 — 개수는 context_docs로 맞춰 놓고 인용은 안 맞춘 상태였다.
    sources = sorted({d.metadata.get("source", "?") for d in used})
    # 근거 텍스트를 그대로 남긴다 — 사후에 재현하려면 재작성된 질의까지
    # 알아야 해서(temperature 0.3이라 재현 불가) 여기서 저장해야 한다.
    return {"answer": answer, "sources": sources,
            "contexts": [d.page_content for d in used]}


class GroundednessVerdict(BaseModel):
    """generate가 방금 쓴 답변이 근거 문서에 실제로 있는 내용에만
    기반하는지. GENERATE_PROMPT가 "지어내지 마세요"라고 지시는 하지만,
    지시를 따랐는지 확인하는 단계가 지금까지 없었다 — grade가 검색 쪽의
    fail-open을 조용히 흡수하지 않도록 3값으로 바꾼 것과 같은 이유로,
    생성 쪽에도 관측 지점을 하나 놓는다.
    """
    grounded: bool = Field(
        description="답변의 핵심 주장이 전부 문서에서 확인되면 true, "
                    "문서에 없는 내용을 지어냈으면 false")
    unsupported_claim: str = Field(
        default="",
        description="grounded가 false일 때만: 문서에서 확인 안 되는 주장을 "
                    "한 문장으로. grounded가 true면 빈 문자열")


VERIFY_PROMPT = ChatPromptTemplate.from_template(
    "아래 문서를 근거로 답변이 생성되었습니다. 답변의 핵심 주장이 문서 "
    "내용과 일치하는지, 문서에 없는 수치나 사실을 지어내지 않았는지 "
    "확인하세요.\n\n문서:\n{context}\n\n답변: {answer}"
)

_structured_verify_cache: dict[str, object] = {}


def _structured_verify_llm():
    model = config.LLM_MODEL
    if model not in _structured_verify_cache:
        _structured_verify_cache[model] = get_llm().with_structured_output(
            GroundednessVerdict, include_raw=True)
    return _structured_verify_cache[model]


def verify(state: AgentState) -> dict:
    """답변이 근거에 실제로 기반하는지 사후 확인한다.

    generate 직후에 붙는 관측 노드다 — 판정 실패든(unparsed) 아니든
    **답변 자체는 절대 안 바꾼다**(fail-open, grade·rewrite와 같은 정책).
    지금은 결과를 State에 기록만 하고 라우팅을 바꾸지 않는다. 재생성
    루프를 붙이는 건(예: grounded=False면 generate를 다시) 다음 단계고,
    A/B로 값을 하는지부터 재는 게 먼저다 — 이 프로젝트가 반복해서 배운
    "손잡이를 켜기 전에 잰다" 원칙.
    """
    if not config.VERIFY_GROUNDING:
        return {"grounded": None, "unsupported_claim": None}

    docs = context_docs(state["documents"], state.get("strategy"))
    chain = VERIFY_PROMPT | _structured_verify_llm()
    result = chain.invoke(
        {"context": context_text(docs), "answer": state["answer"]})
    parsed = result.get("parsed")
    if parsed is None:
        print(f"[verify] 판정 파싱 실패 — 기록만 하고 통과: "
              f"{(result.get('raw').content if result.get('raw') else '')[:60]!r}")
        return {"grounded": None, "unsupported_claim": None}
    return {"grounded": parsed.grounded,
            "unsupported_claim": parsed.unsupported_claim or None}


# ── Graph ──────────────────────────────────────────────

def route_strategy(state: AgentState) -> dict:
    """질문 유형에 맞는 컨텍스트 전략을 State에 싣는다 (그래프 진입점).

    예전엔 run()이 컨텍스트 매니저로 **전역 config를 잠깐 바꿨다가 되돌리는**
    방식이었다. 세 가지가 걸렸다:

      · FastAPI는 동기 핸들러를 스레드풀에서 돌린다. 두 요청이 동시에 들어오면
        전역을 서로 덮어써서, aggregation 질문이 켠 GENERATE_TOP_N=3을 옆
        스레드의 fact 질문이 읽는다. 코드 주석에 "알려진 한계"로 적혀 있었다.
      · 스트리밍 경로는 제너레이터라 run()을 못 쓰고 api.py가 컨텍스트 매니저를
        직접 열어야 했다 — 라우팅이 붙는 자리가 결국 둘로 갈렸다.
      · 라우팅이 그래프 밖에 있어 LangGraph 추적·스트리밍에 안 보였다.

    전략을 State로 나르면 셋 다 사라진다. 전역을 안 건드리니 스레드 안전하고,
    스트리밍도 그냥 그래프를 돌리면 되고, 노드라서 추적에 잡힌다.
    """
    if not config.TYPE_ROUTING:
        return {"strategy": {}}

    import route

    return {"strategy": route.ROUTES.get(
        route.classify_question_type(state["question"]), {})}


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
    g.add_node("route_strategy", timed(route_strategy))
    g.add_node("retrieve", timed(retrieve))
    g.add_node("grade", timed(grade))
    g.add_node("rewrite", timed(rewrite))
    g.add_node("generate", timed(generate))
    g.add_node("verify", timed(verify))

    g.add_edge(START, "route_strategy")
    g.add_edge("route_strategy", "retrieve")
    g.add_conditional_edges("retrieve", needs_grading, ["grade", "generate"])
    g.add_conditional_edges("grade", decide_next, ["rewrite", "generate"])
    g.add_conditional_edges("rewrite", after_rewrite, ["retrieve", "generate"])
    # verify는 항상 거친다 — config.VERIFY_GROUNDING이 꺼져 있으면 verify()가
    # 즉시 {"grounded": None}만 반환하고 빠진다(LLM 호출 없음). 그래프
    # 구조 자체를 조건부로 만들지 않은 이유는 stream_mode=["messages",...]가
    # "generate" 다음 노드가 항상 있다고 가정하면 스트리밍 이벤트 필터링
    # (api.py의 langgraph_node == "generate")이 더 단순해지기 때문이다.
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile()


def run(question: str, graph=None) -> dict:
    """질의 1건을 **라우팅을 적용해** 실행하고 State 전체를 반환한다.

    이 함수가 생긴 이유 — 예전엔 `_type_routing`을 `ask()` 안에서만 걸었는데,
    정작 실사용·평가 경로는 `graph.invoke()`를 직접 불러서 **라우팅을 통째로
    건너뛰고 있었다**(api.py·cli.py·eval/evaluate.py). 즉 `TYPE_ROUTING=1`을
    켜도 서빙에는 반영되지 않았고, evaluate.py로 잰 "라우팅 켠 전후" 수치는
    같은 코드 경로를 두 번 잰 것이라 애초에 차이가 나올 수 없었다(실측으로
    확인: aggregation 질문 실행 중 관측된 (NEIGHBOR_WINDOW, GENERATE_TOP_N)이
    invoke 직접 호출은 (0,5), ask()는 (1,3)).

    그래서 **진입점을 하나로 모은다**. 질의를 실행하는 코드는 전부 이 함수를
    거치게 해서, 라우팅이 붙는 자리가 한 군데만 존재하도록 한다.

    graph를 넘기면 그걸 쓴다 — api.py는 기동 시 1회만 컴파일해 두고 재사용한다.
    """
    g = graph if graph is not None else build_graph()
    return g.invoke({"question": question, "query": question, "rewrites": 0})


def uses_team(question: str) -> bool:
    """이 질의를 멀티에이전트 팀으로 보낼 것인가 (config.TEAM_ROUTING 게이트).

    route를 지연 임포트하는 이유는 _type_routing과 같다 — team.py가
    graph.py를 임포트하므로 모듈 최상위에서 부르면 순환이 된다.
    """
    if not config.TEAM_ROUTING:
        return False
    import route
    return route.should_use_team(question)


def ask(question: str) -> dict:
    """질의 1건에 답한다 — 서빙·평가가 공유하는 단일 진입점.

    TEAM_ROUTING이 켜져 있고 멀티홉 질문이면 team.py로 보낸다. 팀 경로는
    하위 답변을 종합하므로 rewrites 개념이 없어 0으로 채운다(질의 재작성은
    worker 안에서 별도로 일어난다). 어느 경로를 탔는지는 `route` 키로
    돌려준다 — 서빙 로그·평가에서 구분할 수 있어야 한다.
    """
    if uses_team(question):
        import team
        r = team.ask_team(question)
        return {"answer": r["answer"], "sources": r["sources"],
                "rewrites": 0, "route": "team",
                "sub_questions": r["sub_questions"]}

    result = run(question)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "rewrites": result["rewrites"],
        "route": "single",
    }


