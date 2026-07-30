"""결정적 로직 단위 테스트 — LLM/인덱스 없이 CI에서 돈다."""
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from evaluate import is_pass  # noqa: E402
from team import _parse_sub_questions  # noqa: E402


# ── 채점기 v2 ──

CASE_PAPERS = {"expected_keywords": ["7"], "answer_patterns": ["7\\s*편"]}


def test_refusal_always_fails():
    case = {"expected_keywords": ["2292"], "answer_patterns": ["2292"]}
    assert not is_pass("문서에서 찾을 수 없습니다. 2292라는 값은...", case)


def test_bare_number_no_longer_passes():
    # v1 키워드 채점의 허위 통과 사례: '7'이 다른 숫자에 우연히 포함
    assert not is_pass("2017년부터 연구를 시작했습니다.", CASE_PAPERS)


def test_number_with_unit_passes():
    assert is_pass("제1저자 논문은 총 7편입니다.", CASE_PAPERS)
    assert is_pass("논문 7 편을 게재했습니다.", CASE_PAPERS)


def test_all_patterns_required():
    case = {"expected_keywords": ["Jenkins", "ArgoCD"],
            "answer_patterns": ["Jenkins", "ArgoCD"]}
    assert not is_pass("Jenkins를 사용했습니다.", case)
    assert is_pass("Jenkins와 ArgoCD를 사용했습니다.", case)


def test_keyword_fallback_when_no_patterns():
    case = {"expected_keywords": ["Pyannote"]}
    assert is_pass("pyannote 모델을 파인튜닝했습니다.", case)


# ── planner 출력 파싱 ──

def test_parse_valid_json_array():
    out = _parse_sub_questions(
        '생각해보면... ["A는 언제 시작했어?", "B는 언제 시작했어?"]', "원 질문")
    assert out == ["A는 언제 시작했어?", "B는 언제 시작했어?"]


def test_parse_caps_at_three():
    out = _parse_sub_questions('["q1", "q2", "q3", "q4"]', "원 질문")
    assert len(out) == 3


def test_parse_garbage_falls_back():
    assert _parse_sub_questions("JSON이 아닌 텍스트", "원 질문") == ["원 질문"]
    assert _parse_sub_questions("[1, 2, 3]", "원 질문") == ["원 질문"]


def test_parse_broken_json_falls_back():
    assert _parse_sub_questions('["미완성 배열', "원 질문") == ["원 질문"]


# ── 인제스트 전처리 ──

def test_clean_markdown_color_macro():
    from ingest import clean_markdown
    src = "### $\\color{3399ff}{About}$ $\\color{3399ff}{Me}$\n\n경력 요약"
    assert clean_markdown(src) == "### About Me\n\n경력 요약"


def test_clean_markdown_strips_images():
    from ingest import clean_markdown
    assert clean_markdown("앞 ![img](a%20b.png) 뒤") == "앞  뒤"


# ── 검수(inspect_data.py)로 뒤늦게 찾은 노이즈들 ──

def test_notion_internal_link_keeps_text_drops_encoded_path():
    """노션 내부 링크는 텍스트만 남는다 — 코퍼스의 9.4%를 먹던 노이즈."""
    from ingest import clean_markdown
    src = "①   [Experience](%EC%9D%B4%EC%9C%A4%EC%84%A0%20%ED%8F%AC.md)"
    assert clean_markdown(src) == "①   Experience"


def test_external_url_is_preserved():
    """실제 URL은 남긴다 — '깃허브 주소' 같은 질문에 답해야 한다."""
    from ingest import clean_markdown
    src = "[git](https://github.com/ckc5800)"
    assert clean_markdown(src) == src


def test_external_url_with_encoding_is_preserved():
    """논문 링크처럼 http URL 안의 퍼센트 인코딩은 정상이므로 건드리지 않는다."""
    from ingest import clean_markdown
    src = "[논문](https://kiss.kstudy.com/?articleTitle=GAN%EC%9D%84+%ED%99%9C)"
    assert clean_markdown(src) == src


def test_nested_link_leftover_is_removed():
    """[[A](url)(kisti) ](내부경로) 같은 중첩 링크의 잔여 경로까지 지운다."""
    from ingest import clean_markdown
    src = "[[한국](https://namu.wiki/w/%ED%95%9C)(kisti) ](%ED%95%9C%EA%B5%AD%2020a.md)"
    out = clean_markdown(src)
    assert "%ED%95%9C%EA%B5%AD%2020a.md" not in out   # 내부 경로는 사라지고
    assert "https://namu.wiki/w/%ED%95%9C" in out     # 외부 URL은 남는다


def test_html_tags_are_stripped():
    from ingest import clean_markdown
    src = '<aside>\n<img src="notion.png" alt="notion.png" width="40px" /> notion'
    out = clean_markdown(src)
    assert "<aside>" not in out and "<img" not in out
    assert "notion" in out


# ── grade/generate 청크 수 불일치 회귀 테스트 ──
#
# grade가 6개를 보고 generate가 3개만 보면, 정답이 4~6등일 때 grade는
# "충분하다"고 통과시키지만 generate는 그 근거를 못 받아 거부한다
# ("근무한 회사들" 질문이 실제로 이 상태였다 — README v6). context_docs()로
# 둘이 같은 개수를 보도록 묶었으니, 그 계약이 깨지지 않는지 고정한다.

def test_context_docs_caps_at_generate_top_n():
    from graph import GENERATE_TOP_N, context_docs

    docs = [f"doc{i}" for i in range(6)]
    assert len(context_docs(docs)) == GENERATE_TOP_N


def test_context_docs_keeps_top_ranked_first():
    from graph import context_docs

    docs = ["rank1", "rank2", "rank3", "rank4", "rank5", "rank6"]
    # grade는 이 순서 그대로 판정(순위 유지). generate만 별도로 뒤집는다.
    assert context_docs(docs) == ["rank1", "rank2", "rank3"]


def test_context_docs_shorter_than_top_n_passes_through():
    from graph import context_docs

    assert context_docs(["only"]) == ["only"]


# ── 다이어그램 분리 회귀 테스트 ──
#
# 청크 평균이 594자로 산문(545자)보다 커 보이는 원인을 추적하니 ASCII
# 아키텍처 다이어그램(```markdown 펜스 안의 박스 그림)이었다 — 800자 상한이
# 도중에 그어져 박스가 반토막 나는 청크가 나왔다. 다이어그램은 문장 단위가
# 없어 문자 기준 분할이 애초에 안 맞으므로 통짜로 뗀다.

def test_diagram_fence_is_extracted():
    from ingest import extract_diagrams

    fence = "```markdown\n" + "┌─┐\n" * 20 + "```"     # 박스문자 밀도 高
    text = f"앞 문단\n\n{fence}\n\n뒷 문단"
    body, diagrams = extract_diagrams(text)

    assert len(diagrams) == 1
    assert diagrams[0].page_content == fence
    assert fence not in body
    assert "[다이어그램: 1]" in body


def test_non_diagram_fence_is_kept_inline():
    """API 예시처럼 박스문자가 없는 짧은 펜스는 본문에 남는다."""
    from ingest import extract_diagrams

    text = "앞 문단\n\n```\nPOST /api/v2/tts-engine/synthesize/sse\n```\n\n뒷 문단"
    body, diagrams = extract_diagrams(text)

    assert diagrams == []
    assert "POST /api/v2/tts-engine/synthesize/sse" in body


def test_diagram_extraction_does_not_leave_extra_blank_lines():
    from ingest import extract_diagrams

    fence = "```markdown\n" + "┌─┐\n" * 20 + "```"
    text = f"앞 문단\n\n{fence}\n\n뒷 문단"
    body, _ = extract_diagrams(text)

    assert "\n\n\n" not in body


# ── BM25 한국어 bigram 토크나이저 ──

def test_bm25_bigram_overlap_with_josa():
    from graph import bm25_tokenize
    # '회사들'(질의)과 '회사'(문서)가 bigram '회사'로 겹친다
    assert "회사" in bm25_tokenize("회사들")
    assert "회사" in bm25_tokenize("회사")


def test_bm25_keeps_full_tokens_and_ascii():
    from graph import bm25_tokenize
    tokens = bm25_tokenize("Jenkins 파이프라인")
    assert "jenkins" in tokens          # 영문은 소문자 어절 그대로
    assert "파이프라인" in tokens        # 원 어절 유지
    assert "파이" in tokens             # bigram 추가


# ── grade/generate 컨텍스트 '길이' 불일치 회귀 테스트 ──
#
# 개수(GENERATE_TOP_N)는 context_docs로 맞췄지만 길이는 안 맞아 있었다:
# grade만 page_content[:500]으로 잘라 봤다. 청크 중앙값이 711자, 다이어그램
# 청크는 4천 자가 넘어서 grade가 근거의 상당 부분을 못 보는 상태였다.

def test_grade_and_generate_see_the_same_text():
    from langchain_core.documents import Document

    from graph import context_docs, context_text

    docs = [Document(page_content="가" * 900, metadata={"source": "a.md"}),
            Document(page_content="나" * 900, metadata={"source": "b.md"})]
    text = context_text(context_docs(docs))
    for d in docs:
        assert d.page_content in text     # 잘리지 않는다


def test_context_text_joins_with_separator():
    from langchain_core.documents import Document

    from graph import context_text

    docs = [Document(page_content="A", metadata={}),
            Document(page_content="B", metadata={})]
    assert context_text(docs) == "A\n---\nB"


# ── 출처 과대보고 회귀 테스트 ──
#
# generate는 top-3만 프롬프트에 넣는데 sources는 검색된 TOP_K(6개) 전부에서
# 뽑고 있었다 — 답변 근거가 아닌 문서가 출처로 나갔다.

def test_sources_come_only_from_the_docs_in_the_prompt(monkeypatch):
    from types import SimpleNamespace

    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda

    import graph

    seen = {}

    def fake_llm(prompt_value):
        seen["context"] = prompt_value.to_string()
        return SimpleNamespace(content="답변")

    monkeypatch.setattr(graph, "get_llm",
                        lambda *a, **k: RunnableLambda(fake_llm))

    docs = [Document(page_content=f"본문{i}", metadata={"source": f"{i}.md"})
            for i in range(6)]
    out = graph.generate({"question": "질문", "documents": docs})

    assert out["sources"] == ["0.md", "1.md", "2.md"]      # 4~6위는 빠진다
    assert "본문3" not in seen["context"]                   # 프롬프트에도 없다


# ── 인덱스 지연 초기화 레이스 회귀 테스트 ──
#
# _vectorstore를 먼저 대입하고 그 뒤에 BM25를 만들면, 그 사이에 들어온
# 스레드가 "준비됐다"고 보고 아직 None인 _bm25를 받아 간다. FastAPI의 동기
# 핸들러는 스레드풀에서 돌기 때문에 동시 요청 2건이면 재현된다.

def test_load_indexes_is_thread_safe(monkeypatch):
    from langchain_community import retrievers

    import graph
    import vectorstore

    class SlowBM25:
        k = 0

        @classmethod
        def from_documents(cls, chunks, preprocess_func=None):
            time.sleep(0.2)          # 인덱스 구축이 오래 걸리는 상황
            return cls()

    def slow_load(*a, **k):
        time.sleep(0.05)
        return "STORE"

    monkeypatch.setattr(vectorstore, "load", slow_load)
    monkeypatch.setattr(retrievers, "BM25Retriever", SlowBM25)

    graph._vectorstore = graph._bm25 = None
    try:
        errors, results = [], []

        def worker():
            try:
                results.append(graph._load_indexes())
            except Exception as e:                        # noqa: BLE001
                errors.append(e)

        # 첫 요청을 먼저 띄우고, **BM25를 만드는 중간에** 나머지를 들여보낸다.
        # 동시에 출발시키면 전부 초기화 전에 통과해 버려서 레이스가 안 난다 —
        # 위험 구간은 '_vectorstore는 세팅됐고 _bm25는 아직'인 그 사이다.
        first = threading.Thread(target=worker)
        first.start()
        time.sleep(0.1)                  # load(0.05s) 끝, BM25(0.2s) 진행 중

        rest = [threading.Thread(target=worker) for _ in range(7)]
        for t in rest:
            t.start()
        for t in [first, *rest]:
            t.join(timeout=10)

        assert not errors
        assert len(results) == 8
        for store, bm25 in results:
            assert store == "STORE"
            assert bm25 is not None      # 옛 코드는 여기서 None을 받았다
    finally:
        graph._vectorstore = graph._bm25 = None


# ── RRF 중복 제거: 출처가 검색 순서에 따라 흔들리지 않는다 ──

def test_rrf_dedup_keeps_first_seen_source(monkeypatch):
    from langchain_core.documents import Document

    import graph

    same = "같은 사실이 두 문서에 중복돼 있다"
    vec = [Document(page_content=same, metadata={"source": "resume.md"})]
    kw = [Document(page_content=same, metadata={"source": "portfolio.md"})]

    monkeypatch.setattr(graph, "_load_indexes", lambda: (
        SimpleStore(vec), SimpleRetriever(kw)))

    docs = graph.hybrid_search("질의")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "resume.md"   # 벡터 검색 결과가 먼저


class SimpleStore:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k):
        return self.docs[:k]


class SimpleRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, query):
        return self.docs


# ── vectorstore.search: kind 인자가 전역 config를 이긴다 ──

def test_search_kind_argument_wins_over_global_config(monkeypatch):
    import config
    import vectorstore

    monkeypatch.setattr(config, "VECTOR_STORE", "qdrant")   # 전역은 qdrant

    calls = {}

    class Store:
        def similarity_search(self, query, k, **kw):
            calls["k"], calls["kw"] = k, kw
            return [FakeDoc("a.md"), FakeDoc("b.md"), FakeDoc("a.md")]

    out = vectorstore.search(Store(), "질의", k=2, source="a.md", kind="faiss")

    # faiss 경로 = 넉넉히 뽑아 사후 필터링. 필터를 검색에 주입하지 않는다.
    assert calls["k"] == 20 and "filter" not in calls["kw"]
    assert [d.metadata["source"] for d in out] == ["a.md", "a.md"]


class FakeDoc:
    def __init__(self, source):
        self.metadata = {"source": source}
        self.page_content = source


# ── calculate: AST 화이트리스트가 막지 못하는 자원 고갈 ──

def test_calculate_rejects_huge_exponent():
    from tools import calculate

    out = calculate.invoke({"expression": "9**9**9**9"})
    assert "오류" in out          # 계산을 시도하지 않고 거부


def test_calculate_still_does_normal_math():
    from tools import calculate

    assert calculate.invoke({"expression": "(2292-334)/2292*100"}) == "85.4276"
    assert calculate.invoke({"expression": "2**10"}) == "1024"


# ── 채점 기준이 없는 케이스를 조용히 통과시키지 않는다 ──

def test_case_without_any_criterion_raises():
    with pytest.raises(ValueError):
        is_pass("아무 답변", {"question": "기준 없는 케이스"})


# ── grade 판정 파싱 (3값) ──
#
# 예전 파서는 yes/no 두 값이라 **파싱 실패가 sufficient에 흡수**됐다.
# fail-open 정책은 유지하되, 실패를 별도 값으로 세는 것이 핵심이다.

def test_verdict_plain_answers():
    from graph import GRADE_NO, GRADE_YES, parse_verdict
    assert parse_verdict("yes") == GRADE_YES
    assert parse_verdict("no") == GRADE_NO
    assert parse_verdict("YES\n") == GRADE_YES


def test_verdict_with_punctuation():
    """'no.' 'no,' — 옛 파서는 startswith 특례로 겨우 잡았다."""
    from graph import GRADE_NO, parse_verdict
    assert parse_verdict("no.") == GRADE_NO
    assert parse_verdict("no, 관련 내용이 없습니다") == GRADE_NO
    assert parse_verdict("The documents contain no relevant info") == GRADE_NO


def test_verdict_korean_answers():
    from graph import GRADE_NO, GRADE_YES, parse_verdict
    for neg in ("아니오", "아니요", "아니다", "아니에요", "아님"):
        assert parse_verdict(neg) == GRADE_NO
    for pos in ("예", "네"):
        assert parse_verdict(pos) == GRADE_YES
    assert parse_verdict("관련 정보가 있습니다") == GRADE_YES
    assert parse_verdict("관련 정보가 없습니다") == GRADE_NO


def test_verdict_catches_anipnida():
    """'아닙니다'는 아+닙+니+다라서 '아니' 부분매칭으로 안 잡힌다.

    옛 파서는 `"아니" in verdict`였으므로 이 흔한 부정형을 **놓쳤다** —
    부정인데 sufficient로 통과시키던 케이스다.
    """
    from graph import GRADE_NO, parse_verdict
    assert "아니" not in "아닙니다"          # 옛 파서가 못 잡은 이유
    assert parse_verdict("아닙니다") == GRADE_NO


def test_verdict_animyeon_is_not_a_negative():
    """'아니면'은 부정이 아니다 — 옛 파서('아니' 부분매칭)의 오탐 케이스."""
    from graph import GRADE_NO, parse_verdict
    out = parse_verdict("문서가 관련 있는지 아니면 판단이 필요한지 보겠습니다")
    assert out != GRADE_NO          # 불필요한 재검색으로 가지 않는다


def test_verdict_lead_token_wins_over_trailing_prose():
    """판정어가 맨 앞에 있으면 뒤에 붙는 사족은 무시한다."""
    from graph import GRADE_YES, parse_verdict
    assert parse_verdict("yes. " + "다만 관련 없는 부분도 " * 20) == GRADE_YES
    # 지시를 어기고 둘 다 말해도, 앞에 나온 판정을 취한다
    # (fail-open이라 yes/unparsed는 어차피 같은 경로지만 계측이 정확해진다)
    assert parse_verdict("yes and no") == GRADE_YES


def test_verdict_garbage_is_unparsed():
    from graph import GRADE_UNPARSED, parse_verdict
    assert parse_verdict("") == GRADE_UNPARSED
    assert parse_verdict("음... 잘 모르겠습니다") == GRADE_UNPARSED
    # 서술문에서 긍정·부정이 동시에 잡히면 판정 실패로 센다
    assert parse_verdict(
        "관련 정보가 있습니다. 그런데 관련 정보가 없습니다") == GRADE_UNPARSED


def test_grade_node_fails_open_on_unparsed(monkeypatch):
    """파싱 실패는 sufficient로 통과시킨다 (정책 유지) — 단 조용히는 아니다."""
    import graph

    monkeypatch.setattr(graph, "judge_relevance",
                        lambda q, d: (graph.GRADE_UNPARSED, "음..."))
    out = graph.grade({"question": "q", "documents": []})
    assert out["grade"] == "sufficient"


def test_grade_node_routes_explicit_negative(monkeypatch):
    import graph

    monkeypatch.setattr(graph, "judge_relevance",
                        lambda q, d: (graph.GRADE_NO, "no"))
    out = graph.grade({"question": "q", "documents": []})
    assert out["grade"] == "insufficient"


# ── rewrite 출력 가드 ──
#
# 예전에는 LLM 출력 전체가 그대로 검색 질의가 됐다 — 접두어·설명까지
# BM25 토큰으로 들어갔고, 빈 출력에 대한 가드도 없었다.

def test_clean_rewrite_strips_prefix_and_quotes():
    from graph import clean_rewrite
    assert clean_rewrite('재작성된 질문: "TTS TTFB 개선 수치"', "원 질문") \
        == "TTS TTFB 개선 수치"
    assert clean_rewrite("질문: ArgoCD CI/CD 도구", "원 질문") == "ArgoCD CI/CD 도구"


def test_clean_rewrite_takes_first_line_only():
    from graph import clean_rewrite
    raw = "TTFB 개선 수치\n\n설명: 이 질문은 검색이 잘 되도록..."
    assert clean_rewrite(raw, "원 질문") == "TTFB 개선 수치"


def test_clean_rewrite_rejects_empty_and_identical():
    from graph import clean_rewrite
    assert clean_rewrite("", "원 질문") is None
    assert clean_rewrite("   \n  ", "원 질문") is None
    assert clean_rewrite("원 질문", "원 질문") is None      # 같은 질의 = 같은 결과


def test_clean_rewrite_caps_length():
    from graph import clean_rewrite
    out = clean_rewrite("가" * 500, "원 질문")
    assert len(out) == 200


def test_rewrite_marks_failure_and_skips_retrieval(monkeypatch):
    from types import SimpleNamespace

    from langchain_core.runnables import RunnableLambda

    import graph

    monkeypatch.setattr(graph, "get_llm", lambda *a, **k: RunnableLambda(
        lambda _: SimpleNamespace(content="   ")))     # 빈 재작성
    out = graph.rewrite({"question": "원 질문", "rewrites": 0})

    assert out["rewrite_failed"] is True
    assert "query" not in out                  # 질의를 갈아끼우지 않는다
    assert out["rewrites"] == 1
    assert graph.after_rewrite(out) == "generate"      # 재검색 생략


def test_after_rewrite_retries_on_success():
    from graph import after_rewrite
    assert after_rewrite({"rewrite_failed": False}) == "retrieve"
    assert after_rewrite({}) == "retrieve"             # 키가 없으면 정상 경로


# ── 무의미한 grade 호출 제거 ──

def test_needs_grading_skips_when_no_rewrite_budget(monkeypatch):
    import config
    from graph import needs_grading

    monkeypatch.setattr(config, "MAX_REWRITES", 1)
    assert needs_grading({"rewrites": 0}) == "grade"
    # 재작성 여력이 없으면 판정이 라우팅을 바꿀 수 없다 → 묻지 않는다
    assert needs_grading({"rewrites": 1}) == "generate"

    # team.py의 worker는 rewrites=MAX_REWRITES로 시작한다 (재작성 끔)
    monkeypatch.setattr(config, "MAX_REWRITES", 0)
    assert needs_grading({"rewrites": 0}) == "generate"


def test_graph_compiles_with_the_new_edges():
    from graph import build_graph
    build_graph()          # 조건부 엣지 3개가 유효한지 (컴파일 시 검증된다)


# ── LangGraph가 조용히 버리는 키 방지 ──
#
# 이 프로젝트 첫 버그: grade가 반환하는 키가 AgentState에 선언돼 있지 않아
# LangGraph가 값을 조용히 버렸고, 재작성률이 100%로 나왔다. 원인 추적에
# 며칠이 걸렸다 — 노드가 반환하는 키는 전부 선언돼 있어야 한다.

def test_every_node_output_key_is_declared_in_state():
    from graph import AgentState

    declared = set(AgentState.__annotations__)
    for key in ("question", "query", "documents", "rewrites", "grade",
                "rewrite_failed", "answer", "sources"):
        assert key in declared, f"AgentState에 {key} 선언 누락"
