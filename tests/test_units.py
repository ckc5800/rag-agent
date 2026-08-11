"""결정적 로직 단위 테스트 — LLM/인덱스 없이 CI에서 돈다."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from evaluate import is_pass, normalize_punctuation  # noqa: E402
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


def test_fullwidth_punctuation_normalized_before_grading():
    # qwen이 한국어로 답하면서도 중국어 학습 데이터 흔적으로 전각 마침표(。)를
    # 섞어 낼 때가 있다 — 정규화 없이는 "\\.$" 같은 반각 패턴이 못 잡는다.
    case = {"answer_patterns": ["7\\s*편\\."]}
    assert is_pass("제1저자 논문은 총 7편。", case)


def test_normalize_punctuation_maps_common_cjk_fullwidth():
    assert normalize_punctuation("완료。확인：필요？") == "완료.확인:필요?"
    assert normalize_punctuation("이미 반각.") == "이미 반각."  # 이미 반각이면 그대로


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
    import config
    from graph import context_docs

    docs = [f"doc{i}" for i in range(6)]
    assert len(context_docs(docs)) == config.GENERATE_TOP_N


def test_context_docs_keeps_top_ranked_first():
    import config
    from graph import context_docs

    docs = [f"rank{i}" for i in range(1, 7)]
    n = config.GENERATE_TOP_N
    # 상수값이 아니라 계약을 고정한다 — N은 실측으로 바뀔 수 있다(3 → 5).
    # grade는 이 순서 그대로 판정(순위 유지). generate만 별도로 뒤집는다.
    assert context_docs(docs) == [f"rank{i}" for i in range(1, n + 1)]


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


# ── Parent-Child 청킹 ──
#
# child로 검색·판정하고 generate 직전에만 parent로 확장한다. 같은 parent에
# 속한 child가 여러 개 뽑히면 parent 원문이 프롬프트에 중복으로 안 들어가야
# 한다 — 확장 로직의 핵심은 사실상 이 중복 제거뿐이라 여기를 고정한다.

def test_build_parent_child_links_children_to_correct_parent():
    from ingest_parent_child import build_parent_child
    from langchain_core.documents import Document

    long_text = ("## 섹션\n\n" + "문장입니다. " * 200)  # parent_size(2000) 넘게
    doc = Document(page_content=long_text, metadata={"source": "test.md"})
    parents, children = build_parent_child([doc])

    assert len(parents) >= 1
    assert len(children) >= len(parents)  # parent당 child 1개 이상
    parent_ids = {p.metadata["parent_id"] for p in parents}
    for child in children:
        assert child.metadata["parent_id"] in parent_ids


def test_expand_to_parents_dedupes_same_parent():
    import graph_parent_child as gpc
    from langchain_core.documents import Document

    gpc._parent_by_id = {"p1": "parent 원문 A", "p2": "parent 원문 B"}
    children = [
        Document(page_content="child1", metadata={"parent_id": "p1"}),
        Document(page_content="child2", metadata={"parent_id": "p1"}),  # 같은 parent
        Document(page_content="child3", metadata={"parent_id": "p2"}),
    ]
    parents = gpc.expand_to_parents(children)

    assert len(parents) == 2  # p1은 한 번만
    assert [p.page_content for p in parents] == ["parent 원문 A", "parent 원문 B"]


def test_expand_to_parents_preserves_rank_order():
    """중복 제거 후에도 첫 등장 순서(=검색 순위)를 유지해야 한다."""
    import graph_parent_child as gpc
    from langchain_core.documents import Document

    gpc._parent_by_id = {"p1": "A", "p2": "B", "p3": "C"}
    children = [
        Document(page_content="c", metadata={"parent_id": "p3"}),
        Document(page_content="c", metadata={"parent_id": "p1"}),
        Document(page_content="c", metadata={"parent_id": "p2"}),
    ]
    parents = gpc.expand_to_parents(children)

    assert [p.page_content for p in parents] == ["C", "A", "B"]


# ── BM25 한국어 형태소 분석 토크나이저(Kiwi) ──

def test_bm25_bigram_overlap_with_josa():
    from graph import bm25_tokenize
    # '회사들'(질의)과 '회사'(문서)가 어간 '회사'로 겹친다 — 예전엔 문자
    # bigram으로 근사했지만, Kiwi는 '들'(XSN 접미사)을 실제로 인식해 뗀다.
    assert "회사" in bm25_tokenize("회사들")
    assert "회사" in bm25_tokenize("회사")


def test_bm25_keeps_full_tokens_and_ascii():
    from graph import bm25_tokenize
    tokens = bm25_tokenize("Jenkins 파이프라인")
    assert "jenkins" in tokens          # 영문은 소문자 어절 그대로
    assert "파이프라인" in tokens        # 복합명사를 노이즈 없이 통째로 인식
    assert "파이" not in tokens         # bigram 노이즈가 더는 안 생긴다


def test_bm25_strips_thousands_comma():
    # "2,292"(코퍼스 표기)와 "2292"(질의)가 콤마 때문에 겹치는 토큰이
    # 0개였던 결함 — 천단위 콤마는 구두점이 아니라 삭제 대상이다.
    from graph import bm25_tokenize
    assert set(bm25_tokenize("2,292")) & set(bm25_tokenize("2292"))
    assert set(bm25_tokenize("12,345,678")) & set(bm25_tokenize("12345678"))


def test_bm25_keeps_spaced_enumeration_comma_separate():
    # 공백이 있는 열거 콤마("Jenkins, ArgoCD")는 천단위가 아니므로 그대로
    # 별개 토큰이어야 한다 — 위 콤마 삭제가 여기까지 지우면 안 된다.
    from graph import bm25_tokenize
    tokens = bm25_tokenize("Jenkins, ArgoCD")
    assert "jenkins" in tokens and "argocd" in tokens


def test_bm25_normalizes_math_italic_unicode():
    # PDF 수식 추출(pypdf)이 만드는 수학 이탤릭 유니코드("𝐼𝑜𝑈")는
    # [0-9A-Za-z]에 안 걸려 _RUNS가 통째로 건너뛰었다 — "IoU"와 겹치는
    # 토큰이 0개였던 결함.
    from graph import bm25_tokenize
    assert set(bm25_tokenize("IoU")) & set(bm25_tokenize("𝐼𝑜𝑈"))
    assert set(bm25_tokenize("Lfocal")) & set(bm25_tokenize("𝐿𝑓𝑜𝑐𝑎𝑙"))


def test_bm25_normalizes_circled_digits_and_superscripts():
    # 원문자 글머리("①②③")·위첨자("N²")도 같은 이유로 ASCII 숫자와
    # 겹치는 토큰이 0개였다 — NFKC 정규화로 표준형(ASCII)에 맞춘다.
    from graph import bm25_tokenize
    assert set(bm25_tokenize("1 Navigator")) & set(bm25_tokenize("① Navigator"))
    assert set(bm25_tokenize("N2")) & set(bm25_tokenize("O(N²)"))


def test_bm25_hangul_syllables_unaffected_by_nfkc():
    # 완성형 한글 음절은 NFKC에서도 NFC와 동일해야 한다(정준 분해 후
    # 재조합 결과가 같음) — 정규화 도입이 Kiwi 형태소 분석을 깨면 안 된다.
    # '알려주세요'는 어간 '알리'+'주'로, 조사/어미 없이 정확히 분해된다.
    from graph import bm25_tokenize
    tokens = bm25_tokenize("회사들을 알려주세요")
    assert "회사" in tokens and "알리" in tokens and "주" in tokens


def test_bm25_decomposes_korean_compound_noun_regardless_of_spacing():
    # "화자분할"(붙여쓰기)과 "화자 분할"(띄어쓰기)이 Kiwi 형태소 경계
    # 인식으로 동일하게 분해된다 — bigram으로는 못 하던 정확한 복합명사
    # 분해(한국어 전처리 재점검, 2026-08).
    from graph import bm25_tokenize
    assert set(bm25_tokenize("화자분할")) == set(bm25_tokenize("화자 분할")) == {"화자", "분할"}


def test_bm25_keeps_serial_number_as_single_token():
    # "10-2538225-0000" 같은 일련번호는 Kiwi가 W_SERIAL 태그로 통째로
    # 인식한다 — 예전 런(run) 방식이 하던 걸 형태소 분석기가 대신한다.
    from graph import bm25_tokenize
    assert "10-2538225-0000" in bm25_tokenize("특허 번호 10-2538225-0000 입니다")


def test_bm25_josa_particle_does_not_break_ascii_token_match():
    # "Throughput은"(SL+JX 조사)과 "Throughput:"(SL+SP 기호)이 조사·기호를
    # 버리고 둘 다 'throughput' 하나로 겹친다 — 원래 런 기반 수정의 동기가
    # 됐던 결함(README "BM25 토크나이저 결함")을 Kiwi로도 재확인.
    from graph import bm25_tokenize
    assert "throughput" in bm25_tokenize("Throughput은 얼마인가요")
    assert "throughput" in bm25_tokenize("Throughput:")


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

    import config

    n = config.GENERATE_TOP_N
    docs = [Document(page_content=f"본문{i}", metadata={"source": f"{i}.md"})
            for i in range(n + 2)]                 # 항상 N보다 2개 많게
    out = graph.generate({"question": "질문", "documents": docs})

    # 계약: 출처도 프롬프트도 상위 N개까지만. N 값 자체는 실측으로 바뀐다.
    assert out["sources"] == sorted(f"{i}.md" for i in range(n))
    assert f"본문{n}" not in seen["context"]        # N+1번째는 들어가지 않는다
    assert out["contexts"] == [f"본문{i}" for i in range(n)]


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


# ── 거부가 정답인 케이스 (환각 측정) ──
#
# 예전 채점기는 거부를 **무조건 오답**으로 셌다. 그래서 "코퍼스에 답이 없는
# 질문"을 평가셋에 넣는 것 자체가 불가능했고, 환각을 재는 지표가 없었다.

def test_expect_refusal_passes_when_model_refuses():
    case = {"question": "이윤선의 혈액형은?", "expect_refusal": True}
    assert is_pass("문서에서 찾을 수 없습니다.", case)
    assert is_pass("해당 정보가 없습니다.", case)


def test_expect_refusal_fails_when_model_makes_something_up():
    case = {"question": "이윤선의 혈액형은?", "expect_refusal": True}
    assert not is_pass("이윤선의 혈액형은 A형입니다.", case)


def test_expect_refusal_needs_no_answer_patterns():
    """거부 케이스는 정답 패턴이 없어도 기준 누락 오류를 내지 않는다."""
    assert is_pass("문서에서 찾을 수 없습니다.",
                   {"question": "q", "expect_refusal": True})


def test_normal_case_still_fails_on_refusal():
    case = {"expected_keywords": ["2292"], "answer_patterns": ["2292"]}
    assert not is_pass("문서에서 찾을 수 없습니다. 2292…", case)


# ── 채점 패턴이 코퍼스와 매치되는가 ──
#
# 팝 노이즈 패턴이 'Residual Buffer'만 허용했는데 문서엔 '잉여 버퍼'도 있어서,
# 근거에 충실한 정답이 오답으로 집계되고 있었다. 어떤 청크와도 매치되지 않는
# 패턴은 그 케이스가 **영원히 실패**한다는 뜻이라 명백한 버그다.

def _corpus_and_cases():
    import json

    chunks = [json.loads(line) for line
              in (ROOT / "data" / "chunks.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    cases = json.loads((ROOT / "eval" / "eval_set.json").read_text(
        encoding="utf-8"))
    return chunks, "\n".join(c["page_content"] for c in chunks), cases


def test_extractive_answer_patterns_match_the_corpus():
    """원문에서 그대로 뽑아 답하는 유형은 패턴이 코퍼스에 있어야 한다.

    집계·비교는 제외한다 — "2023년 논문 몇 편?"의 답 '3편'은 표의 행을
    세야 나오는 값이라 원문에 없는 게 정상이다. 이 유형은 앵커로 검증한다.
    """
    import re

    import graph  # noqa: F401  (src 경로가 잡혔는지 확인용)

    _, corpus, cases = _corpus_and_cases()
    derived = {"aggregation", "comparison"}
    for case in cases:
        if case.get("expect_refusal") or case.get("type") in derived:
            continue
        for p in case.get("answer_patterns", []):
            assert re.search(p, corpus), \
                f"패턴 {p!r}이 코퍼스에 없다 — {case['question']}는 영원히 실패한다"


def test_gold_anchors_exist_in_the_corpus():
    """앵커는 정답 근거 청크를 가리킨다 — 사라졌으면 문항이 죽은 것이다."""
    chunks, _, cases = _corpus_and_cases()
    for case in cases:
        for a in case.get("gold_anchors", []):
            assert any(a in c["page_content"] for c in chunks), \
                f"앵커 {a!r}를 담은 청크가 없다 — {case['question']}"


def test_refusal_cases_have_no_answer_in_the_corpus():
    """거부가 정답인 문항에 답이 생기면 그 문항은 더 이상 거부 케이스가 아니다."""
    _, corpus, cases = _corpus_and_cases()
    import re
    for case in cases:
        if not case.get("expect_refusal"):
            continue
        for p in case.get("answer_patterns", []):
            assert not re.search(p, corpus), \
                f"거부 문항인데 코퍼스에 답이 있다 — {case['question']}"


# ── 위치 메타데이터 & 이웃 확장 ──
#
# 근거를 더 주려고 하위 랭크 청크를 **추가**하면 generate가 랭크 역순 배치라
# 프롬프트 맨 앞(모델의 주의가 가장 약한 자리)에 놓여 소용이 없다는 것을
# 두 번 확인했다(top-6, TOP_K=15). 이웃 확장은 순위 자리를 유지한 채 각
# 청크의 문맥만 넓히는 방식이라 성격이 다르다.

def test_chunks_have_position_metadata():
    chunks, _, _ = _corpus_and_cases()
    for c in chunks:
        for key in ("source", "chunk_index", "doc_index", "doc_total"):
            assert key in c["metadata"], f"{key} 누락: {c['metadata']}"
    idx = [c["metadata"]["chunk_index"] for c in chunks]
    assert idx == list(range(len(chunks))), "chunk_index가 연속이 아니다"


def test_neighbor_expansion_off_by_default(monkeypatch):
    import config
    import graph
    from langchain_core.documents import Document

    monkeypatch.setattr(config, "NEIGHBOR_WINDOW", 0)
    docs = [Document(page_content="a", metadata={"chunk_index": 1,
                                                 "source": "x.md"})]
    assert graph.expand_with_neighbors(docs) is docs      # 손대지 않는다


def test_neighbor_expansion_keeps_count_and_order(monkeypatch):
    """개수와 순위는 그대로, 각 청크의 내용만 넓어진다."""
    import config
    import graph
    from langchain_core.documents import Document

    table = {i: Document(page_content=f"c{i}",
                         metadata={"chunk_index": i, "source": "x.md"})
             for i in range(5)}
    monkeypatch.setattr(config, "NEIGHBOR_WINDOW", 1)
    monkeypatch.setattr(graph, "_load_indexes", lambda: (None, None))
    monkeypatch.setattr(graph, "_chunks_by_index", lambda: table)

    out = graph.expand_with_neighbors([table[2], table[0]])
    assert len(out) == 2                        # 개수 유지
    assert out[0].page_content == "c1\nc2\nc3"   # 앞뒤가 붙는다
    assert out[1].page_content == "c0\nc1"       # 경계에서 잘림


def test_neighbor_expansion_stops_at_document_boundary(monkeypatch):
    """다른 문서의 텍스트를 붙이면 노이즈다."""
    import config
    import graph
    from langchain_core.documents import Document

    table = {
        0: Document(page_content="a0", metadata={"chunk_index": 0, "source": "a.md"}),
        1: Document(page_content="b0", metadata={"chunk_index": 1, "source": "b.md"}),
        2: Document(page_content="b1", metadata={"chunk_index": 2, "source": "b.md"}),
    }
    monkeypatch.setattr(config, "NEIGHBOR_WINDOW", 1)
    monkeypatch.setattr(graph, "_load_indexes", lambda: (None, None))
    monkeypatch.setattr(graph, "_chunks_by_index", lambda: table)

    out = graph.expand_with_neighbors([table[1]])
    assert out[0].page_content == "b0\nb1"       # a.md 는 안 붙는다


def test_diagram_chunks_are_not_expanded(monkeypatch):
    """다이어그램은 이미 통짜(산문의 6.4배)라 더 넓힐 이유가 없다."""
    import config
    import graph
    from langchain_core.documents import Document

    d = Document(page_content="box", metadata={"chunk_index": 1,
                                               "source": "x.md",
                                               "kind": "diagram"})
    monkeypatch.setattr(config, "NEIGHBOR_WINDOW", 1)
    monkeypatch.setattr(graph, "_load_indexes", lambda: (None, None))
    monkeypatch.setattr(graph, "_chunks_by_index", lambda: {1: d})
    assert graph.expand_with_neighbors([d])[0].page_content == "box"


# ── 인덱스-청크 결속 ──
#
# 벡터 검색은 인덱스를, BM25는 chunks.jsonl을 각각 읽는다. 둘이 어긋나면
# 서로 다른 청킹을 RRF로 섞게 되는데 **증상이 없다**. 매니페스트로 대조한다.

def test_index_manifest_matches_chunks():
    import json

    import config
    import ingest

    manifest_path = ROOT / "data" / "index_manifest.json"
    if not manifest_path.exists():
        pytest.skip("인덱스 매니페스트 없음 (ingest 미실행 환경)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunks_md5"] == ingest.chunk_fingerprint(config.CHUNKS_PATH)


def test_consistency_check_raises_on_mismatch(tmp_path, monkeypatch):
    import json

    import config
    import graph

    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"page_content":"a","metadata":{}}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"chunks_md5": "다른값", "n_chunks": 1}),
                        encoding="utf-8")
    monkeypatch.setattr(config, "CHUNKS_PATH", str(chunks))
    monkeypatch.setattr(config, "INDEX_MANIFEST", str(manifest))

    with pytest.raises(graph.IndexError_):
        graph.check_index_consistency()


def test_run_metadata_records_what_makes_results_comparable():
    """결과 비교 가능성을 판단하는 데 필요한 항목이 빠지지 않았는지."""
    from runmeta import run_metadata

    meta = run_metadata()
    for key in ("llm_model", "embed_model", "ollama_version",
                "index_chunks_md5", "params", "host"):
        assert key in meta
    for key in ("TOP_K", "GENERATE_TOP_N", "MAX_REWRITES", "CHUNK_SIZE"):
        assert key in meta["params"]


def test_eval_set_composition():
    """유형 구성이 무너지지 않았는지 — 특히 거부 문항이 사라지면
    환각을 재는 지표가 통째로 없어진다."""
    from collections import Counter

    _, _, cases = _corpus_and_cases()
    kinds = Counter(c.get("type", "fact") for c in cases)
    assert len(cases) >= 50
    assert kinds["refusal"] >= 10          # 환각 측정
    assert kinds["aggregation"] >= 5       # 집계 — 현재 알려진 약점
    assert kinds["enumeration"] >= 5       # 열거 — 현재 알려진 약점
    assert all(c.get("type") for c in cases), "type 없는 문항이 있다"


# ── grade 판정 (구조화 출력) ──
#
# 예전엔 "yes/no만 출력하라"고 지시하고 자유 문장을 정규식(앵커+서술문 2단)
# 으로 역파싱했다. eval_grade.py로 그 파서를 직접 재보니 오탐(충분한데
# 재검색 보냄) 20%가 나왔고, corrective 루프 A/B 재검증(51문항 층화 표본)
# 에서 그 오탐이 실제로 정답률을 깎는 것까지 확인됐다 — grade가 이미
# 충분한 근거를 "부족"으로 오판해 불필요한 재작성을 걸고, 재작성된 질의가
# 원래보다 검색을 흐렸다.
#
# ChatOllama.with_structured_output()으로 모델 출력 자체를 GradeVerdict
# 스키마에 강제해 이 파싱 버그 클래스를 없앴다("아닙니다"가 음절 분해상
# "아니"와 매치 안 되는 것 같은 문제가 구조적으로 발생할 수 없다). 파싱
# 관련 회귀 테스트(아닙니다·아니면·사족 등)는 그 파서와 함께 사라졌다 —
# 스키마가 boolean이라 애초에 그런 모호성이 없다.
#
# LCEL 체인(GRADE_PROMPT | 구조화 LLM)의 실제 동작은 LLM 호출이 필요해
# 여기서(LLM 없이 도는 스위트) 재현하지 않는다 — 직접 호출로 검증했다:
# 관련 있는 컨텍스트 → relevant=True(17.5s), 무관한 컨텍스트 →
# relevant=False(88.2s). 여기서는 grade() 노드가 judge_relevance의 3값
# 반환을 올바르게 라우팅하는지만 확인한다(judge_relevance는 통째로 목).

def test_grade_verdict_schema():
    from graph import GradeVerdict
    assert GradeVerdict(relevant=True).relevant is True
    assert GradeVerdict(relevant=False).relevant is False


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


def test_clean_rewrite_skips_preamble_line():
    """3B가 실제로 뱉은 형태 — 안내문 뒤에 진짜 질문이 온다.

    첫 줄만 취하면 질문을 통째로 잃는다(가드가 오히려 손해가 되는 경우).
    """
    from graph import clean_rewrite
    raw = ("화자 분할 모델을 설명하는 문장을 만드는 데 도움이 되겠습니다. "
           "재작성된 질문은 다음과 같습니다:\n"
           "화자 분할에 사용한 모델은 무엇인가?")
    assert clean_rewrite(raw, "원 질문") == "화자 분할에 사용한 모델은 무엇인가?"


def test_clean_rewrite_rejects_preamble_only():
    from graph import clean_rewrite
    assert clean_rewrite("재작성된 질문은 다음과 같습니다:", "원 질문") is None


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


# ── 유형별 라우팅 (route.py) ──
#
# fact·temporal·refusal에서 오탐이 나면 원래 좋던 기본 설정을 나쁜 설정으로
# 바꿔버리므로(README의 NEIGHBOR_WINDOW·CONTEXT_ORDER 실측 참고), 그 세
# 유형에서 오탐 0건인 것을 회귀 테스트로 고정한다. enumeration은 리콜이
# 낮아도(안전하게 fact로 폴백) 손해가 없어 여기서는 안 다룬다.

def test_classify_type_no_false_positive_on_fact_temporal_refusal():
    from route import classify_question_type

    safe_questions = [
        "TTS 시스템의 동시 처리 채널은 몇 개로 확장했나요?",   # fact, "몇 개"
        "TTS 서비스는 몇 개 언어를 지원하나요?",              # fact, "몇 개"
        "STT 엔진은 어떤 모델을 기반으로 만들었나요?",         # fact, "어떤"
        "석사 학위는 어느 대학교에서 받았나요?",               # temporal 아님이지만 "어느" 포함 확인
        "등록된 특허 번호를 알려주세요.",                      # fact, "알려주세요"(들 없음)
    ]
    for q in safe_questions:
        assert classify_question_type(q) == "fact", q


def test_classify_type_catches_aggregation_and_comparison():
    from route import classify_question_type

    assert classify_question_type("이윤선의 제1저자 논문은 몇 편인가요?") == "aggregation"
    assert classify_question_type("등록된 특허는 총 몇 건인가요?") == "aggregation"
    assert classify_question_type(
        "인피닉과 이든티앤에스 중 더 오래 근무한 회사의 근무 기간은 얼마인가요?"
    ) == "comparison"


def test_classify_type_catches_some_enumeration_safely():
    from route import classify_question_type

    assert classify_question_type("이윤선이 근무한 회사들을 알려주세요.") == "enumeration"
    # 신호가 약한 열거형은 fact로 안전하게 폴백한다(오분류보다 미분류가 낫다)
    assert classify_question_type("TTS 관리자 대시보드의 프론트엔드 기술 스택은 무엇인가요?") == "fact"


def test_route_strategy_node_returns_overrides(monkeypatch):
    """route_strategy 노드가 유형에 맞는 전략을 State로 돌려주는가."""
    import config
    from graph import route_strategy

    monkeypatch.setattr(config, "TYPE_ROUTING", False)
    assert route_strategy({"question": "이윤선의 제1저자 논문은 몇 편인가요?"}) \
        == {"strategy": {}}                       # 꺼져 있으면 전략 없음

    monkeypatch.setattr(config, "TYPE_ROUTING", True)
    # aggregation → 이웃 확장 + 개선이 측정된 top-3로 함께 이동.
    # W=1을 top-5 위에 얹으면 컨텍스트 7.7k자(미측정 조합)가 되어 base가
    # 맞히던 질문을 깨뜨렸다 — route.py ROUTES 주석 참고.
    out = route_strategy({"question": "이윤선의 제1저자 논문은 몇 편인가요?"})
    assert out["strategy"] == {"NEIGHBOR_WINDOW": 1, "GENERATE_TOP_N": 3}

    out = route_strategy({"question": "TTS 프로젝트에서 TTFB를 얼마나 개선했나요?"})
    assert out["strategy"] == {}                  # fact는 오버라이드 없음


def test_route_strategy_never_mutates_global_config(monkeypatch):
    """전역을 안 건드리는 것이 이 설계의 요점이다.

    예전엔 run()이 컨텍스트 매니저로 config를 잠깐 바꿨다 되돌렸다. FastAPI가
    동기 핸들러를 스레드풀에서 돌리므로, 동시 요청 두 건이 서로의 전역을
    덮어썼다(코드 주석에 "알려진 한계"로 적혀 있던 자리). 전략을 State로
    나르면 그 경합이 구조적으로 사라진다.
    """
    import config
    from graph import route_strategy

    monkeypatch.setattr(config, "TYPE_ROUTING", True)
    before = {k: getattr(config, k) for k in ("NEIGHBOR_WINDOW", "GENERATE_TOP_N")}
    route_strategy({"question": "이윤선의 제1저자 논문은 몇 편인가요?"})
    after = {k: getattr(config, k) for k in ("NEIGHBOR_WINDOW", "GENERATE_TOP_N")}
    assert before == after


def test_context_docs_uses_strategy_over_global(monkeypatch):
    """State의 전략이 전역 기본값을 이긴다 (없으면 전역을 쓴다)."""
    import config
    from graph import context_docs

    monkeypatch.setattr(config, "GENERATE_TOP_N", 5)
    monkeypatch.setattr(config, "NEIGHBOR_WINDOW", 0)
    docs = [f"d{i}" for i in range(6)]

    assert len(context_docs(docs)) == 5                       # 전역
    assert len(context_docs(docs, {"GENERATE_TOP_N": 3})) == 3  # 전략이 우선
    assert len(context_docs(docs, {})) == 5                   # 빈 전략이면 전역


# ── 리랭커 (graph.rerank) ──
#
# LLM 호출을 재현하지 않고 LCEL 체인만 목으로 대체해 후처리 로직(순서
# 재배열, 실패 시 원 순서 유지)을 검증한다. RunnableLambda로 감싸면
# `RERANK_PROMPT | fake`가 실제 LCEL 파이프처럼 동작해 실제 코드 경로를
# 그대로 통과한다.

def _fake_rerank_llm(result, monkeypatch):
    from langchain_core.runnables import RunnableLambda
    import graph
    monkeypatch.setattr(graph, "_structured_rerank_llm",
                        lambda: RunnableLambda(lambda _: result))


def test_rerank_reorders_by_parsed_indices(monkeypatch):
    import graph
    from langchain_core.documents import Document

    docs = [Document(page_content=f"doc{i}") for i in range(3)]
    _fake_rerank_llm(
        {"parsed": graph.RerankOrder(ranked_indices=[2, 0, 1]), "raw": None},
        monkeypatch)
    out = graph.rerank("q", docs)
    assert [d.page_content for d in out] == ["doc2", "doc0", "doc1"]


def test_rerank_falls_back_on_incomplete_permutation(monkeypatch):
    """인덱스 누락(2개만 반환 등) — 원 순서를 그대로 유지한다."""
    import graph
    from langchain_core.documents import Document

    docs = [Document(page_content=f"doc{i}") for i in range(3)]
    _fake_rerank_llm(
        {"parsed": graph.RerankOrder(ranked_indices=[0, 1]), "raw": None},
        monkeypatch)
    out = graph.rerank("q", docs)
    assert [d.page_content for d in out] == ["doc0", "doc1", "doc2"]


def test_rerank_falls_back_on_unparsed(monkeypatch):
    import graph
    from langchain_core.documents import Document

    docs = [Document(page_content=f"doc{i}") for i in range(3)]
    _fake_rerank_llm({"parsed": None, "raw": "", "parsing_error": "x"}, monkeypatch)
    out = graph.rerank("q", docs)
    assert [d.page_content for d in out] == ["doc0", "doc1", "doc2"]


def test_rerank_skips_single_doc():
    """문서가 0~1개면 재정렬할 게 없어 LLM을 부르지 않는다."""
    from graph import rerank
    assert rerank("q", []) == []


def test_retrieve_reranks_only_when_config_enabled(monkeypatch):
    import config
    import graph

    calls = []
    monkeypatch.setattr(graph, "hybrid_search", lambda q: ["d1", "d2"])
    monkeypatch.setattr(graph, "rerank", lambda q, d: calls.append(1) or d)

    monkeypatch.setattr(config, "RERANK", False)
    graph.retrieve({"query": "q", "question": "q"})
    assert calls == []          # 꺼져 있으면 rerank 호출 안 함

    monkeypatch.setattr(config, "RERANK", True)
    graph.retrieve({"query": "q", "question": "q"})
    assert calls == [1]


# ── 응답 캐시 (cache.py) ──
#
# api.py의 서빙 경계에만 건다 — graph.ask()에는 안 건다. eval 스크립트가
# 같은 질문을 설정만 바꿔(MAX_REWRITES 등) 반복 호출하는 게 이 프로젝트의
# 기본 측정 방식이라, graph 안에 캐시가 있으면 A/B가 조용히 오염된다.
# 그래서 캐시 키에 설정 지문을 넣어 설정이 다르면 자동으로 다른 항목이
# 되게 만들었다 — 여기서는 그 지문 분리가 실제로 동작하는지만 확인한다.

@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    import cache
    monkeypatch.setattr(cache, "CACHE_PATH", tmp_path / "answer_cache.json")
    monkeypatch.setattr(cache, "_cache", None)
    return cache


def test_cache_miss_then_hit(isolated_cache):
    cache = isolated_cache
    assert cache.get("질문") is None
    cache.put("질문", {"answer": "답", "sources": [], "rewrites": 0})
    assert cache.get("질문") == {"answer": "답", "sources": [], "rewrites": 0}


def test_cache_key_changes_with_config(isolated_cache, monkeypatch):
    """같은 질문이라도 설정이 다르면 다른 캐시 항목이어야 한다 — 아니면
    eval 스크립트의 A/B 비교가 옛 설정의 캐시된 답을 그대로 받는다."""
    import config
    cache = isolated_cache

    monkeypatch.setattr(config, "MAX_REWRITES", 0)
    cache.put("질문", {"answer": "OFF일 때 답"})

    monkeypatch.setattr(config, "MAX_REWRITES", 1)
    assert cache.get("질문") is None            # 설정이 바뀌어 미스여야 함
    cache.put("질문", {"answer": "ON일 때 답"})

    monkeypatch.setattr(config, "MAX_REWRITES", 0)
    assert cache.get("질문")["answer"] == "OFF일 때 답"   # 원래 설정으로 돌아오면 그때 캐시


def test_cache_persists_to_disk(isolated_cache):
    cache = isolated_cache
    cache.put("질문", {"answer": "답"})
    assert cache.CACHE_PATH.exists()

    # 프로세스 재시작을 흉내: 메모리 캐시를 비우고 디스크에서 다시 읽기
    cache._cache = None
    assert cache.get("질문") == {"answer": "답"}


# ── 질의 트레이스 로그 (tracelog.py) ──

def test_tracelog_appends_jsonl(tmp_path, monkeypatch):
    import tracelog
    monkeypatch.setattr(tracelog, "TRACE_PATH", tmp_path / "query_trace.jsonl")

    tracelog.log("질문1", {"answer": "답1", "sources": ["a.md"], "rewrites": 0},
                1.23, cached=False)
    tracelog.log("질문2", {"answer": "답2", "sources": [], "rewrites": 1},
                0.5, cached=True)

    lines = tracelog.TRACE_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row1 = json.loads(lines[0])
    assert row1["question"] == "질문1"
    assert row1["answer"] == "답1"
    assert row1["cached"] is False
    assert row1["elapsed_sec"] == 1.2          # round(…, 1)
    assert "ts" in row1


def test_tracelog_truncates_long_answers(tmp_path, monkeypatch):
    """트레이스는 디버깅용이지 원문 저장소가 아니다 — 너무 길면 자른다."""
    import tracelog
    monkeypatch.setattr(tracelog, "TRACE_PATH", tmp_path / "t.jsonl")
    tracelog.log("q", {"answer": "가" * 1000, "sources": [], "rewrites": 0}, 1.0)
    row = json.loads(tracelog.TRACE_PATH.read_text(encoding="utf-8"))
    assert len(row["answer"]) <= 500


# ── /health 강화 ──

def test_health_reflects_preflight_problems(monkeypatch):
    """예전엔 무조건 ok였다 — 이제 preflight 점검 결과를 그대로 반영해야 한다."""
    import api

    monkeypatch.setattr(api, "check_all", lambda strict: ["Ollama 연결 실패"])
    out = api.health()
    assert out["status"] == "degraded"
    assert out["problems"] == ["Ollama 연결 실패"]

    monkeypatch.setattr(api, "check_all", lambda strict: [])
    out = api.health()
    assert out["status"] == "ok"
    assert out["problems"] == []


# ── groundedness 검증 (graph.verify) ──
#
# generate 직후 답변이 근거에 실제로 기반하는지 사후 확인하는 관측
# 노드다. grade·rerank와 같은 원칙(fail-open, 조용히 흡수하지 않음)을
# 따르는지, 그리고 꺼져 있으면 LLM을 아예 안 부르는지를 확인한다.

def test_verify_skips_llm_call_when_disabled(monkeypatch):
    """기본은 꺼짐 — LLM을 부르지 않고 즉시 None을 반환해야 한다."""
    import config
    import graph

    monkeypatch.setattr(config, "VERIFY_GROUNDING", False)
    called = []
    monkeypatch.setattr(graph, "_structured_verify_llm",
                        lambda: called.append(1))
    out = graph.verify({"question": "q", "documents": [], "answer": "a"})
    assert out == {"grounded": None, "unsupported_claim": None}
    assert called == []          # LLM 캐시 함수 자체를 호출하지 않았다


def test_verify_returns_parsed_verdict(monkeypatch):
    import config
    import graph
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(config, "VERIFY_GROUNDING", True)
    verdict = graph.GroundednessVerdict(grounded=False, unsupported_claim="근거 없는 날짜 주장")
    monkeypatch.setattr(graph, "_structured_verify_llm",
                        lambda: RunnableLambda(lambda _: {"parsed": verdict, "raw": None}))
    out = graph.verify({"question": "q", "documents": [], "answer": "a"})
    assert out == {"grounded": False, "unsupported_claim": "근거 없는 날짜 주장"}


def test_verify_fails_open_on_unparsed(monkeypatch):
    """파싱 실패는 답변을 바꾸지 않고 grounded=None으로 조용히 기록만."""
    import config
    import graph
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(config, "VERIFY_GROUNDING", True)
    monkeypatch.setattr(
        graph, "_structured_verify_llm",
        lambda: RunnableLambda(lambda _: {"parsed": None, "raw": None}))
    out = graph.verify({"question": "q", "documents": [], "answer": "a"})
    assert out == {"grounded": None, "unsupported_claim": None}


def test_graph_compiles_with_verify_node():
    from graph import build_graph
    build_graph()   # verify 노드 추가 후에도 그래프가 컴파일되는지


# ── 시맨틱 청킹 (semantic_chunk.py) ──
#
# 핵심 로직은 임베딩을 인자로 받는 순수 함수라 Ollama 없이 재현 가능하다.
# 클러스터 둘(코사인 유사도가 뚜렷이 갈리는 벡터 두 묶음)을 만들어
# breakpoint가 정확히 그 경계에서 잡히는지 확인한다.

def test_split_sentences_basic():
    from semantic_chunk import split_sentences
    out = split_sentences("이것은 문장입니다. 이것도 문장입니다.\n\n다른 문단입니다.")
    assert out == ["이것은 문장입니다.", "이것도 문장입니다.", "다른 문단입니다."]


def test_find_breakpoints_detects_cluster_transition():
    from semantic_chunk import find_breakpoints
    embeddings = [
        [1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.9, 0.1, 0.0],   # 클러스터 A
        [0.0, 0.0, 1.0], [0.05, 0.0, 0.95], [0.0, 0.05, 0.9],  # 클러스터 B
    ]
    bp = find_breakpoints(embeddings, percentile=80)
    assert 2 in bp   # 인덱스 2(세 번째 문장) 뒤에서 끊겨야 한다


def test_find_breakpoints_empty_on_uniform_embeddings():
    """전부 비슷하면 튀는 지점이 없어 breakpoint도 없어야 한다."""
    from semantic_chunk import find_breakpoints
    embeddings = [[1.0, 0.0, 0.0]] * 5
    assert find_breakpoints(embeddings, percentile=95) == set()


def test_merge_into_chunks_respects_breakpoints():
    from semantic_chunk import merge_into_chunks
    sents = [f"문장{i}" for i in range(6)]
    chunks = merge_into_chunks(sents, breakpoints={2}, max_chars=1000, min_chars=1)
    assert chunks == ["문장0 문장1 문장2", "문장3 문장4 문장5"]


def test_merge_into_chunks_enforces_max_chars_even_without_breakpoint():
    """breakpoint가 없어도(주제 전환이 없는 긴 구간) 상한을 넘기지 않는다."""
    from semantic_chunk import merge_into_chunks
    sents = ["가" * 300, "나" * 300, "다" * 300]
    chunks = merge_into_chunks(sents, breakpoints=set(), max_chars=500, min_chars=1)
    assert all(len(c) <= 700 for c in chunks)   # 문장 하나가 최대 300+공백이라 여유
    assert len(chunks) >= 2                      # 상한 때문에 최소 한 번은 갈렸다


def test_merge_into_chunks_absorbs_short_fragments():
    from semantic_chunk import merge_into_chunks
    sents = ["첫 문장" * 20, "짧음", "그다음 문장" * 20]
    chunks = merge_into_chunks(sents, breakpoints={0, 1}, max_chars=1000, min_chars=10)
    assert not any(len(c) < 10 for c in chunks)   # 짧은 조각이 단독으로 안 남는다


def test_chunk_semantically_uses_injected_embed_fn():
    """embed_fn을 주입받아 쓴다 — 실제 Ollama 호출 없이 파이프라인 전체 검증."""
    from semantic_chunk import chunk_semantically

    def fake_embed(sentences):
        # 앞 절반은 벡터 A, 뒤 절반은 벡터 B — 문장 개수와 무관하게 동작해야 함
        half = len(sentences) // 2
        return [[1.0, 0.0]] * half + [[0.0, 1.0]] * (len(sentences) - half)

    text = "가나다. 라마바. 사아자.\n\n차카타. 파하거. 너더러."
    chunks = chunk_semantically(text, fake_embed, percentile=50, max_chars=1000, min_chars=1)
    assert len(chunks) >= 2   # 클러스터 전환이 최소 한 번은 잡혀야 한다


# ── 라우팅 배선 회귀 테스트 ──
#
# 유형별 라우팅은 오래 `graph.ask()` 안에서만 걸려 있었고, 정작 실사용·평가
# 경로(api.py·cli.py·eval/evaluate.py)는 `graph.invoke()`를 직접 불러 **라우팅을
# 통째로 건너뛰고 있었다**. 그래서 TYPE_ROUTING=1을 켜도 서빙에 반영되지 않았고,
# evaluate.py로 잰 "켠 전후" 수치는 같은 코드 경로를 두 번 잰 것이었다.
# 진입점을 graph.run()으로 모아 고쳤으니, 그 계약이 깨지지 않는지 고정한다.

def test_strategy_flows_from_node_to_context_docs(monkeypatch):
    """route_strategy가 실은 전략이 grade·generate까지 전달되는가.

    라우팅이 그래프 밖(run의 컨텍스트 매니저)에 있던 시절, 실사용 경로가
    graph.invoke를 직접 불러 라우팅을 통째로 건너뛴 적이 있다. 이제는
    노드라서 그래프를 도는 한 반드시 거치지만, 전략이 실제로 context_docs
    까지 닿는지는 별도로 고정한다.
    """
    import config
    import graph as g
    import route

    from types import SimpleNamespace

    from langchain_core.runnables import RunnableLambda

    seen = []
    monkeypatch.setattr(g, "context_docs",
                        lambda docs, strategy=None: seen.append(strategy) or [])
    monkeypatch.setattr(g, "get_llm", lambda *a, **k: RunnableLambda(
        lambda _: SimpleNamespace(content="답변")))
    monkeypatch.setattr(config, "TYPE_ROUTING", True)

    state = g.route_strategy({"question": "등록된 특허는 총 몇 건인가요?"})
    g.generate({"question": "q", "documents": [], **state})

    assert seen == [route.ROUTES["aggregation"]], f"전략 미전달: {seen}"



def test_uses_team_is_gated_by_config_flag():
    """TEAM_ROUTING이 꺼져 있으면 멀티홉 질문도 팀으로 안 보낸다."""
    import config
    import graph as g

    multihop = "인피닉과 이든티앤에스 중 더 오래 근무한 회사의 근무 기간은 얼마인가요?"
    prev = config.TEAM_ROUTING
    try:
        config.TEAM_ROUTING = False
        assert g.uses_team(multihop) is False
        config.TEAM_ROUTING = True
        assert g.uses_team(multihop) is True          # comparison → 팀
        assert g.uses_team("영어 OPIc 등급은 무엇인가요?") is False   # fact → 단일
    finally:
        config.TEAM_ROUTING = prev


def test_enumeration_catches_numeral_plus_gaji():
    """'센서 두 가지는 무엇인가요' — 수사+가지는 답이 목록임을 질문이 못박는다.
    '몇 가지'만 보다가 놓쳤던 형태."""
    from route import classify_question_type
    assert classify_question_type(
        "3D 시맨틱 세그멘테이션 연구에서 융합한 센서 두 가지는 무엇인가요?") == "enumeration"
    assert classify_question_type("몇 가지 방법이 있나요?") == "enumeration"


def test_classifier_keeps_precision_on_lookalike_fact_questions():
    """리콜을 넓힐 때 정밀도가 깨지지 않는지 고정한다.

    'X는 무엇인가요'·'어떤 X를 사용'은 표면형이 같아도 대부분 fact/refusal이라
    (75문항 실측: 무엇인가요 16건 중 enumeration은 3건뿐) 잡으면 손해다.
    이 문항들이 fact로 남아 있어야 한다.
    """
    from route import classify_question_type
    for q in ["영어 OPIc 등급은 무엇인가요?",
              "화자 분할에는 어떤 모델을 사용했나요?",
              "3D 시맨틱 세그멘테이션 연구에서 사용한 GPU는 무엇인가요?",
              "이윤선의 혈액형은 무엇인가요?"]:
        assert classify_question_type(q) == "fact", q


# ── HyDE (가상 답변 단락 검색) ──

def test_clean_hypothetical_strips_preamble_and_joins_lines():
    from graph import clean_hypothetical
    raw = ("재작성된 질문은 다음과 같습니다:\n"
           "딥러닝 프레임워크로는 TensorFlow를 사용했습니다.\n"
           "버전은 2.8.0입니다.")
    # 안내문 줄은 버리고, 남은 줄은 첫 줄만이 아니라 전부 합친다 —
    # 단락 전체가 검색 신호라 clean_rewrite와 달리 정보를 버리면 손해다.
    assert clean_hypothetical(raw) == (
        "딥러닝 프레임워크로는 TensorFlow를 사용했습니다. 버전은 2.8.0입니다.")


def test_clean_hypothetical_returns_none_when_unusable():
    from graph import clean_hypothetical
    assert clean_hypothetical("") is None
    assert clean_hypothetical("다음과 같습니다:") is None   # 안내문뿐


def test_clean_hypothetical_caps_length():
    from graph import _MAX_HYDE_CHARS, clean_hypothetical
    out = clean_hypothetical("가" * 2000)
    assert out is not None and len(out) == _MAX_HYDE_CHARS


def _doc(text: str, source: str = "a.md"):
    from langchain_core.documents import Document
    return Document(page_content=text, metadata={"source": source})


def test_rrf_fuse_dedups_by_content_and_keeps_first_seen():
    from graph import rrf_fuse
    # 내용이 같으면 하나로 합쳐지고, 먼저 본 것(앞 목록 상위 랭크)의
    # 메타데이터가 유지된다 — 출처가 검색 순서에 따라 바뀌면 안 된다.
    fused = rrf_fuse([[_doc("같은 내용", "resume.md")],
                      [_doc("같은 내용", "portfolio.md")]])
    assert len(fused) == 1
    assert fused[0].metadata["source"] == "resume.md"


def test_rrf_fuse_doc_in_more_lists_ranks_higher():
    from graph import rrf_fuse
    # 같은 랭크라면 더 많은 목록에 등장한 문서가 위로 온다 — HyDE 목록을
    # "추가"하는 설계의 근거: 질의·가상 단락 양쪽에서 잡히면 표가 겹친다.
    both = _doc("질의와 가상 단락 양쪽에서 잡힌 청크")
    only = _doc("한쪽에서만 잡힌 청크")
    fused = rrf_fuse([[only, both], [both], [both]])
    assert fused[0].page_content == both.page_content


def test_hyde_term_query_keeps_only_novel_ascii_terms():
    from graph import hyde_term_query
    q = "세그멘테이션 모델 구현에 사용한 딥러닝 프레임워크는 무엇인가요?"
    hypo = ("딥러닝 프레임워크 중 하나로 사용된 것은 TensorFlow입니다. "
            "이 프레임워크는 세그멘테이션 모델 구현에 유용하게 활용되었습니다.")
    out = hyde_term_query(q, hypo)
    # 다리가 되는 영숫자 용어만 남는다 — 질의 중복(세그멘테이션·모델)과
    # 한국어 일반 어휘(유용·활용)는 1차 실측에서 확인된 희석 요인이라 버린다.
    assert out == "tensorflow"


def test_hyde_term_query_none_when_no_novel_terms():
    from graph import hyde_term_query
    # 가상 단락이 질의를 되풀이하기만 하면(회사 목록 문항의 실측 사례)
    # 추가할 신호가 없다 — 빈 질의로 BM25를 부르지 않도록 None.
    assert hyde_term_query("근무한 회사들을 알려주세요",
                           "근무한 회사들을 알려주세요.") is None


# ── 표를 통짜 청크로 (다이어그램과 같은 처리) ──
#
# publications.md의 표가 청크 3개로 쪼개져 있었다. 평가셋 집계 7문항 중
# 4문항이 그 표를 세야 답이 나오는데, 잘린 표로는 **원리적으로** 셀 수 없다.
# 3B의 산술 문제로 보이던 것이 실은 근거가 잘린 문제였다.

def test_table_is_extracted_whole():
    from ingest import extract_tables

    text = ("앞 문단\n\n"
            "| 게재 일자 | 논문명 | 기타 |\n"
            "| --- | --- | --- |\n"
            "| 2023.08 | A | 1저자 |\n"
            "| 2022.11 | B | 1저자 |\n\n"
            "뒷 문단")
    body, tables = extract_tables(text)

    assert len(tables) == 1
    assert tables[0].metadata["kind"] == "table"
    assert "2023.08" in tables[0].page_content and "2022.11" in tables[0].page_content
    assert "|" not in body and "[표: 1]" in body


def test_table_row_wrapped_across_lines_stays_in_one_block():
    """셀 안에 줄바꿈이 있으면 이어지는 줄이 들여쓰기된 채 '|'로 시작한다.

    '^\|'로 잡으면 한 표가 두 조각으로 끊긴다 — publications.md의
    2021.03·2018.04 행이 실제로 그렇다.
    """
    from ingest import extract_tables

    text = ("| 일자 | 제목 | 기타 |\n"
            "| --- | --- | --- |\n"
            "| 2021.03 | [아주 긴 제목](http://x)\n"
            " | 스마트미디어학회 | 1저자 |\n"
            "| 2018.06 | C | 2저자 |\n")
    _, tables = extract_tables(text)

    assert len(tables) == 1, "줄바꿈된 행에서 표가 끊겼다"
    assert "스마트미디어학회" in tables[0].page_content
    assert "2018.06" in tables[0].page_content


def test_pipe_lines_without_separator_are_not_a_table():
    """구분선 없는 파이프 나열을 표로 오인하지 않는다."""
    from ingest import extract_tables

    text = "| 이건 표가 아니다\n| 그냥 파이프로 시작하는 줄\n| 세 줄이지만 구분선이 없다\n"
    body, tables = extract_tables(text)
    assert tables == [] and body.strip() == text.strip()


def test_publications_table_supports_counting():
    """실제 코퍼스의 표가 온전하면 필터별 행 세기로 집계 정답이 나온다."""
    import ingest

    _, blocks = ingest.load_documents()
    pub = [b for b in blocks
           if b.metadata.get("kind") == "table"
           and b.metadata["source"] == "publications.md"]
    assert len(pub) == 1, "publications 표가 통짜가 아니다"

    lines = pub[0].page_content.splitlines()
    for term, expected in [("1저자", 7), ("2저자", 1),
                           ("한국자동차공학회", 2), ("한국항공우주학회", 1)]:
        assert sum(1 for ln in lines if term in ln) == expected, term


def test_whole_chunks_are_never_neighbor_expanded(monkeypatch):
    """통짜 청크(다이어그램·표)에는 이웃을 붙이지 않는다.

    표를 추가하기 전엔 `kind == "diagram"`만 걸렀다. 표도 이미 완결된
    블록이라, 앞뒤를 붙이면 문서의 다른 부분이 딸려 들어가 오염된다.
    """
    import config
    import graph
    from langchain_core.documents import Document

    table = Document(page_content="| a | b |",
                     metadata={"chunk_index": 1, "source": "x.md", "kind": "table"})
    prose = Document(page_content="p1",
                     metadata={"chunk_index": 1, "source": "x.md"})
    lookup = {0: Document(page_content="p0", metadata={"chunk_index": 0, "source": "x.md"}),
              1: prose,
              2: Document(page_content="p2", metadata={"chunk_index": 2, "source": "x.md"})}

    monkeypatch.setattr(config, "NEIGHBOR_WINDOW", 1)
    monkeypatch.setattr(graph, "_chunks_by_index", lambda: lookup)

    assert graph.expand_with_neighbors([table])[0].page_content == "| a | b |"
    assert graph.expand_with_neighbors([prose])[0].page_content == "p0\np1\np2"


def test_inspect_data_excludes_whole_chunks_from_prose_stats():
    """1,773자 표가 산문 길이 분포에 섞이면 '상한 초과'가 결함처럼 보고된다."""
    import ingest

    _, blocks = ingest.load_documents()
    kinds = {b.metadata.get("kind") for b in blocks}
    assert kinds == {"diagram", "table"}
    # 통짜 청크는 전부 kind가 붙어 있어야 산문 통계에서 걸러진다
    assert all(b.metadata.get("kind") for b in blocks)
