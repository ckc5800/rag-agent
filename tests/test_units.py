"""결정적 로직 단위 테스트 — LLM/인덱스 없이 CI에서 돈다."""
import sys
from pathlib import Path

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
