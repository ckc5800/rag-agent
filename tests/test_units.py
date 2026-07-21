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
