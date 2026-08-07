"""Graph RAG 결정적 로직 단위 테스트 — LLM/인덱스 없이 CI에서 돈다.
(트리플 추출 자체는 LLM 호출이라 여기서 검증하지 않는다. add_triples 이후
그래프 연산과, bm25_tokenize 기반 시드 매칭·랭킹만 순수 함수로 검증한다.)
"""
import sys
from pathlib import Path

import networkx as nx
import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import kg  # noqa: E402
from kg import Triple, add_triples, normalize, search, seed_nodes  # noqa: E402


@pytest.fixture(autouse=True)
def fake_chunk_index(monkeypatch):
    """search()가 실제 data/chunks.jsonl 대신 이 가짜 맵을 쓰게 한다 —
    단위 테스트가 코퍼스 내용에 의존하지 않도록."""
    fake = {
        1: Document(page_content="인피닉 근무 이력", metadata={"chunk_index": 1}),
        2: Document(page_content="이든티앤에스 근무·위치", metadata={"chunk_index": 2}),
    }
    monkeypatch.setattr(kg, "_chunks_cache", fake)
    yield


def test_normalize_merges_ascii_case_and_whitespace():
    assert normalize("FastAPI") == normalize("fastapi") == "fastapi"
    assert normalize("  이윤선  님 ") == "이윤선 님"  # 한글은 대소문자 없음, 공백만 정리


def test_add_triples_builds_nodes_and_edges():
    g = nx.MultiDiGraph()
    triples = [Triple(subject="이윤선", relation="근무했다", object="MiCo AI")]
    added = add_triples(g, triples, chunk_index=3, source="resume.md")
    assert added == 1
    assert g.has_edge("이윤선", "mico ai")  # object가 ASCII 포함이라 소문자 정규화
    edge = g.get_edge_data("이윤선", "mico ai")[0]
    assert edge["relation"] == "근무했다" and edge["chunk_index"] == 3


def test_add_triples_skips_empty_entities():
    g = nx.MultiDiGraph()
    triples = [Triple(subject="  ", relation="x", object="회사")]
    assert add_triples(g, triples, 0, "a.md") == 0
    assert g.number_of_nodes() == 0


def _graph_with_two_facts():
    g = nx.MultiDiGraph()
    add_triples(g, [Triple(subject="이윤선", relation="근무했다", object="인피닉")],
               chunk_index=1, source="resume.md")
    add_triples(g, [Triple(subject="이윤선", relation="근무했다", object="이든티앤에스")],
               chunk_index=2, source="resume.md")
    add_triples(g, [Triple(subject="이든티앤에스", relation="위치", object="서울")],
               chunk_index=2, source="resume.md")
    return g


def test_seed_nodes_matches_despite_particle_suffix():
    g = _graph_with_two_facts()
    # "인피닉과"처럼 조사가 붙어도 bm25_tokenize의 한글 bigram으로 겹친다
    seeds = seed_nodes("인피닉과 이든티앤에스 중 어디가 오래됐나요", g)
    assert "인피닉" in seeds
    assert "이든티앤에스" in seeds


def test_seed_nodes_empty_when_unrelated_query():
    g = _graph_with_two_facts()
    assert seed_nodes("오늘 날씨 어때요", g) == []


def test_search_ranks_chunk_with_more_seed_edges_first():
    g = _graph_with_two_facts()
    # "이든티앤에스"가 시드로 잡히면 chunk_index=2가 엣지 2개(근무했다+위치)로
    # chunk_index=1(엣지 없음, 이든티앤에스와 무관)보다 위에 와야 한다
    docs = search("이든티앤에스는 어디에 있나요", g, k=5)
    assert docs, "시드가 있는데 결과가 비어 있음"
    assert docs[0].metadata["chunk_index"] == 2


def test_search_returns_empty_without_seed():
    g = _graph_with_two_facts()
    assert search("전혀 관련 없는 질문입니다", g, k=5) == []
