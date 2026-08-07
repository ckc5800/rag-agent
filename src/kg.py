"""Graph RAG — LLM로 (주어, 관계, 목적어) 트리플을 추출해 지식 그래프를 만들고,
질의에 등장하는 엔티티를 시드로 그래프를 순회해 관련 청크를 찾는다.

동기: 이 코퍼스는 "이윤선-근무-회사", "프로젝트-사용-기술" 같은 개체 중심
관계가 많고, enumeration·comparison형 질문(여러 청크에 흩어진 사실을 모아야
하는 질문)이 실제로 8+2문항 있다. 하이브리드 검색은 청크 단위라 이런 질문에
불리할 수 있다는 것이 가설 — eval/eval_kg_retrieval.py로 검증한다.

그래프 저장소: networkx MultiDiGraph, JSON(node-link) 직렬화. 서버 없이
파일 하나로 끝난다는 점에서 FAISS·chunks.jsonl과 같은 원칙을 따른다.
"""
import json
import re
from pathlib import Path

import networkx as nx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import config
from graph import bm25_tokenize, get_llm

KG_PATH = config.BASE_DIR / "data" / "knowledge_graph.json"

# 시드 노드로 인정하는 최소 토큰 겹침 비율(노드 토큰 중 질의에도 나타난 비율).
# bm25_tokenize가 Kiwi 형태소 분석으로 조사를 실제로 떼기 때문에("인피닉과"
# → 어간 '인피닉') 조사가 붙어도 겹친다.
#
# 스윕(eval/sweep_seed_match_ratio.py, KLUE-RE·120문항, 2026-08) 결과 0.6은
# 최적이 아니었다:
#
#     ratio   평균 시드   kg_only@1   fused@1   fused@6
#     0.6     1.23        92%         83%       100%
#     0.7     0.97        93%         85%       100%
#     0.8     0.97        93%         85%       100%   (0.7과 동일 — 여기가 elbow)
#
# 0.7에서 평균 시드 수가 1.23→0.97로 줄면서(느슨한 매칭이 걷어내던 무관
# 엔티티 감소) kg_only@1·fused@1이 오히려 오른다. 0.8·0.9는 0.7과 완전히
# 같아 그 이상 올릴 이유가 없다 — 0.7이 "더 이상 엄격해져도 안 바뀌는"
# 문턱이다. 0.7로 채택.
SEED_MATCH_RATIO = 0.7


class Triple(BaseModel):
    subject: str = Field(description="주어 엔티티 — 인물/회사/프로젝트/기술 등 구체적 개체명")
    relation: str = Field(
        description="관계를 나타내는 짧은 한국어 서술어 (예: 근무했다, 사용했다, 개선했다)")
    object: str = Field(description="목적어 엔티티 또는 값")


class TripleList(BaseModel):
    triples: list[Triple] = Field(description="텍스트에서 추출한 사실 트리플. 없으면 빈 리스트")


EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "아래 텍스트에서 사실 관계를 (주어, 관계, 목적어) 트리플로 추출하세요.\n"
    "인물·회사·프로젝트·기술·수치 등 구체적인 개체를 주어/목적어로 삼으세요.\n"
    "같은 개체는 일관된 명칭으로 표기하세요(예: '이윤선님'이 아니라 '이윤선').\n"
    "텍스트에 명시되지 않은 내용은 추론해 만들지 마세요. 최대 8개.\n\n"
    "텍스트:\n{text}"
)

_extract_llm_cache: dict[str, object] = {}


def _extract_llm():
    model = config.LLM_MODEL
    if model not in _extract_llm_cache:
        _extract_llm_cache[model] = get_llm().with_structured_output(
            TripleList, include_raw=True)
    return _extract_llm_cache[model]


def extract_triples(text: str) -> list[Triple]:
    """청크 하나에서 트리플을 뽑는다. 파싱 실패면 빈 리스트(fail-open) —
    grade·rerank와 같은 정책. 실패율은 ingest_kg.py가 세어 보고한다."""
    chain = EXTRACT_PROMPT | _extract_llm()
    result = chain.invoke({"text": text})
    parsed = result.get("parsed")
    return parsed.triples if parsed else []


def normalize(name: str) -> str:
    """노드 병합 키. 앞뒤·중복 공백 정리 + ASCII는 소문자화(대소문자 표기
    흔들림 병합), 한글은 대소문자가 없어 그대로 둔다."""
    s = re.sub(r"\s+", " ", name.strip())
    return s.lower() if s.isascii() else s


def add_triples(g: nx.MultiDiGraph, triples: list[Triple],
                chunk_index: int | None, source: str | None) -> int:
    """트리플들을 그래프에 엣지로 추가한다. 추가된 엣지 수를 반환한다.
    LLM 호출 없는 순수 함수라 단위 테스트 대상이다."""
    added = 0
    for t in triples:
        s, o = normalize(t.subject), normalize(t.object)
        if not s or not o:
            continue
        g.add_node(s, label=t.subject.strip())
        g.add_node(o, label=t.object.strip())
        g.add_edge(s, o, relation=t.relation.strip(),
                   chunk_index=chunk_index, source=source)
        added += 1
    return added


def build_graph(chunks: list[Document]) -> tuple[nx.MultiDiGraph, dict]:
    """청크마다 트리플을 추출(LLM)해 그래프를 짓는다. (그래프, 추출 통계) 반환.

    다이어그램 청크(kind=diagram)는 산문이 아니라 건너뛴다 — 트리플 추출은
    문장 단위 사실 관계를 전제하는데 박스 그림에는 없다.
    """
    g = nx.MultiDiGraph()
    stats = {"chunks_seen": 0, "chunks_empty": 0, "triples": 0}
    for c in chunks:
        if c.metadata.get("kind") == "diagram":
            continue
        stats["chunks_seen"] += 1
        triples = extract_triples(c.page_content)
        if not triples:
            stats["chunks_empty"] += 1
        stats["triples"] += add_triples(
            g, triples, c.metadata.get("chunk_index"), c.metadata.get("source"))
    return g, stats


def save(g: nx.MultiDiGraph, path: Path = KG_PATH) -> None:
    data = nx.node_link_data(g, edges="edges")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path = KG_PATH) -> nx.MultiDiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges")


# ── 검색 ────────────────────────────────────────────────

_chunks_cache: dict[int, Document] | None = None


def _chunks_by_index() -> dict[int, Document]:
    """chunk_index → Document. graph.py의 이웃 확장용 캐시와 별개로 둔다 —
    kg.py를 독립 실험으로 유지한다(그래프 모듈 내부 캐시 무효화 규칙에
    얽매이지 않는다)."""
    global _chunks_cache
    if _chunks_cache is None:
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
        _chunks_cache = table
    return _chunks_cache


def seed_nodes(query: str, g: nx.MultiDiGraph) -> list[str]:
    """질의 토큰과 SEED_MATCH_RATIO 이상 겹치는 노드를 시드로 고른다.
    LLM 없이 bm25_tokenize만 쓴다 — eval_retrieval.py처럼 검색 평가는
    LLM 없이 수 초 안에 끝나야 한다는 이 프로젝트의 원칙을 따른다."""
    q_tokens = set(bm25_tokenize(query))
    if not q_tokens:
        return []
    seeds = []
    for node in g.nodes:
        n_tokens = set(bm25_tokenize(node))
        if not n_tokens:
            continue
        overlap = len(n_tokens & q_tokens) / len(n_tokens)
        if overlap >= SEED_MATCH_RATIO:
            seeds.append(node)
    return seeds


def search(query: str, g: nx.MultiDiGraph, k: int,
           chunks_by_index: dict[int, Document] | None = None) -> list[Document]:
    """질의 엔티티를 시드로 1-hop 이웃 엣지를 모아, 그 출처 청크를 점수순으로
    반환한다. 점수 = 시드에 연결된 엣지 수 — 시드와 관련된 사실을 더 많이
    담은 청크일수록 위로 온다. 시드가 하나도 안 잡히면 빈 리스트(그래프가
    포기하고 하이브리드에 넘기는 신호 — fused_search가 이걸로 자동 처리).

    chunks_by_index를 주면 그걸 쓰고, 안 주면 원 코퍼스(config.CHUNKS_PATH)
    조회표를 쓴다 — klue_re.py처럼 다른 코퍼스의 그래프를 검색할 때
    이 모듈의 원 코퍼스 캐시와 섞이지 않게 하려고 둔 확장 지점이다.
    """
    seeds = seed_nodes(query, g)
    if not seeds:
        return []

    chunk_scores: dict[int, float] = {}
    for s in seeds:
        for _, _, data in list(g.out_edges(s, data=True)) + list(g.in_edges(s, data=True)):
            idx = data.get("chunk_index")
            if idx is None:
                continue
            chunk_scores[idx] = chunk_scores.get(idx, 0.0) + 1.0

    chunks_by_idx = chunks_by_index if chunks_by_index is not None else _chunks_by_index()
    ranked = sorted(chunk_scores, key=chunk_scores.get, reverse=True)
    return [chunks_by_idx[i] for i in ranked[:k] if i in chunks_by_idx]


def fused_search(query: str, g: nx.MultiDiGraph, k: int, hybrid_search_fn=None,
                 chunks_by_index: dict[int, Document] | None = None) -> list[Document]:
    """하이브리드 검색(벡터+BM25)과 그래프 검색을 RRF로 융합.
    graph.hybrid_search와 같은 RRF(K=60) — 이미 검증된 융합 방식을 재사용.

    hybrid_search_fn을 안 주면 원 코퍼스의 graph.hybrid_search를 쓴다.
    klue_re.py처럼 다른 코퍼스 인덱스로 융합할 때는 그 코퍼스의
    hybrid_search 함수를 주입한다(chunks_by_index도 마찬가지).
    """
    import hashlib

    if hybrid_search_fn is None:
        from graph import hybrid_search as hybrid_search_fn

    hy_docs = hybrid_search_fn(query)
    kg_docs = search(query, g, k, chunks_by_index=chunks_by_index)

    K = 60
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in (hy_docs, kg_docs):
        for rank, doc in enumerate(docs):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_key[key] for key in ranked[:k]]
