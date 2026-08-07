"""KLUE-RE 2차 코퍼스에서 Graph RAG 재검증 — 하이브리드 vs 그래프 vs 융합.

원 코퍼스(이력서 등, 48청크)에서는 하이브리드가 이미 recall@1 90%로 천장이라
Graph RAG가 이길 여지가 없었다. 이 코퍼스는 문장 1,500개·정답 라벨 그래프
(LLM 추출 노이즈 없음)로 그 결론이 "코퍼스가 작아서"였는지 "그래프 검색
자체가 이 방식으로는 안 맞아서"였는지를 가른다. LLM 호출이 없어(그래프는
이미 ingest_klue.py가 정답 라벨로 만들어 둠) 결정적이고 수 초 안에 끝난다.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402

import config  # noqa: E402
import kg  # noqa: E402
import klue_re  # noqa: E402
from graph import bm25_tokenize  # noqa: E402

RESULTS = Path(__file__).parent / "results_klue_retrieval.json"
KS = [1, 3, 6]


def score(docs_fn, cases: list[dict], label: str) -> tuple[list[dict], dict]:
    rows = []
    for case in cases:
        gold_md5 = {g["md5"] for g in case["gold"]}
        docs = docs_fn(case["question"])
        ranks = [hashlib.md5(d.page_content.encode("utf-8")).hexdigest() for d in docs]
        first = next((i + 1 for i, h in enumerate(ranks) if h in gold_md5), None)
        rows.append({
            "question": case["question"], "relation": case["relation"], "gold_rank": first,
            "hits": {f"@{k}": (first is not None and first <= k) for k in KS},
        })
    summary = {f"recall@{k}": round(
        sum(r["hits"][f"@{k}"] for r in rows) / len(rows) * 100) for k in KS}
    hit_ranks = [r["gold_rank"] for r in rows if r["gold_rank"]]
    summary["mrr"] = round(sum(1 / r for r in hit_ranks) / len(rows), 3)
    print(f"\n[{label}] " + "  ".join(f"{k}={v}" for k, v in summary.items()))
    return rows, summary


def by_relation(rows: list[dict]) -> dict[str, str]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        buckets[r["relation"]].append(r["hits"]["@6"])
    return {t: f"{sum(v)}/{len(v)}" for t, v in sorted(buckets.items())}


def load_klue_index():
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    store = FAISS.load_local(klue_re.KLUE_DB_DIR, embeddings,
                             allow_dangerous_deserialization=True)
    chunks = []
    with open(klue_re.KLUE_CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Document(page_content=d["page_content"], metadata=d["metadata"]))
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=bm25_tokenize)
    bm25.k = config.TOP_K
    return store, bm25, {c.metadata["chunk_index"]: c for c in chunks}


def main():
    cases = json.loads(klue_re.KLUE_RETRIEVAL_SET.read_text(encoding="utf-8"))
    g = kg.load(klue_re.KLUE_GRAPH_PATH)
    store, bm25, chunks_by_index = load_klue_index()
    print(f"그래프: 노드 {g.number_of_nodes()}개, 엣지 {g.number_of_edges()}개, "
          f"청크 {len(chunks_by_index)}개, 질문 {len(cases)}건")

    def hy(q):
        return klue_re.hybrid_search(q, store, bm25, config.TOP_K)

    def kgo(q):
        return kg.search(q, g, config.TOP_K, chunks_by_index=chunks_by_index)

    def fused(q):
        return kg.fused_search(q, g, config.TOP_K, hybrid_search_fn=hy,
                               chunks_by_index=chunks_by_index)

    results = {}
    for label, fn in (("hybrid", hy), ("kg_only", kgo), ("hybrid+kg_rrf", fused)):
        rows, summary = score(fn, cases, label)
        summary["by_relation_hit@6"] = by_relation(rows)
        results[label] = summary

    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")


if __name__ == "__main__":
    main()
