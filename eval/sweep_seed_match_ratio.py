"""SEED_MATCH_RATIO 스윕 — Graph RAG 시드 판정 임계값이 실제로 최적인가.

kg.py의 SEED_MATCH_RATIO(=0.6)는 도입 당시 "스윕은 안 했고, RRF 융합이
이 실험의 핵심이라 민감도는 후순위로 미뤘다"고 명시적으로 남겨둔 값이다.
Graph RAG가 실제로 채택된 건 KLUE-RE 코퍼스(README "Graph RAG 실험 2")라
이 스윕도 거기서 돌린다 — 원 코퍼스는 애초에 Graph RAG가 미채택이라 임계값을
바꿔도 결론에 영향이 없다.

가설: 낮추면(느슨한 매칭) 시드가 더 잡혀 kg_only recall이 오르되 무관한
엔티티도 섞여 정밀도가 떨어질 수 있고, 높이면(엄격한 매칭) 반대가 된다.
LLM 호출 없이 그래프 순회만 하므로 결정적이고 수 초 안에 끝난다.

    python eval/sweep_seed_match_ratio.py
    python eval/sweep_seed_match_ratio.py --values 0.3 0.5 0.7 0.9
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_ollama import OllamaEmbeddings  # noqa: E402

import config  # noqa: E402
import kg  # noqa: E402
from runmeta import run_metadata  # noqa: E402
import klue_re  # noqa: E402
from eval_klue_retrieval import score  # noqa: E402
from graph import bm25_tokenize  # noqa: E402

RESULTS = Path(__file__).parent / "results_seed_match_ratio.json"


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


def main() -> int:
    ap = argparse.ArgumentParser(description="SEED_MATCH_RATIO 스윕 (KLUE-RE, 검색 단독)")
    ap.add_argument("--values", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()

    cases = json.loads(klue_re.KLUE_RETRIEVAL_SET.read_text(encoding="utf-8"))
    g = kg.load(klue_re.KLUE_GRAPH_PATH)
    store, bm25, chunks_by_index = load_klue_index()
    # LLM은 안 쓰지만 임베딩 모델이 융합 결과를 좌우하므로 함께 남긴다.
    # (n_chunks·index_chunks_md5는 **원 코퍼스** 지문이라 KLUE 실험과는 무관 —
    #  여기서 의미 있는 건 embed_model·host 쪽이다.)
    env = run_metadata()
    print(f"그래프: 노드 {g.number_of_nodes()}개, 엣지 {g.number_of_edges()}개, "
          f"질문 {len(cases)}건 · embed={env['embed_model']}\n")

    def hy(q):
        return klue_re.hybrid_search(q, store, bm25, config.TOP_K)

    original_ratio = kg.SEED_MATCH_RATIO
    rows = []
    try:
        print(f"{'ratio':>6} {'평균 시드 수':>10} {'kg_only@1':>10} {'kg_only@3':>10} "
              f"{'kg_only@6':>10} {'kg MRR':>8}   {'fused@1':>9} {'fused@6':>9} {'fused MRR':>10}")
        for ratio in args.values:
            kg.SEED_MATCH_RATIO = ratio

            seed_counts = [len(kg.seed_nodes(c["question"], g)) for c in cases]
            avg_seeds = sum(seed_counts) / len(seed_counts)

            def kgo(q):
                return kg.search(q, g, config.TOP_K, chunks_by_index=chunks_by_index)

            def fused(q):
                return kg.fused_search(q, g, config.TOP_K, hybrid_search_fn=hy,
                                       chunks_by_index=chunks_by_index)

            _, kg_summary = score(kgo, cases, f"kg_only ratio={ratio}")
            _, fused_summary = score(fused, cases, f"fused ratio={ratio}")

            row = {"ratio": ratio, "avg_seeds_per_question": round(avg_seeds, 2),
                   "kg_only": kg_summary, "fused": fused_summary}
            rows.append(row)
            print(f"{ratio:>6} {avg_seeds:>10.2f} "
                  f"{kg_summary['recall@1']:>9}% {kg_summary['recall@3']:>9}% "
                  f"{kg_summary['recall@6']:>9}% {kg_summary['mrr']:>8} "
                  f"  {fused_summary['recall@1']:>8}% {fused_summary['recall@6']:>8}% "
                  f"{fused_summary['mrr']:>10}")
    finally:
        kg.SEED_MATCH_RATIO = original_ratio      # 다른 스크립트에 영향 없게 원복

    best_fused = max(rows, key=lambda r: (r["fused"]["recall@6"], r["fused"]["mrr"]))
    print(f"\nfused recall@6 최댓값: ratio={best_fused['ratio']} "
          f"({best_fused['fused']['recall@6']}%, MRR {best_fused['fused']['mrr']})")
    print(f"현재 코드 값: SEED_MATCH_RATIO={original_ratio}")

    RESULTS.write_text(json.dumps({"env": env, "corpus": "klue-re", "rows": rows},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
