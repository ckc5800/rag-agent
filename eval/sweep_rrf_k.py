"""RRF_K 스윕 — "표준값" 60을 실측 없이 그냥 썼던 것을 처음 검증한다.

graph.hybrid_search·kg.fused_search가 공유하는 RRF 완충 상수
(score = Σ 1/(K+rank))를 도입 당시 "표준값"이라는 이유만으로 60으로
정하고 스윕한 적이 없었다. K가 작을수록 상위 랭크 차이가 점수에 크게
반영되고(순위에 민감), K가 클수록 랭크 차이가 완만해진다(더 평평한 융합).

eval_retrieval.py와 같은 gold(retrieval_set.json, md5 고정)로 recall@k를
잰다. LLM 호출이 없어 임베딩만 쓰고 수 초 안에 끝난다.

    python eval/sweep_rrf_k.py
    python eval/sweep_rrf_k.py --values 10 30 60 100 200
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import graph  # noqa: E402

RETRIEVAL_SET = Path(__file__).parent / "retrieval_set.json"
RESULTS = Path(__file__).parent / "results_rrf_k.json"
KS = [1, 3, 6]


def hybrid_search_with_k(query: str, k_const: int):
    """graph.hybrid_search와 동일 로직, RRF_K만 인자로 받는다."""
    vectorstore, bm25 = graph._load_indexes()
    vec_docs = vectorstore.similarity_search(query, k=config.TOP_K)
    kw_docs = bm25.invoke(query)

    scores: dict[str, float] = {}
    by_key = {}
    for docs in (vec_docs, kw_docs):
        for rank, doc in enumerate(docs):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_const + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_key[k] for k in ranked[:config.TOP_K]]


def main() -> int:
    ap = argparse.ArgumentParser(description="RRF_K 스윕 (검색 단독)")
    ap.add_argument("--values", type=int, nargs="+",
                    default=[1, 5, 10, 20, 40, 60, 100, 200])
    args = ap.parse_args()

    cases = json.loads(RETRIEVAL_SET.read_text(encoding="utf-8"))
    print(f"{len(cases)}문항 · 현재 코드 값 RRF_K={config.RRF_K}\n")
    print(f"{'RRF_K':>6} {'recall@1':>9} {'recall@3':>9} {'recall@6':>9} {'MRR':>7}")

    rows = []
    for k_const in args.values:
        ranks = []
        for case in cases:
            gold_md5 = {g["md5"] for g in case["gold"]}
            docs = hybrid_search_with_k(case["question"], k_const)
            hashes = [hashlib.md5(d.page_content.encode("utf-8")).hexdigest() for d in docs]
            first = next((i + 1 for i, h in enumerate(hashes) if h in gold_md5), None)
            ranks.append(first)

        summary = {f"recall@{k}": round(
            sum(1 for r in ranks if r and r <= k) / len(cases) * 100) for k in KS}
        summary["mrr"] = round(sum(1 / r for r in ranks if r) / len(cases), 3)
        rows.append({"rrf_k": k_const, **summary})
        print(f"{k_const:>6} {summary['recall@1']:>8}% {summary['recall@3']:>8}% "
              f"{summary['recall@6']:>8}% {summary['mrr']:>7}")

    best = max(rows, key=lambda r: (r["recall@1"], r["mrr"]))
    print(f"\nrecall@1·MRR 기준 최댓값: RRF_K={best['rrf_k']} "
          f"({best['recall@1']}%, MRR {best['mrr']})")
    print(f"현재 코드 값: RRF_K={config.RRF_K}")

    RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
