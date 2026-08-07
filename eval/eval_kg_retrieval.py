"""Graph RAG 검색 평가 — 하이브리드 vs 그래프 단독 vs 융합(RRF).

eval_retrieval.py와 같은 gold(retrieval_set.json, md5 고정)로 recall@k/MRR을
재고, eval_set.json의 질문 유형과 조인해 유형별로도 쪼갠다.

가설: enumeration·comparison형(여러 청크에 흩어진 사실을 모아야 하는 질문 —
"근무한 회사들을 알려주세요", "인피닉과 이든티앤에스 중 더 오래 근무한 곳은")
에서 그래프가 하이브리드보다 나을 것이다. 여러 청크에 흩어진 회사·기간 정보가
그래프에서는 엔티티 하나로 모이기 때문이다. fact·refusal형은 청크 하나로
끝나는 단순 조회라 차이가 없거나 그래프가 오히려 못할 것으로 예상한다.
LLM 호출이 없어 결정적이고 수 초 안에 끝난다(그래프 자체는 ingest_kg.py가
미리 만들어 둔 것을 로드만 한다).
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import kg  # noqa: E402
from graph import hybrid_search  # noqa: E402

RETRIEVAL_SET = Path(__file__).parent / "retrieval_set.json"
EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results_kg_retrieval.json"
KS = [1, 3, 6]  # 6 = config.TOP_K


def load_types() -> dict[str, str]:
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    return {q["question"]: q.get("type", "?") for q in data}


def score(docs_fn, cases: list[dict], label: str) -> tuple[list[dict], dict]:
    rows = []
    for case in cases:
        gold_md5 = {g["md5"] for g in case["gold"]}
        docs = docs_fn(case["question"])
        ranks = [hashlib.md5(d.page_content.encode("utf-8")).hexdigest() for d in docs]
        first = next((i + 1 for i, h in enumerate(ranks) if h in gold_md5), None)
        rows.append({
            "question": case["question"], "gold_rank": first,
            "hits": {f"@{k}": (first is not None and first <= k) for k in KS},
        })
    summary = {f"recall@{k}": round(
        sum(r["hits"][f"@{k}"] for r in rows) / len(rows) * 100) for k in KS}
    hit_ranks = [r["gold_rank"] for r in rows if r["gold_rank"]]
    summary["mrr"] = round(sum(1 / r for r in hit_ranks) / len(rows), 3)
    print(f"\n[{label}] " + "  ".join(f"{k}={v}" for k, v in summary.items()))
    return rows, summary


def by_type(rows: list[dict], types: dict[str, str]) -> dict[str, str]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        buckets[types.get(r["question"], "?")].append(r["hits"]["@6"])
    return {t: f"{sum(v)}/{len(v)}" for t, v in sorted(buckets.items())}


def main():
    cases = json.loads(RETRIEVAL_SET.read_text(encoding="utf-8"))
    types = load_types()
    g = kg.load()
    print(f"그래프: 노드 {g.number_of_nodes()}개, 엣지 {g.number_of_edges()}개, "
          f"질문 {len(cases)}건")

    results = {}
    for label, fn in (
        ("hybrid", hybrid_search),
        ("kg_only", lambda q: kg.search(q, g, config.TOP_K)),
        ("hybrid+kg_rrf", lambda q: kg.fused_search(q, g, config.TOP_K)),
    ):
        rows, summary = score(fn, cases, label)
        summary["by_type_hit@6"] = by_type(rows, types)
        print(f"  유형별 hit@6: {summary['by_type_hit@6']}")
        results[label] = summary

    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")


if __name__ == "__main__":
    main()
