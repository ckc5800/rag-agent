"""TOP_K 스윕 — 하이브리드 검색이 실제로 융합할 여지가 있는가.

`TOP_K`는 각 검색기가 내놓는 후보 수이자 RRF 융합 결과의 길이다. 지금 6인데
`GENERATE_TOP_N`이 5다. **후보 6개 중 5개를 쓰는 구조라 융합이 거의 무의미하다**
— 두 검색기가 뭘 내놓든 대부분 그대로 통과한다. RRF의 요점은 "한쪽에서만
상위인 문서를 끌어올리는 것"인데, 창이 좁으면 끌어올릴 것도 버릴 것도 없다.

후보를 넉넉히 뽑고 상위 N개만 쓰는 것이 원래 의도다. 어디까지 올리면 되는지
잰다. **LLM을 쓰지 않아** 임베딩 호출만 있고 수십 초면 끝난다.

    python eval/sweep_top_k.py
    python eval/sweep_top_k.py --values 6 12 24
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import graph  # noqa: E402
from runmeta import run_metadata  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results_top_k.json"


def rank_of(docs, anchors: list[str]) -> int | None:
    for i, d in enumerate(docs, 1):
        if any(a in d.page_content for a in anchors):
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="TOP_K 스윕 (검색 단독)")
    ap.add_argument("--values", type=int, nargs="+",
                    default=[6, 10, 15, 20, 30])
    args = ap.parse_args()

    cases = [c for c in json.loads(EVAL_SET.read_text(encoding="utf-8"))
             if c.get("gold_anchors")]
    n = len(cases)
    store, bm25 = graph._load_indexes()
    gen_n = config.GENERATE_TOP_N

    print(f"{n}문항 · GENERATE_TOP_N={gen_n} (generate가 실제로 보는 개수)\n")
    print(f"{'TOP_K':>6} {'recall@1':>9} {'recall@3':>9} "
          f"{f'recall@{gen_n}':>10} {'vec 단독':>9} {'bm25 단독':>10}")

    rows = []
    for k in args.values:
        config.TOP_K = k
        bm25.k = k                      # 인덱스 빌드 시점 값이라 직접 갱신해야 한다

        hyb, vec, kw = [], [], []
        for c in cases:
            a = c["gold_anchors"]
            hyb.append(rank_of(graph.hybrid_search(c["question"]), a))
            vec.append(rank_of(store.similarity_search(c["question"], k=k), a))
            kw.append(rank_of(bm25.invoke(c["question"]), a))

        def rec(ranks, at):
            return sum(1 for r in ranks if r and r <= at) / n * 100

        row = {"top_k": k,
               "recall@1": round(rec(hyb, 1)), "recall@3": round(rec(hyb, 3)),
               f"recall@{gen_n}": round(rec(hyb, gen_n)),
               "vector_at_gen_n": round(rec(vec, gen_n)),
               "bm25_at_gen_n": round(rec(kw, gen_n))}
        rows.append(row)
        print(f"{k:>6} {row['recall@1']:>8}% {row['recall@3']:>8}% "
              f"{row[f'recall@{gen_n}']:>9}% {row['vector_at_gen_n']:>8}% "
              f"{row['bm25_at_gen_n']:>9}%")

    best = max(rows, key=lambda r: r[f"recall@{gen_n}"])
    print(f"\ngenerate가 보는 범위(recall@{gen_n}) 최댓값: "
          f"TOP_K={best['top_k']} 에서 {best[f'recall@{gen_n}']}%")
    print("vec/bm25 단독 열은 같은 창에서 각 검색기만 썼을 때다 — 하이브리드가"
          "\n이 둘보다 높아야 융합이 값을 하는 것이다.")
    print(f"\n주의: 후보를 늘려도 generate가 받는 건 {gen_n}개로 동일하므로"
          "\n      컨텍스트 길이·지연은 변하지 않는다. 순위만 좋아진다.")

    # 이 스크립트는 루프에서 config.TOP_K를 바꾸므로 지문의 params.TOP_K는
    # **마지막 스텝** 값이다. 스윕한 값 자체는 rows에 들어 있다.
    RESULTS.write_text(json.dumps({"env": run_metadata(), "rows": rows},
                                  ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
