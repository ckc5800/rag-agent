"""노드별 지연 프로파일 — corrective 루프가 비용을 얼마나 쓰는가.

이 저장소는 recall·정답률·판정기 정확도를 다 쟀지만 **시간 배분은 안 쟀다.**
그래서 지금까지 못 하던 계산이 하나 있다:

    grade 판정기 정확도 87%, 오탐 20%(= 충분한데 재검색으로 보냄)  ← 이미 측정
    corrective 루프가 구제한 문항 10개 중 1개                      ← 이미 측정
    corrective 루프가 먹는 지연                                    ← 없었다

세 번째가 있어야 "grade를 유지할 값이 있나"를 따질 수 있다. 이 스크립트가
그 숫자를 만든다. graph.timed가 State에 남기는 노드별 시간을 평가셋 전체에
대해 모은다.

**프로덕션 경로(graph.run)로 돈다** — 유형 라우팅까지 포함한 실제 비용을
재야 의미가 있다. 이 저장소는 경로를 바꿨더니 결론의 부호가 뒤집힌 적이
있다(ab_rewrite).

    python eval/profile_nodes.py                # 평가셋 전체 1회
    python eval/profile_nodes.py --limit 10     # 앞 10문항만
    python eval/profile_nodes.py --repeat 3     # 편차를 보려면
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from evaluate import EVAL_SET, is_pass  # noqa: E402
from runmeta import run_metadata  # noqa: E402

RESULTS = Path(__file__).parent / "results_node_profile.json"

# corrective 루프에 속하는 노드. 재작성이 발동하면 retrieve가 한 번 더 도는데,
# timings는 누적이라 2회차 검색 비용이 retrieve 안에 섞여 있다 — 그래서 루프
# 비용은 "재작성이 발동한 질의"와 아닌 질의를 갈라서 봐야 정확하다.
LOOP_NODES = ("grade", "rewrite")


def summarize(rows: list[dict]) -> dict:
    """질의별 timings 목록 → 노드별 통계와 루프 비용."""
    nodes = sorted({n for r in rows for n in r["timings"]})
    per_node = {
        n: sorted(r["timings"].get(n, 0.0) for r in rows) for n in nodes}
    total = sum(sum(v) for v in per_node.values())

    stats = {}
    for n, vals in per_node.items():
        s = sum(vals)
        stats[n] = {
            "total_sec": round(s, 2),
            "share_pct": round(s / total * 100, 1) if total else 0.0,
            "median_sec": round(statistics.median(vals), 3),
        }

    rewrote = [r for r in rows if r["rewrites"] > 0]
    plain = [r for r in rows if r["rewrites"] == 0]

    def med_total(rs):
        return round(statistics.median([sum(r["timings"].values()) for r in rs]), 2) \
            if rs else None

    return {
        "n_queries": len(rows),
        "nodes": stats,
        "loop_share_pct": round(
            sum(stats[n]["share_pct"] for n in LOOP_NODES if n in stats), 1),
        "rewrite_rate_pct": round(len(rewrote) / len(rows) * 100, 1),
        "median_total_sec": med_total(rows),
        "median_total_when_rewrote": med_total(rewrote),
        "median_total_when_not": med_total(plain),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="노드별 지연 프로파일")
    ap.add_argument("--limit", type=int, default=None, help="앞 N문항만")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    from graph import build_graph, run

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]

    graph = build_graph()
    rows = []
    print(f"{len(cases)}문항 × {args.repeat}회 · 프로덕션 경로(graph.run)\n")
    for i in range(args.repeat):
        for case in cases:
            q = case["question"]
            t0 = time.time()
            state = run(q, graph=graph)
            rows.append({
                "question": q,
                "type": case.get("type", "fact"),
                "timings": state.get("timings") or {},
                "rewrites": state.get("rewrites", 0),
                "pass": is_pass(state["answer"], case),
                "wall_sec": round(time.time() - t0, 2),
            })
        print(f"  {i + 1}/{args.repeat} 완료", flush=True)

    s = summarize(rows)

    print(f"\n===== 노드별 배분 ({s['n_queries']}회 질의) =====")
    for n, st in sorted(s["nodes"].items(), key=lambda kv: -kv[1]["share_pct"]):
        print(f"  {n:16} {st['total_sec']:>8.2f}s  {st['share_pct']:>5.1f}%  "
              f"(중앙값 {st['median_sec']:.3f}s)")

    print(f"\n===== corrective 루프 비용 =====")
    print(f"  grade+rewrite 지분   {s['loop_share_pct']}%")
    print(f"  재작성 발생률        {s['rewrite_rate_pct']}%")
    print(f"  질의 지연 중앙값     전체 {s['median_total_sec']}s")
    if s["median_total_when_rewrote"] is not None:
        print(f"                       재작성 O {s['median_total_when_rewrote']}s"
              f" / 재작성 X {s['median_total_when_not']}s")
    print("\n  해석 지침: grade는 정확도 87%·오탐 20%로 이미 측정돼 있고,"
          "\n  corrective 루프는 10문항 중 1문항을 구제했다(ab_rewrite)."
          "\n  여기 나온 비용과 함께 놓고 유지 여부를 판단할 것 —"
          "\n  MAX_REWRITES=0으로 두면 grade·rewrite 노드가 통째로 빠진다.")

    RESULTS.write_text(json.dumps({"env": run_metadata(), "summary": s,
                                   "rows": rows}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
