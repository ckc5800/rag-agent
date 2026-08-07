"""NEIGHBOR_WINDOW A/B — 반복 재측정 (config.py에 명시된 미완 TODO).

첫 측정(51문항 × 2회를 두 번, GENERATE_TOP_N=3 조건)은 전체 정답률에서
재현되지 않았다(75%→80%, 74%→75%, 편차 ±6%p 안). 다만 두 실행 모두
같은 방향(aggregation·temporal 개선)이었다 — "반복을 늘려 다시 재는 것이
다음 순서"로 남겨 둔 TODO를 여기서 마무리한다. 지금은 GENERATE_TOP_N=5가
기본이라 그 조건에서 다시 잰다(예전 측정은 TOP_N=3 시절).

    python eval/ab_neighbor_window.py                # 0 vs 1, 각 2회
    python eval/ab_neighbor_window.py --repeat 3
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from evaluate import EVAL_SET, is_pass  # noqa: E402

RESULTS = Path(__file__).parent / "results_ab_neighbor_window.json"


def run_condition(cases: list[dict], window: int, repeat: int) -> dict:
    from graph import build_graph

    # TYPE_ROUTING이 기본 켜져 있어(2026-08~) aggregation 질문은 실행 중
    # route.ROUTES가 NEIGHBOR_WINDOW를 1로 강제 오버라이드한다 — 여기서
    # 재는 건 "전역 기본값을 얼마로 할까"이지 라우팅 자체가 아니므로,
    # 오버라이드가 이 전역값 스윕을 덮어쓰지 않도록 끈다.
    config.TYPE_ROUTING = False
    config.NEIGHBOR_WINDOW = window
    graph = build_graph()

    tally, secs = Counter(), []
    for i in range(repeat):
        for case in cases:
            q = case["question"]
            t0 = time.time()
            out = graph.invoke({"question": q, "query": q, "rewrites": 0})
            secs.append(time.time() - t0)
            tally[q] += is_pass(out["answer"], case)
        print(f"    window={window} {i + 1}/{repeat} 완료", flush=True)
    return {"tally": tally, "median_sec": sorted(secs)[len(secs) // 2]}


def main() -> int:
    ap = argparse.ArgumentParser(description="NEIGHBOR_WINDOW A/B")
    ap.add_argument("--values", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    n = len(cases) * args.repeat
    print(f"{len(cases)}문항 × {len(args.values)}조건 × {args.repeat}회 "
          f"(GENERATE_TOP_N={config.GENERATE_TOP_N})\n")

    out = {}
    for v in args.values:
        print(f"── NEIGHBOR_WINDOW = {v} ──")
        out[v] = run_condition(cases, v, args.repeat)

    print(f"\n{'유형':<12} " + "".join(f"W={v:<8}" for v in args.values))
    types = sorted({c.get("type", "fact") for c in cases})
    for t in types:
        qs = [c["question"] for c in cases if c.get("type", "fact") == t]
        row = f"{t:<12} "
        for v in args.values:
            hit = sum(out[v]["tally"][q] for q in qs)
            row += f"{hit:>3}/{len(qs) * args.repeat:<9}"
        print(row)

    print()
    for v in args.values:
        total = sum(out[v]["tally"].values())
        print(f"W={v}: 정답률 {total / n * 100:.0f}% ({total}/{n}) · "
              f"지연 중앙값 {out[v]['median_sec']:.1f}s")

    if len(args.values) == 2:
        a, b = args.values
        print(f"\nW={a} → W={b} 에서 뒤집힌 문항:")
        for c in cases:
            q = c["question"]
            d = out[b]["tally"][q] - out[a]["tally"][q]
            if d:
                print(f"  {'↑' if d > 0 else '↓'} ({c.get('type')}) "
                      f"{out[a]['tally'][q]}/{args.repeat} → "
                      f"{out[b]['tally'][q]}/{args.repeat}  {q[:44]}")

    RESULTS.write_text(json.dumps(
        {"repeat": args.repeat, "generate_top_n": config.GENERATE_TOP_N,
         "conditions": {str(v): {"accuracy": round(
             sum(out[v]["tally"].values()) / n * 100, 1),
             "median_sec": round(out[v]["median_sec"], 2),
             "per_question": dict(out[v]["tally"])} for v in args.values}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
