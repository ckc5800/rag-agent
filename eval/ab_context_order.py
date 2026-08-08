"""CONTEXT_ORDER A/B — 반복 재측정 (config.py에 명시된 미완 TODO).

첫 측정(51문항 × 2회, 단일 실행)은 역순 74% / 정순 74% / 샌드위치 78%로
샌드위치가 앞섰지만 편차 ±6%p 안이었고 비교 유형이 반토막(comparison
역순 4/4→샌드위치 2/4 악화, enumeration 7/16→12/16 개선)이라 결론을 못
냈다. "반복을 늘려 다시 재는 것이 다음 순서"로 남겨 둔 TODO를 마무리한다.

    python eval/ab_context_order.py                     # reversed vs sandwich, 각 2회
    python eval/ab_context_order.py --values reversed ranked sandwich --repeat 3
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
from runmeta import run_metadata  # noqa: E402

RESULTS = Path(__file__).parent / "results_ab_context_order.json"


def run_condition(cases: list[dict], order: str, repeat: int) -> dict:
    from graph import build_graph

    # TYPE_ROUTING이 기본 켜져 있어(2026-08~) enumeration 질문은 실행 중
    # route.ROUTES가 CONTEXT_ORDER를 sandwich로 강제 오버라이드한다 — 여기서
    # 재는 건 "전역 기본값을 뭘로 할까"이지 라우팅 자체가 아니므로,
    # 오버라이드가 이 전역값 스윕을 덮어쓰지 않도록 끈다.
    config.TYPE_ROUTING = False
    config.CONTEXT_ORDER = order
    graph = build_graph()

    tally, secs = Counter(), []
    for i in range(repeat):
        for case in cases:
            q = case["question"]
            t0 = time.time()
            out = graph.invoke({"question": q, "query": q, "rewrites": 0})
            secs.append(time.time() - t0)
            tally[q] += is_pass(out["answer"], case)
        print(f"    order={order} {i + 1}/{repeat} 완료", flush=True)
    return {"tally": tally, "median_sec": sorted(secs)[len(secs) // 2]}


def main() -> int:
    ap = argparse.ArgumentParser(description="CONTEXT_ORDER A/B")
    ap.add_argument("--values", nargs="+", default=["reversed", "sandwich"])
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    n = len(cases) * args.repeat
    # 환경 지문은 **조건 루프가 config를 변형하기 전에** 찍는다.
    env = run_metadata()
    print(f"{len(cases)}문항 × {len(args.values)}조건 × {args.repeat}회 "
          f"(model={env['llm_model']})\n")

    out = {}
    for v in args.values:
        print(f"── CONTEXT_ORDER = {v} ──")
        out[v] = run_condition(cases, v, args.repeat)

    print(f"\n{'유형':<12} " + "".join(f"{v:<12}" for v in args.values))
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
        print(f"{v}: 정답률 {total / n * 100:.0f}% ({total}/{n}) · "
              f"지연 중앙값 {out[v]['median_sec']:.1f}s")

    if len(args.values) == 2:
        a, b = args.values
        print(f"\n{a} → {b} 에서 뒤집힌 문항:")
        for c in cases:
            q = c["question"]
            d = out[b]["tally"][q] - out[a]["tally"][q]
            if d:
                print(f"  {'↑' if d > 0 else '↓'} ({c.get('type')}) "
                      f"{out[a]['tally'][q]}/{args.repeat} → "
                      f"{out[b]['tally'][q]}/{args.repeat}  {q[:44]}")

    RESULTS.write_text(json.dumps(
        {"env": env, "repeat": args.repeat,
         "conditions": {v: {"accuracy": round(
             sum(out[v]["tally"].values()) / n * 100, 1),
             "median_sec": round(out[v]["median_sec"], 2),
             "per_question": dict(out[v]["tally"])} for v in args.values}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
