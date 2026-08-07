"""GENERATE_PROMPT_VARIANT A/B — targeted 프롬프트의 첫 정량 실측.

targeted는 실측된 두 실패 패턴(열거 목록의 첫 항목만 답함, 같은 대상의
숫자가 여러 개일 때 범위 혼동)을 겨냥한 지시 두 줄을 프롬프트에 더한다.
"프롬프트를 늘려 소형 모델이 나빠지는 경우를 이미 겪었으므로 기본은
base로 두고 잰다"고만 남아 있고 실제 전체 평가셋 대조는 없었다.

    python eval/ab_prompt_variant.py                # base vs targeted, 각 2회
    python eval/ab_prompt_variant.py --repeat 3
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

RESULTS = Path(__file__).parent / "results_ab_prompt_variant.json"


def run_condition(cases: list[dict], variant: str, repeat: int) -> dict:
    from graph import build_graph

    config.GENERATE_PROMPT_VARIANT = variant
    graph = build_graph()

    tally, secs = Counter(), []
    for i in range(repeat):
        for case in cases:
            q = case["question"]
            t0 = time.time()
            out = graph.invoke({"question": q, "query": q, "rewrites": 0})
            secs.append(time.time() - t0)
            tally[q] += is_pass(out["answer"], case)
        print(f"    variant={variant} {i + 1}/{repeat} 완료", flush=True)
    return {"tally": tally, "median_sec": sorted(secs)[len(secs) // 2]}


def main() -> int:
    ap = argparse.ArgumentParser(description="GENERATE_PROMPT_VARIANT A/B")
    ap.add_argument("--values", nargs="+", default=["base", "targeted"])
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    n = len(cases) * args.repeat
    print(f"{len(cases)}문항 × {len(args.values)}조건 × {args.repeat}회\n")

    out = {}
    for v in args.values:
        print(f"── GENERATE_PROMPT_VARIANT = {v} ──")
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
        {"repeat": args.repeat,
         "conditions": {v: {"accuracy": round(
             sum(out[v]["tally"].values()) / n * 100, 1),
             "median_sec": round(out[v]["median_sec"], 2),
             "per_question": dict(out[v]["tally"])} for v in args.values}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
