"""RERANK A/B — 도입 이후 한 번도 측정하지 않았던 기능의 첫 실측.

diagnose.py가 지목한 병목(검색은 top-3에 근거를 올렸는데 생성이 놓치는
10건)을 겨냥해 만든 LLM 재정렬 단계다. 질의당 LLM 호출이 1회 늘어
지연이 커진다는 이유로 "아직 A/B 전"인 채 기본이 꺼져 있었다.

    python eval/ab_rerank.py                # 0 vs 1, 각 2회
    python eval/ab_rerank.py --repeat 3
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

RESULTS = Path(__file__).parent / "results_ab_rerank.json"


def run_condition(cases: list[dict], rerank: bool, repeat: int) -> dict:
    from graph import build_graph

    config.RERANK = rerank
    graph = build_graph()

    tally, secs = Counter(), []
    for i in range(repeat):
        for case in cases:
            q = case["question"]
            t0 = time.time()
            out = graph.invoke({"question": q, "query": q, "rewrites": 0})
            secs.append(time.time() - t0)
            tally[q] += is_pass(out["answer"], case)
        print(f"    RERANK={rerank} {i + 1}/{repeat} 완료", flush=True)
    return {"tally": tally, "median_sec": sorted(secs)[len(secs) // 2]}


def main() -> int:
    ap = argparse.ArgumentParser(description="RERANK A/B")
    ap.add_argument("--values", type=int, nargs="+", default=[0, 1])
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
        print(f"── RERANK = {bool(v)} ──")
        out[v] = run_condition(cases, bool(v), args.repeat)

    print(f"\n{'유형':<12} " + "".join(f"RERANK={bool(v)!s:<7}" for v in args.values))
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
        print(f"RERANK={bool(v)}: 정답률 {total / n * 100:.0f}% ({total}/{n}) · "
              f"지연 중앙값 {out[v]['median_sec']:.1f}s")

    if len(args.values) == 2:
        a, b = args.values
        print(f"\nRERANK off → on 에서 뒤집힌 문항:")
        for c in cases:
            q = c["question"]
            d = out[b]["tally"][q] - out[a]["tally"][q]
            if d:
                print(f"  {'↑' if d > 0 else '↓'} ({c.get('type')}) "
                      f"{out[a]['tally'][q]}/{args.repeat} → "
                      f"{out[b]['tally'][q]}/{args.repeat}  {q[:44]}")

    RESULTS.write_text(json.dumps(
        {"env": env, "repeat": args.repeat,
         "conditions": {str(bool(v)): {"accuracy": round(
             sum(out[v]["tally"].values()) / n * 100, 1),
             "median_sec": round(out[v]["median_sec"], 2),
             "per_question": dict(out[v]["tally"])} for v in args.values}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
