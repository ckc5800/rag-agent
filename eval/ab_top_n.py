"""GENERATE_TOP_N A/B — top-3 컷오프가 병목인가.

진단(eval/diagnose.py)에서 나온 사실:

    hybrid recall@3 = 83%,  recall@6 = 95%
    검색 기인 실패 6건의 gold rank = 4, 5, 5, 6, 5, 6  ← **전부 top-6 안**

즉 근거는 검색되는데 generate가 3개만 봐서 못 받는다. 반대로 6개를 넣으면
컨텍스트가 2배가 되어 소형 모델의 lost-in-the-middle에 걸린다(top-3으로
줄인 것이 원래 그 대응이었다). 어느 쪽이 큰지는 재봐야 안다.

    python eval/ab_top_n.py                 # 3 vs 6, 각 2회
    python eval/ab_top_n.py --values 3 4 6  # 4까지 포함
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

RESULTS = Path(__file__).parent / "results_ab_top_n.json"


def run_condition(cases: list[dict], top_n: int, repeat: int) -> dict:
    from graph import build_graph

    # 프로덕션 경로(graph.run)는 TYPE_ROUTING으로 질문 유형별 오버라이드를
    # 건다. 여기서는 graph.invoke를 직접 불러 그 경로를 우회하는데, 그게
    # **암묵적이면 위험하다** — 나중에 run()으로 바꾸는 순간 조용히 다른 것을
    # 재게 된다(ab_rewrite는 실제로 경로를 바꾸자 결론의 부호가 뒤집혔다).
    # 우회를 코드로 명시해 둔다.
    config.TYPE_ROUTING = False
    config.GENERATE_TOP_N = top_n     # context_docs가 호출 시점에 읽는다
    graph = build_graph()

    tally, secs = Counter(), []
    for i in range(repeat):
        for case in cases:
            q = case["question"]
            t0 = time.time()
            out = graph.invoke({"question": q, "query": q, "rewrites": 0})
            secs.append(time.time() - t0)
            tally[q] += is_pass(out["answer"], case)
        print(f"    top-{top_n} {i + 1}/{repeat} 완료", flush=True)
    return {"tally": tally, "median_sec": sorted(secs)[len(secs) // 2]}


def main() -> int:
    ap = argparse.ArgumentParser(description="GENERATE_TOP_N A/B")
    ap.add_argument("--values", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--repeat", type=int, default=2,
                    help="3B 런 편차가 ±6%p라 1회로는 결론이 안 난다")
    args = ap.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    by_q = {c["question"]: c for c in cases}
    n = len(cases) * args.repeat
    # 환경 지문은 **조건 루프가 config.GENERATE_TOP_N을 변형하기 전에** 찍는다.
    env = run_metadata()
    print(f"{len(cases)}문항 × {len(args.values)}조건 × {args.repeat}회 "
          f"(model={env['llm_model']})\n")

    out = {}
    for v in args.values:
        print(f"── GENERATE_TOP_N = {v} ──")
        out[v] = run_condition(cases, v, args.repeat)

    print(f"\n{'유형':<12} " + "".join(f"top-{v:<8}" for v in args.values))
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
        print(f"top-{v}: 정답률 {total / n * 100:.0f}% ({total}/{n}) · "
              f"지연 중앙값 {out[v]['median_sec']:.1f}s")

    # 어떤 질문이 뒤집혔는지 — 이게 원인 규명의 핵심이다
    if len(args.values) == 2:
        a, b = args.values
        print(f"\ntop-{a} → top-{b} 에서 뒤집힌 문항:")
        for c in cases:
            q = c["question"]
            d = out[b]["tally"][q] - out[a]["tally"][q]
            if d:
                print(f"  {'↑' if d > 0 else '↓'} ({c.get('type')}) "
                      f"{out[a]['tally'][q]}/{args.repeat} → "
                      f"{out[b]['tally'][q]}/{args.repeat}  {q[:44]}")

    RESULTS.write_text(json.dumps(
        {"env": env, "repeat": args.repeat,
         "conditions": {str(v): {"accuracy": round(
             sum(out[v]["tally"].values()) / n * 100, 1),
             "median_sec": round(out[v]["median_sec"], 2),
             "per_question": dict(out[v]["tally"])} for v in args.values}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
