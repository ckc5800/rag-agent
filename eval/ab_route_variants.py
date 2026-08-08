"""route.ROUTES 조합 A/B — 유형별 오버라이드의 내용 자체를 겨룬다.

ab_type_routing.py는 "라우팅을 켜느냐 마느냐"를 재고, 이 스크립트는
**켠 상태에서 어떤 오버라이드가 최선이냐**를 잰다.

동기: ab_neighbor_window.py 재측정에서 enumeration이 W=0 12/18 →
W=1 18/18(만점)로 가장 큰 신호를 냈는데, 정작 ROUTES는 enumeration에
NEIGHBOR_WINDOW를 안 준다(sandwich만 준다). 그런데 그 18/18은 전역
CONTEXT_ORDER=reversed 조건에서 나온 값이라, sandwich + W=1은 **아직
아무도 안 재본 조합**이다 — route.py 주석이 경고하는 바로 그 함정
("처음엔 W=1만 얹었더니 미측정 조합이 됐고 ... 실측으로 확인")이라
가정하지 않고 직접 잰다.

    python eval/ab_route_variants.py
    python eval/ab_route_variants.py --repeat 3
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from evaluate import is_pass  # noqa: E402
from runmeta import run_metadata  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results_ab_route_variants.json"
TARGET_TYPES = ("aggregation", "enumeration")

# 겨룰 ROUTES 조합들. "current"가 현재 코드값이다.
VARIANTS: dict[str, dict[str, dict]] = {
    "current": {
        "aggregation": {"NEIGHBOR_WINDOW": 1, "GENERATE_TOP_N": 3},
        "enumeration": {"CONTEXT_ORDER": "sandwich"},
    },
    # enumeration에도 이웃 확장을 준다(sandwich 유지) — 미측정 조합.
    "enum_w1": {
        "aggregation": {"NEIGHBOR_WINDOW": 1, "GENERATE_TOP_N": 3},
        "enumeration": {"CONTEXT_ORDER": "sandwich", "NEIGHBOR_WINDOW": 1},
    },
    # enumeration을 ab_neighbor_window.py가 18/18을 낸 조건 그대로 —
    # sandwich를 빼고 역순(전역 기본) + W=1.
    "enum_w1_reversed": {
        "aggregation": {"NEIGHBOR_WINDOW": 1, "GENERATE_TOP_N": 3},
        "enumeration": {"NEIGHBOR_WINDOW": 1},
    },
    # 위(채택본)에 targeted 프롬프트를 aggregation에만 얹는다.
    # 근거: ab_prompt_variant.py(55문항 × 4회)에서 targeted는 전역으로는
    # base와 완전 동률(78%, 171/220)이었지만 내역이 상쇄였다 —
    # aggregation 10/24 → 13/24로 오르고 fact 76/88 → 71/88로 내렸다.
    # 특히 "한국자동차공학회에 게재한 논문은 몇 편?"이 0/4 → 4/4로
    # 구제됐는데, 이건 _TARGETED_RULES의 두 번째 규칙(같은 대상의 숫자가
    # 여러 개면 질문이 묻는 범위를 확인하라)이 정확히 겨냥한 실패다.
    # NEIGHBOR_WINDOW와 같은 모양("어떤 유형엔 좋고 어떤 유형엔 나쁘다")이라
    # 같은 해법(유형별 라우팅)이 통하는지 잰다.
    "enum_w1_agg_targeted": {
        "aggregation": {"NEIGHBOR_WINDOW": 1, "GENERATE_TOP_N": 3,
                        "GENERATE_PROMPT_VARIANT": "targeted"},
        "enumeration": {"NEIGHBOR_WINDOW": 1},
    },
}


def run_condition(cases: list[dict], variant: str, repeat: int) -> dict:
    import route
    from graph import ask

    config.TYPE_ROUTING = True
    route.ROUTES = VARIANTS[variant]

    tally = Counter()
    for i, case in enumerate(cases, 1):
        q = case["question"]
        for r in range(repeat):
            out = ask(q)
            hit = is_pass(out["answer"], case)
            tally[q] += hit
            print(f"    [{variant}] [{i}/{len(cases)} {r + 1}/{repeat}] "
                  f"{'PASS' if hit else 'FAIL'} ({case.get('type')})", flush=True)
    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description="route.ROUTES 조합 A/B")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    cases = [c for c in json.loads(EVAL_SET.read_text(encoding="utf-8"))
             if c.get("type") in TARGET_TYPES]
    n = len(cases) * args.repeat
    # 환경 지문은 **조합 루프가 config·route.ROUTES를 변형하기 전에** 찍는다.
    env = run_metadata()
    print(f"{len(cases)}문항(aggregation·enumeration) × {len(args.variants)}조합 "
          f"× {args.repeat}회 (model={env['llm_model']})\n")

    out = {}
    for v in args.variants:
        print(f"── ROUTES = {v} ──")
        out[v] = run_condition(cases, v, args.repeat)

    print(f"\n{'유형':<12} " + "".join(f"{v:<20}" for v in args.variants))
    for t in TARGET_TYPES:
        qs = [c["question"] for c in cases if c.get("type") == t]
        row = f"{t:<12} "
        for v in args.variants:
            hit = sum(out[v][q] for q in qs)
            row += f"{hit:>3}/{len(qs) * args.repeat:<16}"
        print(row)

    print()
    for v in args.variants:
        total = sum(out[v].values())
        print(f"{v}: 정답률 {total / n * 100:.0f}% ({total}/{n})")

    base = args.variants[0]
    for v in args.variants[1:]:
        print(f"\n{base} → {v} 에서 뒤집힌 문항:")
        flipped = False
        for c in cases:
            q = c["question"]
            d = out[v][q] - out[base][q]
            if d:
                flipped = True
                print(f"  {'↑' if d > 0 else '↓'} ({c.get('type')}) "
                      f"{out[base][q]}/{args.repeat} → {out[v][q]}/{args.repeat}"
                      f"  {q[:44]}")
        if not flipped:
            print("  (없음)")

    RESULTS.write_text(json.dumps(
        {"env": env, "repeat": args.repeat,
         "variants": {v: VARIANTS[v] for v in args.variants},
         "results": {v: {"accuracy": round(sum(out[v].values()) / n * 100, 1),
                         "per_question": dict(out[v])} for v in args.variants}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
