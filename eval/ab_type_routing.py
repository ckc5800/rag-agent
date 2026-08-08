"""유형별 라우팅(TYPE_ROUTING) A/B — aggregation·enumeration 문항만 잰다.

route.ROUTES는 aggregation·enumeration 유형에만 오버라이드를 건다. 다른
유형(fact/temporal/comparison/trap/refusal)은 오버라이드가 없어 TYPE_ROUTING
on/off가 코드상 동일하게 동작하므로, 이 A/B는 라벨이 aggregation·
enumeration인 문항만 대상으로 한다(75문항 중 16문항).

**대상 안에서 다시 둘로 갈라 본다.** 라벨이 그 유형이라고 다 라우팅되는 게
아니다 — 분류기는 정밀도 우선이라 애매하면 fact로 폴백하고(enumeration
리콜 5/9), 폴백된 문항은 ON/OFF가 완전히 같은 코드 경로다. 섞어서 합계를
내면 그 노이즈가 효과를 덮는다. 실제로 75문항 실측에서 헤드라인은
71% → 71% 동률로 보였는데:

    변수(실제 라우팅됨 12문항)   23/36 → 27/36   +4판정, 악화 0
    대조군(폴백 4문항, 코드 동일) 11/12 →  7/12   -4판정  ← 노이즈

"악화 2건"으로 잡혔던 문항은 **둘 다 라우팅되지 않는 폴백 문항**이었다.
그래서 이 스크립트는 둘을 나눠 보고하고, 폴백 문항을 버리는 대신
**대조군으로 쓴다** — 그 열의 흔들림이 곧 이 실행의 노이즈 크기다.

    python eval/ab_type_routing.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import route  # noqa: E402
import config  # noqa: E402
from evaluate import is_pass  # noqa: E402
from runmeta import run_metadata  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results_ab_type_routing.json"
TARGET_TYPES = ("aggregation", "enumeration")


def run_condition(cases: list[dict], type_routing: bool, repeat: int) -> list[dict]:
    from graph import ask

    config.TYPE_ROUTING = type_routing
    rows = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        passes, answers = 0, []
        for r in range(repeat):
            out = ask(q)
            hit = is_pass(out["answer"], case)
            passes += hit
            answers.append(out["answer"])
            print(f"    [{i}/{len(cases)} {r + 1}/{repeat}] "
                  f"{'PASS' if hit else 'FAIL'} ({case.get('type')}) "
                  f"{out['answer'][:70].strip()}")
        # 답변 원문을 저장한다 — 지난 실행에서 stdout 70자 잘림 때문에
        # 사후 재채점이 막혔다(채점기 결함을 찾고도 일부 문항은 검증 불가).
        # rescore.py가 재채점할 수 있도록 원문을 남긴다.
        rows.append({"question": q, "type": case.get("type"),
                     "passed": passes, "of": repeat, "answers": answers})
    return rows


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    cases = [c for c in json.loads(EVAL_SET.read_text(encoding="utf-8"))
              if c.get("type") in TARGET_TYPES]
    print(f"{len(cases)}문항 (aggregation·enumeration만) × 2조건 × "
          f"{args.repeat}회 = RAG 실행 {len(cases) * 2 * args.repeat}회\n")

    print("── OFF: TYPE_ROUTING=False (기존 전역 기본값) ──")
    off = run_condition(cases, False, args.repeat)
    print("\n── ON: TYPE_ROUTING=True (유형별 오버라이드) ──")
    on = run_condition(cases, True, args.repeat)

    # **라벨이 aggregation·enumeration이라고 다 라우팅되는 게 아니다.**
    # 분류기는 정밀도 우선이라 애매하면 fact로 폴백하고(enumeration 리콜 5/9),
    # 폴백된 문항은 ON/OFF가 **완전히 같은 코드 경로**다. 둘을 섞어 합계를
    # 내면 그 노이즈가 효과를 덮는다 — 실제로 75문항 측정에서 헤드라인은
    # 71%→71% 동률인데, 갈라 보니 라우팅된 쪽은 +4판정·악화 0이고 악화로
    # 잡힌 2건은 전부 폴백(=코드 동일) 문항이었다. 그래서 나눠서 보고한다.
    # 폴백 문항은 버리지 않고 **대조군**으로 쓴다 — 그 열의 흔들림이 곧
    # 이 실행의 노이즈 크기다.
    routed_q = {c["question"] for c in cases
                if route.classify_question_type(c["question"]) in route.ROUTES}

    print("\n===== 문항별 비교 =====")
    rescued, broken, unchanged = [], [], 0
    for a, b in zip(off, on):
        delta = b["passed"] - a["passed"]
        is_routed = a["question"] in routed_q
        tag = "UP" if delta > 0 else ("DN" if delta < 0 else "  ")
        if is_routed:                      # 대조군의 변동은 구제/악화로 세지 않는다
            if delta > 0:
                rescued.append(a["question"])
            elif delta < 0:
                broken.append(a["question"])
            else:
                unchanged += 1
        cls = route.classify_question_type(a["question"])
        where = f"→{cls}" if is_routed else "대조군(폴백)"
        print(f"  {tag} [{a['type']:<11}] {where:<13} OFF {a['passed']}/{a['of']} -> "
              f"ON {b['passed']}/{b['of']}  {a['question'][:38]}")

    def rate(rows, only_routed):
        sel = [r for r in rows if (r["question"] in routed_q) == only_routed]
        got, tot = sum(r["passed"] for r in sel), sum(r["of"] for r in sel)
        return got, tot, (got / tot * 100 if tot else 0.0)

    v_off, v_tot, v_off_r = rate(off, True)
    v_on, _, v_on_r = rate(on, True)
    c_off, c_tot, _ = rate(off, False)
    c_on, _, _ = rate(on, False)

    off_rate = sum(r["passed"] for r in off) / max(sum(r["of"] for r in off), 1)
    on_rate = sum(r["passed"] for r in on) / max(sum(r["of"] for r in on), 1)

    print(f"\n변수(실제 라우팅됨) : {v_off}/{v_tot} → {v_on}/{v_tot}"
          f"  ({v_on - v_off:+d}판정, {v_off_r:.0f}% → {v_on_r:.0f}%)")
    print(f"대조군(폴백=코드 동일): {c_off}/{c_tot} → {c_on}/{c_tot}"
          f"  ({c_on - c_off:+d}판정)  ← 이만큼이 이 실행의 노이즈")
    print(f"합계(섞은 값, 참고용) : OFF {off_rate*100:.0f}% → ON {on_rate*100:.0f}%")
    print(f"\n라우팅된 문항 기준 구제 {len(rescued)} / 악화 {len(broken)} / 변화없음 {unchanged}")
    if rescued:
        print("  구제: " + ", ".join(q[:30] for q in rescued))
    if broken:
        print("  악화: " + ", ".join(q[:30] for q in broken))
    print("\n주의: fact/temporal/comparison/trap/refusal은 라우팅이 코드상"
          " 개입하지 않아(route.ROUTES에 없음) 여기서 재지 않았다.")

    RESULTS.write_text(json.dumps({
        "repeat": args.repeat, "model": config.LLM_MODEL,
        "env": run_metadata(),
        # 합계는 참고용이다 — 라우팅된 문항과 폴백(대조군)이 섞여 있다.
        "off_accuracy": round(off_rate * 100, 1),
        "on_accuracy": round(on_rate * 100, 1),
        # 판단은 이 둘로 한다: 변수의 변화가 대조군의 흔들림보다 큰가.
        "routed": {"off": v_off, "on": v_on, "judgements": v_tot},
        "control_fallback": {"off": c_off, "on": c_on, "judgements": c_tot},
        "routed_questions": sorted(routed_q),
        "rescued": rescued, "broken": broken,
        "off": off, "on": on,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
