"""유형별 라우팅(TYPE_ROUTING) A/B — aggregation·enumeration 문항만 잰다.

route.ROUTES는 aggregation·enumeration 유형에만 오버라이드를 건다. 다른
유형(fact/temporal/comparison/trap/refusal)은 classify_question_type()이
전부 'fact'(또는 comparison, 오버라이드 없음)로 분류하도록 회귀 테스트로
고정해 뒀으므로(tests/test_units.py), 그 유형들은 TYPE_ROUTING on/off가
코드상 완전히 동일하게 동작한다 — LLM을 다시 불러 확인할 필요가 없다.
그래서 이 A/B는 라우팅이 실제로 값을 바꾸는 aggregation(6문항)·
enumeration(8문항) = 14문항만 대상으로 한다.

    python eval/ab_type_routing.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

    print("\n===== 문항별 비교 =====")
    rescued, broken, unchanged = [], [], 0
    for a, b in zip(off, on):
        delta = b["passed"] - a["passed"]
        tag = "UP" if delta > 0 else ("DN" if delta < 0 else "  ")
        if delta > 0:
            rescued.append(a["question"])
        elif delta < 0:
            broken.append(a["question"])
        else:
            unchanged += 1
        print(f"  {tag} [{a['type']:<11}] OFF {a['passed']}/{a['of']} -> "
              f"ON {b['passed']}/{b['of']}  {a['question'][:40]}")

    off_rate = sum(r["passed"] for r in off) / max(sum(r["of"] for r in off), 1)
    on_rate = sum(r["passed"] for r in on) / max(sum(r["of"] for r in on), 1)
    print(f"\n정답률(aggregation+enumeration만)  OFF {off_rate*100:.0f}%  ->  ON {on_rate*100:.0f}%")
    print(f"구제 {len(rescued)} / 악화 {len(broken)} / 변화없음 {unchanged}")
    if rescued:
        print("  구제: " + ", ".join(q[:30] for q in rescued))
    if broken:
        print("  악화: " + ", ".join(q[:30] for q in broken))
    print("\n주의: fact/temporal/comparison/trap/refusal은 라우팅이 코드상"
          " 개입하지 않아(route.ROUTES에 없음) 여기서 재지 않았다.")

    RESULTS.write_text(json.dumps({
        "repeat": args.repeat, "model": config.LLM_MODEL,
        "env": run_metadata(),
        "off_accuracy": round(off_rate * 100, 1),
        "on_accuracy": round(on_rate * 100, 1),
        "rescued": rescued, "broken": broken,
        "off": off, "on": on,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
