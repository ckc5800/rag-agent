"""팀 라우팅 A/B — 멀티홉 질문만 Multi-Agent 팀으로 보내면 값을 하는가.

**배경**

evaluate_team.py --type comparison(8문항×2회) 에서 팀이 단일 RAG를 크게
이겼다: 56% → 88%(+31%p), 지연 1.4배. 그런데 같은 실험에서 팀이 **지는**
경우도 나왔다.

    "제1저자 논문 편수와 등록 특허 건수 중 더 많은 쪽은 몇 편인가요?"
      단일 2/2 — 요약 청크("논문 7편(제1저자), 특허 2건")를 통째로 받아 답
      팀   0/2 — 하위 질문으로 쪼개니 논문 목록을 직접 세게 되고 5편으로 오답

즉 분해는 **서로 다른 청크의 사실을 엮을 때** 이기고, **한 청크에 요약이
이미 있을 때** 오히려 근거를 잃는다. 그래서 전부 팀으로 보내면 안 되고
멀티홉만 보내야 한다는 것이 가설이다.

분류는 route.classify_question_type(규칙 기반)을 그대로 쓴다. LLM 분류기를
새로 두면 그 분류기의 정확도를 또 재야 하고, 라우팅 이득이 분류 오차에
먹힐 수 있다 — 결정적인 쪽이 A/B 해석을 깨끗하게 만든다.

**세 조건을 같은 채점기(evaluate.is_pass)로 비교한다**

    single   전부 단일 RAG (현재 기본값)
    routed   comparison 으로 분류된 질문만 팀, 나머지는 단일
    all_team 전부 팀 (라우팅이 필요한지 확인하는 상한/하한선)

all_team 을 같이 재는 이유 — routed 가 single 을 이기더라도, all_team 이
더 좋으면 분류기는 불필요한 복잡도다. 반대로 all_team 이 나쁘면 라우팅이
이득의 원천임이 증명된다.

    python eval/ab_team_routing.py --repeat 2
    python eval/ab_team_routing.py --types comparison aggregation  # 부분집합만
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import EVAL_SET, is_pass  # noqa: E402
from runmeta import run_metadata  # noqa: E402
from graph import build_graph  # noqa: E402
from route import should_use_team  # noqa: E402
from team import build_team  # noqa: E402

RESULTS = Path(__file__).parent / "results_ab_team_routing.json"
CONDITIONS = ("single", "routed", "all_team")


def main() -> int:
    ap = argparse.ArgumentParser(description="팀 라우팅 A/B")
    ap.add_argument("--repeat", type=int, default=2,
                    help="런 편차가 있어 1회로는 결론이 안 난다")
    ap.add_argument("--types", nargs="+",
                    help="평가할 유형 (기본: 거부를 뺀 전체)")
    args = ap.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.types:
        cases = [c for c in cases if c.get("type", "fact") in args.types]
    if not cases:
        raise SystemExit("해당 유형 문항이 없다")

    single, team = build_graph(), build_team()

    def run(case: dict, use_team: bool) -> tuple[bool, float]:
        q = case["question"]
        t0 = time.time()
        if use_team:
            out = team.invoke({"question": q})
        else:
            out = single.invoke({"question": q, "query": q, "rewrites": 0})
        return is_pass(out["answer"], case), time.time() - t0

    routed_as = {c["question"]: should_use_team(c["question"]) for c in cases}
    n_routed = sum(1 for v in routed_as.values() if v)
    print(f"{len(cases)}문항 × {args.repeat}회 · 팀으로 라우팅될 문항 {n_routed}개\n")

    hits = {k: Counter() for k in CONDITIONS}
    secs = {k: [] for k in CONDITIONS}
    by_type = {k: defaultdict(lambda: [0, 0]) for k in CONDITIONS}

    for i, case in enumerate(cases, 1):
        q, t = case["question"], case.get("type", "fact")
        use_team_routed = routed_as[q]
        for _ in range(args.repeat):
            # single 과 routed 는 라우팅되지 않는 문항에서 같은 경로를 탄다.
            # 그래도 따로 실행한다 — 같은 결과를 재사용하면 런 편차가 한쪽에만
            # 반영돼 델타가 실제보다 작아 보인다.
            for cond, use_team in (("single", False),
                                   ("routed", use_team_routed),
                                   ("all_team", True)):
                ok, sec = run(case, use_team)
                hits[cond][q] += ok
                secs[cond].append(sec)
                by_type[cond][t][0] += ok
                by_type[cond][t][1] += 1
        marks = " ".join(f"{c[:1]}{hits[c][q]}" for c in CONDITIONS)
        flag = " →팀" if use_team_routed else ""
        print(f"[{i}/{len(cases)}] {marks}{flag:<4} {q[:50]}")

    n = len(cases) * args.repeat
    print(f"\n===== {len(cases)}문항 × {args.repeat}회 = {n}판정 =====")
    base = sum(hits["single"].values())
    for cond in CONDITIONS:
        tot = sum(hits[cond].values())
        avg = sum(secs[cond]) / len(secs[cond])
        delta = f"{(tot - base) / n * 100:+.0f}%p" if cond != "single" else "기준"
        print(f"  {cond:<9}{tot:>3}/{n} ({tot / n * 100:>3.0f}%)  "
              f"평균 {avg:.1f}s  {delta}")

    types = sorted({c.get("type", "fact") for c in cases})
    print(f"\n{'유형':<14}" + "".join(f"{c:>10}" for c in CONDITIONS))
    for t in types:
        row = "".join(f"{by_type[c][t][0]:>5}/{by_type[c][t][1]:<4}"
                      for c in CONDITIONS)
        print(f"  {t:<12}{row}")

    RESULTS.write_text(json.dumps({
        "env": run_metadata(),
        "repeat": args.repeat, "judgements": n,
        "totals": {c: sum(hits[c].values()) for c in CONDITIONS},
        "avg_sec": {c: round(sum(secs[c]) / len(secs[c]), 1) for c in CONDITIONS},
        "by_type": {c: {t: by_type[c][t] for t in types} for c in CONDITIONS},
        "routed_questions": [q for q, v in routed_as.items() if v],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
