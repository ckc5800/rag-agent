"""corrective 루프가 실제로 값을 하는가 — MAX_REWRITES 0 vs 1 A/B.

이 프로젝트의 셀링 포인트는 "검색 품질을 스스로 판정해 부족하면 질문을
재작성한다"인데, **그 루프가 정답을 몇 건 구제했는지는 측정된 적이 없다.**
README의 재작성률 30%는 "3건이 재작성 경로로 갔다"는 뜻일 뿐, 그 3건이
재작성 덕분에 통과했는지 그냥 통과했는지를 말해 주지 않는다.

게다가 청크 크기 스윕에서 recall@3이 300~1600자 전 구간 100%였다. 이
코퍼스에서 검색은 사실상 실패하지 않는다는 뜻이고, 그러면 루프가 개입할
일이 애초에 없을 수도 있다.

    OFF: MAX_REWRITES=0  → grade·rewrite 노드가 통째로 빠진 순수 RAG
    ON : MAX_REWRITES=1  → 현재 동작

같은 질문을 양쪽에서 --repeat회씩 돌려 **질문별 통과 횟수**로 비교한다.
1회씩 비교하면 3B의 런 간 편차(±10%p)에 묻힌다 — 도구 체이닝에서 n=1로
결론을 잘못 뒤집었던 적이 있어서, 여기서는 처음부터 반복이 기본이다.

    python eval/ab_rewrite.py                  # 10문항 × 2조건 × 3회
    python eval/ab_rewrite.py --repeat 1       # 빨리 확인 (결론 내리지 말 것)
    python eval/ab_rewrite.py --limit 3        # 앞 3문항만

주의: LLM을 (문항 × 2 × repeat)회 호출한다. CPU 3B 기준 기본 설정이면
수십 분 걸린다. Ollama가 떠 있어야 한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from evaluate import EVAL_SET, is_pass  # noqa: E402
from runmeta import run_metadata  # noqa: E402

RESULTS = Path(__file__).parent / "results_ab_rewrite.json"


def run_condition(cases: list[dict], max_rewrites: int, repeat: int) -> list[dict]:
    """MAX_REWRITES를 바꿔 평가셋을 repeat회 돌린다."""
    from graph import build_graph

    config.MAX_REWRITES = max_rewrites   # needs_grading/decide_next가 호출 시점에 읽는다
    graph = build_graph()

    rows = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        passes, rewrites, secs, answers = 0, 0, [], []
        for r in range(repeat):
            t0 = time.time()
            out = graph.invoke({"question": q, "query": q, "rewrites": 0})
            secs.append(time.time() - t0)
            hit = is_pass(out["answer"], case)
            passes += hit
            rewrites += out["rewrites"] > 0
            answers.append(out["answer"])
            print(f"    [{i}/{len(cases)} {r + 1}/{repeat}] "
                  f"{'PASS' if hit else 'FAIL'} ({secs[-1]:.0f}s, "
                  f"재작성 {out['rewrites']}회)")

        rows.append({
            "question": q, "passed": passes, "of": repeat,
            "rewrote": rewrites,
            "median_sec": round(sorted(secs)[len(secs) // 2]),
            "answers": [a[:200] for a in answers],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="corrective 루프 A/B")
    ap.add_argument("--repeat", type=int, default=3,
                    help="조건별·질문별 반복 횟수 (3B 편차 때문에 기본 3)")
    ap.add_argument("--limit", type=int, default=None, help="앞 N문항만")
    args = ap.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]

    total_calls = len(cases) * 2 * args.repeat
    print(f"{len(cases)}문항 × 2조건 × {args.repeat}회 = RAG 실행 {total_calls}회\n")

    print("── OFF: MAX_REWRITES=0 (grade·rewrite 없음) ──")
    off = run_condition(cases, 0, args.repeat)
    print("\n── ON: MAX_REWRITES=1 (현재 동작) ──")
    on = run_condition(cases, 1, args.repeat)

    print("\n===== 질문별 비교 (통과 횟수) =====")
    rescued, broken, unchanged = [], [], 0
    for a, b in zip(off, on):
        delta = b["passed"] - a["passed"]
        if delta > 0:
            tag = "↑ "
            rescued.append(a["question"])
        elif delta < 0:
            tag = "↓ "
            broken.append(a["question"])
        else:
            tag = "  "
            unchanged += 1
        print(f"  {tag}OFF {a['passed']}/{a['of']} → ON {b['passed']}/{b['of']}"
              f"  (ON에서 재작성 {b['rewrote']}/{b['of']}회) "
              f"{a['question'][:38]}")

    off_rate = sum(r["passed"] for r in off) / max(sum(r["of"] for r in off), 1)
    on_rate = sum(r["passed"] for r in on) / max(sum(r["of"] for r in on), 1)
    off_med = sorted(r["median_sec"] for r in off)[len(off) // 2]
    on_med = sorted(r["median_sec"] for r in on)[len(on) // 2]

    print(f"\n정답률   OFF {off_rate * 100:.0f}%  →  ON {on_rate * 100:.0f}%")
    print(f"지연(중앙값) OFF {off_med}s  →  ON {on_med}s")
    print(f"루프가 구제한 질문 {len(rescued)}개 / 악화시킨 질문 {len(broken)}개 "
          f"/ 변화 없음 {unchanged}개")
    if rescued:
        print("  구제: " + ", ".join(q[:30] for q in rescued))
    if broken:
        print("  악화: " + ", ".join(q[:30] for q in broken))
    print("\n해석: 구제 0건이고 지연만 늘었다면, 이 코퍼스에서 corrective 루프는"
          "\n      비용만 쓰는 장치다. 그걸 아는 것이 루프를 방어하는 것보다 낫다."
          f"\n      (문항 {len(cases)}개 × {args.repeat}회이므로 1~2건 차이는"
          " 편차일 수 있다)")

    RESULTS.write_text(json.dumps({
        "repeat": args.repeat, "model": config.LLM_MODEL,
        "env": run_metadata(),
        "off_accuracy": round(off_rate * 100, 1),
        "on_accuracy": round(on_rate * 100, 1),
        "off_median_sec": off_med, "on_median_sec": on_med,
        "rescued": rescued, "broken": broken,
        "off": off, "on": on,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
