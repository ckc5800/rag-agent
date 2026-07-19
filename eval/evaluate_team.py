"""멀티홉 평가 — 단일 RAG vs Multi-Agent 팀 비교.

멀티홉 질문(두 개 이상의 사실을 조합해야 답할 수 있는 질문)에 대해
단일 Corrective-RAG와 planner-workers-synthesizer 팀의 정답률을 비교한다.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graph import build_graph  # noqa: E402
from team import build_team  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_team.json"
RESULTS = Path(__file__).parent / "results_team.json"


def check(answer: str, keywords: list[str]) -> bool:
    return all(k.lower() in answer.lower() for k in keywords)


def main():
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    single = build_graph()
    team = build_team()

    results = []
    for i, case in enumerate(cases, 1):
        q = case["question"]

        t0 = time.time()
        s = single.invoke({"question": q, "query": q, "rewrites": 0})
        single_ok = check(s["answer"], case["expected_keywords"])
        t_single = time.time() - t0

        t0 = time.time()
        r = team.invoke({"question": q})
        team_ok = check(r["answer"], case["expected_keywords"])
        t_team = time.time() - t0

        print(f"[{i}/{len(cases)}] {q}")
        print(f"    단일 RAG: {'PASS' if single_ok else 'FAIL'} ({t_single:.0f}s) — {s['answer'][:70]}")
        print(f"    팀      : {'PASS' if team_ok else 'FAIL'} ({t_team:.0f}s) — {r['answer'][:70]}")

        results.append({
            "question": q,
            "expected_keywords": case["expected_keywords"],
            "single": {"pass": single_ok, "answer": s["answer"], "sec": round(t_single)},
            "team": {"pass": team_ok, "answer": r["answer"],
                     "sub_questions": r["sub_questions"], "sec": round(t_team)},
        })
        RESULTS.write_text(json.dumps({"partial": True, "cases": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    s_pass = sum(r["single"]["pass"] for r in results)
    t_pass = sum(r["team"]["pass"] for r in results)
    print(f"\n멀티홉 {len(cases)}문항 — 단일 RAG {s_pass}/{len(cases)}, 팀 {t_pass}/{len(cases)}")
    RESULTS.write_text(json.dumps({
        "single_pass": s_pass, "team_pass": t_pass, "total": len(cases),
        "cases": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
