"""멀티홉 평가 — 단일 RAG vs Multi-Agent 팀 비교.

멀티홉 질문(두 개 이상의 사실을 조합해야 답할 수 있는 질문)에 대해
단일 Corrective-RAG와 planner-workers-synthesizer 팀의 정답률을 비교한다.

**표본 주의** — 기본 평가셋(eval_team.json)은 3문항이다. 팀 레이어를 만든
근거가 "단일 RAG는 멀티홉에 약하다"인데, 그 주장을 3문항으로 판정하고
있었다. 1문항이 33%p이므로 어느 쪽이 이겨도 의미가 없다.

그래서 `--type comparison` 을 두었다. eval_set.json 의 comparison 유형
8문항을 가져오고 채점도 evaluate.is_pass(정규식 패턴 + 거부 처리)를 쓴다 —
eval_team.json 의 부분문자열 검사보다 엄격하고, 단일 RAG 정답률과 **같은
잣대**로 비교된다(다른 채점기로 잰 두 수치를 비교하면 안 된다).

    python eval/evaluate_team.py                       # 기존 3문항
    python eval/evaluate_team.py --type comparison     # eval_set 의 8문항
    python eval/evaluate_team.py --type comparison --repeat 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import is_pass  # noqa: E402
from graph import build_graph  # noqa: E402
from team import build_team  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_team.json"
MAIN_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results_team.json"


def check(answer: str, keywords: list[str]) -> bool:
    return all(k.lower() in answer.lower() for k in keywords)


def load_cases(qtype: str | None) -> tuple[list[dict], str]:
    """(문항, 채점 방식 이름). 유형을 지정하면 본 평가셋에서 가져온다."""
    if not qtype:
        return json.loads(EVAL_SET.read_text(encoding="utf-8")), "부분문자열"
    cases = [c for c in json.loads(MAIN_SET.read_text(encoding="utf-8"))
             if c.get("type") == qtype]
    if not cases:
        raise SystemExit(f"eval_set.json 에 type={qtype} 문항이 없다")
    return cases, "정규식 패턴(is_pass)"


def main():
    ap = argparse.ArgumentParser(description="단일 RAG vs 멀티에이전트 팀")
    ap.add_argument("--type", help="eval_set.json 의 유형 (예: comparison)")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    cases, grader = load_cases(args.type)
    grade = (lambda a, c: is_pass(a, c)) if args.type else \
            (lambda a, c: check(a, c["expected_keywords"]))
    print(f"{len(cases)}문항 × {args.repeat}회 · 채점: {grader}\n")

    single = build_graph()
    team = build_team()

    results = []
    for i, case in enumerate(cases, 1):
        q = case["question"]

        s_hits, t_hits, s_secs, t_secs = 0, 0, [], []
        s_last, r_last = None, None
        for _ in range(args.repeat):
            t0 = time.time()
            s_last = single.invoke({"question": q, "query": q, "rewrites": 0})
            s_hits += grade(s_last["answer"], case)
            s_secs.append(time.time() - t0)

            t0 = time.time()
            r_last = team.invoke({"question": q})
            t_hits += grade(r_last["answer"], case)
            t_secs.append(time.time() - t0)

        s, r = s_last, r_last
        single_ok, team_ok = s_hits == args.repeat, t_hits == args.repeat
        t_single = sum(s_secs) / len(s_secs)
        t_team = sum(t_secs) / len(t_secs)
        n = args.repeat

        print(f"[{i}/{len(cases)}] {q}")
        print(f"    단일 RAG: {s_hits}/{n} ({t_single:.0f}s) — {s['answer'][:70]}")
        print(f"    팀      : {t_hits}/{n} ({t_team:.0f}s) — {r['answer'][:70]}")
        if t_hits != s_hits:
            print(f"    ↳ 분해: {r['sub_questions']}")

        results.append({
            "question": q,
            "expected_keywords": case.get("expected_keywords"),
            "single": {"pass": single_ok, "hits": s_hits, "answer": s["answer"],
                       "sec": round(t_single)},
            "team": {"pass": team_ok, "hits": t_hits, "answer": r["answer"],
                     "sub_questions": r["sub_questions"], "sec": round(t_team)},
        })
        RESULTS.write_text(json.dumps({"partial": True, "cases": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    n = args.repeat * len(cases)
    s_hits = sum(r["single"]["hits"] for r in results)
    t_hits = sum(r["team"]["hits"] for r in results)
    s_sec = sum(r["single"]["sec"] for r in results) / len(results)
    t_sec = sum(r["team"]["sec"] for r in results) / len(results)
    print(f"\n{len(cases)}문항 × {args.repeat}회 = {n}판정")
    print(f"  단일 RAG : {s_hits}/{n} ({s_hits / n * 100:.0f}%)  평균 {s_sec:.0f}s")
    print(f"  팀       : {t_hits}/{n} ({t_hits / n * 100:.0f}%)  평균 {t_sec:.0f}s")
    print(f"  델타     : {(t_hits - s_hits) / n * 100:+.0f}%p, "
          f"지연 {t_sec / max(s_sec, 1e-9):.1f}배")
    RESULTS.write_text(json.dumps({
        "type": args.type, "repeat": args.repeat, "judgements": n,
        "single_hits": s_hits, "team_hits": t_hits,
        "single_sec": round(s_sec), "team_sec": round(t_sec),
        "cases": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
