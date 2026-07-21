"""저장된 평가 결과를 채점 v2로 재채점 — 허위 통과 정량화.

LLM을 다시 돌리지 않고, results*.json에 저장된 답변을 새 채점 기준
(정규식 패턴 + 거부 답변 실패 처리)으로 다시 채점한다.
키워드 채점이 얼마나 관대했는지가 숫자로 나온다.

사용: python eval/rescore.py eval/results.json
"""
import json
import sys
from pathlib import Path

from evaluate import EVAL_SET, is_pass


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    cases_by_q = {c["question"]: c
                  for c in json.loads(EVAL_SET.read_text(encoding="utf-8"))}

    old_pass = new_pass = 0
    for r in data["cases"]:
        case = cases_by_q[r["question"]]
        old, new = r["pass"], is_pass(r["answer"], case)
        old_pass += old
        new_pass += new
        if old != new:
            print(f"[{'PASS→FAIL' if old else 'FAIL→PASS'}] {r['question']}")
            print(f"    답변: {r['answer'][:120]}")

    n = len(data["cases"])
    print(f"\n키워드 채점(v1): {old_pass}/{n}  →  패턴 채점(v2): {new_pass}/{n}")


if __name__ == "__main__":
    main()
