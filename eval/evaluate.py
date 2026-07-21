"""RAG Agent 평가 루프.

eval_set.json의 질문을 Agent에 실행하고 두 가지 지표를 측정한다:
  1. Answer Accuracy — 답변이 정답 패턴(answer_patterns)과 일치하는 비율
  2. Rewrite Rate  — 질문 재작성이 발생한 비율 (검색 품질 지표)

채점 v2: 초기 키워드 포함 채점은 "논문 7편"의 '7'이 다른 숫자에 우연히
들어가는 식의 허위 통과가 있었다. 지금은 단위까지 요구하는 정규식 패턴에
전부 일치해야 하고, 거부 답변("찾을 수 없습니다")은 무조건 실패로 센다.

결과는 콘솔 리포트 + eval/results.json 으로 저장되어
프롬프트/청킹/모델 변경 전후 성능을 비교하는 피드백 루프로 사용한다.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results.json"

REFUSAL = re.compile(r"찾을 수 없|알 수 없|정보가 없")


def is_pass(answer: str, case: dict) -> bool:
    """정답 패턴 전부 일치 + 거부 답변 아님."""
    if REFUSAL.search(answer):
        return False
    patterns = case.get("answer_patterns")
    if patterns:
        return all(re.search(p, answer) for p in patterns)
    return all(kw.lower() in answer.lower() for kw in case["expected_keywords"])


def main():
    from graph import build_graph  # noqa: E402  (지연 임포트 — rescore.py가 LLM 없이 is_pass만 쓰게)

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    graph = build_graph()

    results = []
    passed = 0
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        out = graph.invoke(
            {"question": case["question"], "query": case["question"], "rewrites": 0}
        )
        elapsed = time.time() - t0

        answer = out["answer"]
        hit = is_pass(answer, case)
        passed += hit

        status = "PASS" if hit else "FAIL"
        print(f"[{i}/{len(cases)}] {status} ({elapsed:.1f}s, "
              f"재작성 {out['rewrites']}회) {case['question']}")
        if not hit:
            print(f"    정답 패턴: {case.get('answer_patterns', case['expected_keywords'])}")
            print(f"    실제 답변: {answer[:150]}")

        results.append({
            "question": case["question"],
            "answer": answer,
            "expected_keywords": case["expected_keywords"],
            "pass": hit,
            "rewrites": out["rewrites"],
            "latency_sec": round(elapsed, 1),
            "sources": out["sources"],
        })
        # 중간 결과를 매 케이스마다 저장 (타임아웃 나도 부분 결과 보존)
        RESULTS.write_text(
            json.dumps({"partial": True, "cases": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    accuracy = passed / len(cases) * 100
    rewrite_rate = sum(r["rewrites"] > 0 for r in results) / len(cases) * 100
    avg_latency = sum(r["latency_sec"] for r in results) / len(cases)

    print("\n===== 평가 결과 =====")
    print(f"Answer Accuracy : {accuracy:.0f}% ({passed}/{len(cases)})")
    print(f"Rewrite Rate    : {rewrite_rate:.0f}%")
    print(f"Avg Latency     : {avg_latency:.1f}s")

    RESULTS.write_text(
        json.dumps({
            "accuracy": accuracy,
            "rewrite_rate": rewrite_rate,
            "avg_latency_sec": round(avg_latency, 1),
            "cases": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n상세 결과 저장: {RESULTS}")


if __name__ == "__main__":
    main()
