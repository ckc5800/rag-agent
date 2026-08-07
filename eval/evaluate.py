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

# 거부 표현. 좁게 잡으면 **양방향으로 틀린다** — 실측에서 모델이 "정보는 전혀
# 포함되어 있지 않습니다"로 완벽하게 거부했는데 이걸 못 잡아, 거부가 정답인
# 문항을 실패로 셌다. 넓힐 때는 저장된 결과를 재채점해 통과하던 케이스가
# 뒤집히지 않는지 확인할 것 (rescore.py). 이 확장은 51건 중 1건만
# FAIL→PASS로 바꾸고 나머지는 그대로였다.
REFUSAL = re.compile(
    r"찾을 수 없|알 수 없|정보가 없|정보는 없|포함되어 있지 않|포함되지 않"
    r"|언급이 없|언급되어 있지 않|언급되지 않|나와 있지 않|나타나 있지 않"
    r"|제공되지 않|확인할 수 없|명시되어 있지 않|기재되어 있지 않")

# 로컬 LLM(qwen)이 한국어로 답하면서도 중국어 학습 데이터의 흔적으로 전각
# 문장부호(。：，)를 섞어 낼 때가 있다(실측: 55문항 중 3건, 2026-08 확인).
# 지금까지는 우연히 answer_patterns와 안 부딪혀 통과했지만, 정답 패턴이
# 반각 문장부호(마침표 등)를 요구하는 문항이었다면 **정답을 오답으로 채점하는
# 잠재 버그**였다 — 코퍼스(data/chunks.jsonl)는 전각문자 0건으로 깨끗해
# 이건 입력이 아니라 모델 출력의 문제다. 채점 직전에 정규화한다.
_FULLWIDTH_TO_HALFWIDTH = str.maketrans({
    "。": ".", "，": ",", "、": ",", "：": ":", "；": ";",
    "！": "!", "？": "?", "（": "(", "）": ")",
    "「": '"', "」": '"', "『": '"', "』": '"',
})


def normalize_punctuation(text: str) -> str:
    return text.translate(_FULLWIDTH_TO_HALFWIDTH)


def is_pass(answer: str, case: dict) -> bool:
    """정답 패턴 전부 일치 + 거부 답변 아님.

    `expect_refusal: true`인 케이스는 채점이 뒤집힌다 — **거부가 정답**이다.
    코퍼스에 답이 없는 질문("혈액형이 뭐야?")에 모델이 그럴듯한 답을 지어내면
    실패로 세야 하는데, 지금까지는 거부를 무조건 오답 처리해서 그 유형을
    평가셋에 넣는 것 자체가 불가능했다. 환각을 재는 지표가 없었던 셈이다.
    """
    answer = normalize_punctuation(answer)
    refused = bool(REFUSAL.search(answer))
    if case.get("expect_refusal"):
        return refused          # 거부해야 하는데 답을 지어냈으면 실패
    if refused:
        return False
    patterns = case.get("answer_patterns")
    if patterns:
        return all(re.search(p, answer) for p in patterns)
    keywords = case.get("expected_keywords")
    if not keywords:
        # 채점 기준이 아예 없는 케이스를 조용히 통과시키면 정답률이 부풀려진다.
        raise ValueError(
            f"채점 기준이 없습니다 (answer_patterns/expected_keywords 둘 다 없음): "
            f"{case.get('question', case)!r}")
    return all(kw.lower() in answer.lower() for kw in keywords)


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
            criterion = ("거부(문서에 답이 없음)" if case.get("expect_refusal")
                         else case.get("answer_patterns")
                         or case.get("expected_keywords"))
            print(f"    정답 기준: {criterion}")
            print(f"    실제 답변: {answer[:150]}")

        results.append({
            "question": case["question"],
            "type": case.get("type", "fact"),
            "answer": answer,
            "expected_keywords": case.get("expected_keywords"),
            "pass": hit,
            "rewrites": out["rewrites"],
            "latency_sec": round(elapsed, 1),
            "sources": out["sources"],
            "contexts": out.get("contexts", []),
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

    # 유형별로 쪼개서 본다. 전체 정답률 하나로는 "어디가 약한지"를 알 수 없고,
    # 실제로 남은 오답이 집계·열거에 몰려 있다는 것이 이 표에서 보인다.
    # refusal은 환각 지표다 — 답이 없는 질문에 답을 지어내면 실패한다.
    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["pass"])
    print("\n----- 유형별 -----")
    for kind, hits in sorted(by_type.items(),
                             key=lambda kv: -len(kv[1])):
        print(f"  {kind:<12} {sum(hits):>2}/{len(hits):<3} "
              f"{sum(hits) / len(hits) * 100:>3.0f}%")

    from runmeta import run_metadata

    RESULTS.write_text(
        json.dumps({
            # 환경 지문이 없으면 다른 실행과 비교 가능한지 판단할 수 없다.
            # 이 프로젝트에서 실제로 겪은 문제다 (README 90% vs 재측정 74%).
            "env": run_metadata(),
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
