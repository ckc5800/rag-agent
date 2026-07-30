"""grade 노드 단독 평가 — 판정기가 얼마나 믿을 만한가.

검색은 recall@k로, 최종 답변은 정답률로 쟀는데 **판정기 자체를 재는 지표가
없었다.** 그래서 "파싱 실패가 몇 %인가", "관련 있는데 재검색으로 보내는
오탐이 몇 건인가"에 답할 수 없었다. fail-open 정책이 파싱 실패를 조용히
흡수하기 때문에 로그만 봐도 안 보인다.

**정답 라벨을 사람이 새로 붙이지 않는다.** retrieval_set.json의 gold(청크
내용 md5)에서 구조적으로 유도한다 — eval_retrieval.py와 같은 재료다:

    positive   질문의 실제 top-3에 gold가 있으면  → 정답은 sufficient
               (gold가 top-3에 없으면 검색이 진짜 실패한 것이므로
                그 케이스의 정답은 insufficient다. 뒤집어 쓰지 않는다.)
    shuffled   질문 A + **다른 질문 B의 top-3**   → 정답은 insufficient
               (gold가 섞여 들어왔는지 확인하고, 섞였으면 그 케이스는 버린다)
    오지랖     코퍼스에 답이 없는 질문의 실제 top-3 → 정답은 insufficient
               (사람의 판단이 들어가는 유일한 부분. 실무에서 제일 중요한
                유형이라 넣었다 — "모른다고 해야 하는 질문")

기준선도 같이 낸다. **"항상 sufficient"라고 답하는 상수 판정기**를 못 이기면
이 노드는 비용만 쓰는 장치다. 기준선 없는 지표는 혼자서 아무것도 증명하지
못한다는 원칙을 판정기에도 적용한다.

    python eval/eval_grade.py                  # 각 케이스 3회
    python eval/eval_grade.py --repeat 1       # 빨리 한 번만 (권장하지 않음)
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from graph import (GRADE_NO, GRADE_UNPARSED, GRADE_YES, context_docs,  # noqa: E402
                   hybrid_search, judge_relevance)

RETRIEVAL_SET = Path(__file__).parent / "retrieval_set.json"
RESULTS = Path(__file__).parent / "results_grade.json"

# 코퍼스에 답이 없는 질문 — 판정기가 insufficient라고 해야 하는 유형.
# 실무에서는 이게 다수이고(사용자는 문서 범위를 모른다), 여기서 통과시키면
# 모델이 없는 사실을 지어낼 기회를 준다.
OUT_OF_SCOPE = [
    "이윤선의 혈액형이 뭐야?",
    "이윤선이 키우는 반려동물 이름은?",
    "이윤선의 토익 점수는 몇 점이야?",
    "이윤선이 좋아하는 음식은 뭐야?",
]


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_cases() -> list[dict]:
    """검색을 한 번씩 돌려 (질문, 문서, 정답라벨) 케이스를 만든다."""
    labeled = json.loads(RETRIEVAL_SET.read_text(encoding="utf-8"))

    # 질문별 실제 top-3 (프로덕션과 같은 hybrid_search → context_docs)
    tops = {}
    for case in labeled:
        tops[case["question"]] = context_docs(hybrid_search(case["question"]))

    cases = []

    # ── positive: gold가 top-3에 들어왔는가로 정답을 정한다
    for case in labeled:
        q = case["question"]
        gold = {g["md5"] for g in case["gold"]}
        docs = tops[q]
        hit = any(md5(d.page_content) in gold for d in docs)
        cases.append({
            "kind": "positive" if hit else "retrieval-miss",
            "question": q, "docs": docs,
            "expected": GRADE_YES if hit else GRADE_NO,
        })

    # ── shuffled: 다른 질문의 top-3를 붙인다 (결정적으로 무관)
    questions = [c["question"] for c in labeled]
    for i, case in enumerate(labeled):
        q = case["question"]
        other = questions[(i + len(questions) // 2) % len(questions)]
        if other == q:
            continue
        gold = {g["md5"] for g in case["gold"]}
        docs = tops[other]
        if any(md5(d.page_content) in gold for d in docs):
            # 같은 사실이 여러 문서에 중복된 코퍼스라 우연히 정답이 섞일 수
            # 있다. 그러면 '무관'이라는 라벨이 거짓이므로 케이스를 버린다.
            continue
        cases.append({"kind": "shuffled", "question": q, "docs": docs,
                      "expected": GRADE_NO})

    # ── 범위 밖 질문: 실제 검색 결과를 그대로 쓴다
    for q in OUT_OF_SCOPE:
        cases.append({"kind": "out-of-scope", "question": q,
                      "docs": context_docs(hybrid_search(q)),
                      "expected": GRADE_NO})

    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="grade 노드 단독 평가")
    ap.add_argument("--repeat", type=int, default=3,
                    help="케이스별 반복 횟수 (판정이 흔들리는지 보려면 2 이상)")
    args = ap.parse_args()

    print(f"모델 {config.LLM_MODEL} · 케이스별 {args.repeat}회\n")
    cases = build_cases()

    rows = []
    for i, case in enumerate(cases, 1):
        verdicts, raws = [], []
        t0 = time.time()
        for _ in range(args.repeat):
            v, raw = judge_relevance(case["question"], case["docs"])
            verdicts.append(v)
            raws.append(raw.strip()[:80])
        elapsed = (time.time() - t0) / args.repeat

        # fail-open 이후의 실효 판정: UNPARSED는 sufficient로 취급된다
        effective = [GRADE_NO if v == GRADE_NO else GRADE_YES for v in verdicts]
        n_ok = sum(e == case["expected"] for e in effective)
        flipped = len(set(verdicts)) > 1

        mark = {"positive": "P", "retrieval-miss": "M",
                "shuffled": "S", "out-of-scope": "O"}[case["kind"]]
        print(f"[{mark}] {n_ok}/{args.repeat} "
              f"기대 {case['expected']:<4} 실제 {','.join(verdicts):<20} "
              f"{'FLIP ' if flipped else '     '}{elapsed:.1f}s "
              f"{case['question'][:34]}")
        if n_ok < args.repeat:
            print(f"       모델 원문: {raws[0]!r}")

        rows.append({**{k: v for k, v in case.items() if k != "docs"},
                     "verdicts": verdicts, "passed": n_ok,
                     "of": args.repeat, "flipped": flipped,
                     "sec": round(elapsed, 1)})

    # ── 집계 ────────────────────────────────────────────
    total = sum(r["of"] for r in rows)
    correct = sum(r["passed"] for r in rows)
    all_verdicts = [v for r in rows for v in r["verdicts"]]
    unparsed = all_verdicts.count(GRADE_UNPARSED)

    # 오탐 = 충분한데 재검색으로 보냄 / 미탐 = 무관한데 통과시킴
    fp = sum(r["of"] - r["passed"] for r in rows if r["expected"] == GRADE_YES)
    fn = sum(r["of"] - r["passed"] for r in rows if r["expected"] == GRADE_NO)
    n_yes_expected = sum(r["of"] for r in rows if r["expected"] == GRADE_YES)
    n_no_expected = total - n_yes_expected

    # 기준선: 아무것도 안 보고 항상 sufficient — 이걸 못 이기면 무의미하다
    baseline = n_yes_expected / total * 100

    summary = {
        "model": config.LLM_MODEL,
        "repeat": args.repeat,
        "cases": len(rows),
        "accuracy": round(correct / total * 100, 1),
        "always_sufficient_baseline": round(baseline, 1),
        "false_positive_rate": round(fp / max(n_yes_expected, 1) * 100, 1),
        "false_negative_rate": round(fn / max(n_no_expected, 1) * 100, 1),
        "unparsed_rate": round(unparsed / total * 100, 1),
        "flip_cases": sum(r["flipped"] for r in rows),
        "by_kind": {k: v for k, v in Counter(r["kind"] for r in rows).items()},
    }

    print("\n===== grade 단독 평가 =====")
    print(f"판정 정확도        : {summary['accuracy']}%  "
          f"({correct}/{total} 판정)")
    print(f"  기준선(항상 통과): {summary['always_sufficient_baseline']}%"
          "   ← 이걸 못 이기면 판정기는 비용만 쓰는 장치다")
    print(f"오탐(충분→재검색)  : {summary['false_positive_rate']}%  "
          f"— 불필요한 재작성·지연")
    print(f"미탐(무관→통과)    : {summary['false_negative_rate']}%  "
          f"— 없는 사실을 지어낼 기회를 준다")
    print(f"파싱 실패          : {summary['unparsed_rate']}%  "
          f"(fail-open으로 통과 처리됨)")
    print(f"판정이 흔들린 케이스: {summary['flip_cases']}/{len(rows)}  "
          f"(temperature 0인데도 뒤집히면 판정기를 신뢰할 수 없다)")

    RESULTS.write_text(json.dumps({"summary": summary, "cases": rows},
                                  ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
