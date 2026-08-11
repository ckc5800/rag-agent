"""verify(groundedness) 판정기 단독 평가 — 오답 검출기로서 얼마나 믿을 만한가.

graph.verify는 "답변이 근거 문서에 실제로 기반하는가"를 사후 확인하는
관측 노드인데, 만들어 둔 채 "A/B 전이라 기본 꺼둠"으로 남아 있었다.
grade는 eval_grade.py로 쟀으면서 verify는 안 쟀다 — 같은 부채다.

**정답 라벨을 사람이 새로 붙이지 않는다** (eval_grade와 같은 원칙).
evaluate.py가 저장한 결과 파일(질문·답변·실제 사용된 contexts·패턴 채점)을
재료로, 저장된 답변에 verify 체인만 다시 돌려 판정을 채점과 대조한다:

    정답 중 실질 답변      → grounded=True 여야 한다 (아니면 오경보)
    정답 중 거부 답변      → 별도 버킷 (코퍼스에 답이 없어 거부가 정답인
                            문항 — 심판이 "문서에 없다"고 맞는 말을 하면서
                            grounded=False 를 내는 케이스가 실측에서 7건.
                            이건 환각 오경보가 아니라 거부 답변에서
                            grounded 의미가 정의되지 않는 것이다)
    오답 중 실질 답변      → grounded=False 를 기대 (환각/오독 검출)
    오답 중 거부 답변      → 별도 버킷 — 위와 같은 이유. 거부는 지어낸
                            주장이 없어 grounded 의미가 정의되지 않는다.

    (1차 실행은 정답-거부를 정답에 섞어 세서 "오경보 12.5%"로 읽혔다 —
     갈라 보니 실질 오경보는 1건이었다. 버킷 하나가 결론을 바꾼다.)

주의: pass ≠ grounded 가 개념적으로 완전히 같지는 않다 — "근거에 있는
내용으로 틀리게 답하는" 경우(범위 혼동 등)는 grounded=True 이면서
오답일 수 있다. 그래서 검출 실패 케이스는 답변을 파일에 남겨 사람이
분류할 수 있게 한다.

기준선은 **"항상 grounded=True" 상수 판정기** — 오경보 0에 검출 0.
verify가 값하려면 오경보를 거의 안 내면서 검출을 얹어야 한다.

    python eval/eval_verify.py                            # 기본: 7b 결과 run1
    python eval/eval_verify.py --file results.json        # 다른 결과 파일로
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import REFUSAL  # noqa: E402
from graph import VERIFY_PROMPT, _structured_verify_llm  # noqa: E402
from runmeta import run_metadata  # noqa: E402

RESULTS = Path(__file__).parent / "results_verify.json"


def judge(context: str, answer: str) -> tuple[bool | None, str]:
    """(grounded 판정, unsupported_claim). 파싱 실패는 None."""
    chain = VERIFY_PROMPT | _structured_verify_llm()
    result = chain.invoke({"context": context, "answer": answer})
    parsed = result.get("parsed")
    if parsed is None:
        return None, ""
    return parsed.grounded, parsed.unsupported_claim or ""


def main() -> int:
    ap = argparse.ArgumentParser(description="verify 판정기 단독 평가")
    ap.add_argument("--file", default="results_75q_7b_run1.json",
                    help="evaluate.py 결과 파일 (eval/ 기준 상대경로)")
    args = ap.parse_args()

    env = run_metadata()
    src = Path(__file__).parent / args.file
    data = json.loads(src.read_text(encoding="utf-8"))
    cases = data.get("cases") or data.get("results")
    src_model = (data.get("run_metadata") or data.get("env") or {}).get(
        "llm_model", "?")
    print(f"재료: {src.name} ({len(cases)}건, 답변 생성 모델 {src_model})\n")

    rows, secs = [], []
    for c in cases:
        context = "\n---\n".join(c.get("contexts") or [])
        t0 = time.time()
        grounded, claim = judge(context, c["answer"])
        secs.append(time.time() - t0)
        refusal = bool(REFUSAL.search(c["answer"]))
        bucket = (("정답-거부" if refusal else "정답-실질") if c["pass"]
                  else ("오답-거부" if refusal else "오답-실질"))
        rows.append({"question": c["question"], "type": c.get("type"),
                     "pass": c["pass"], "bucket": bucket,
                     "grounded": grounded, "unsupported_claim": claim,
                     "answer": c["answer"]})
        mark = {True: "grounded", False: "UNGROUNDED", None: "unparsed"}[grounded]
        print(f"  [{bucket:>5}] {mark:<10} | {c['question'][:40]}")

    def rate(sub, pred):
        return f"{sum(pred(r) for r in sub)}/{len(sub)}" if sub else "0/0"

    correct = [r for r in rows if r["bucket"] == "정답-실질"]
    correct_ref = [r for r in rows if r["bucket"] == "정답-거부"]
    wrong = [r for r in rows if r["bucket"] == "오답-실질"]
    refusals = [r for r in rows if r["bucket"] == "오답-거부"]
    unparsed = [r for r in rows if r["grounded"] is None]

    print(f"\n===== verify 판정기 ({len(rows)}건) =====")
    print(f"  오경보 (실질 정답인데 UNGROUNDED)  : "
          f"{rate(correct, lambda r: r['grounded'] is False)}")
    print(f"  검출   (실질 오답을 UNGROUNDED)    : "
          f"{rate(wrong, lambda r: r['grounded'] is False)}")
    print(f"  정답-거부 버킷 (의미 미정의, 참고) : "
          f"{rate(correct_ref, lambda r: r['grounded'] is False)} 가 UNGROUNDED")
    print(f"  오답-거부 버킷 (의미 미정의, 참고) : "
          f"{rate(refusals, lambda r: r['grounded'] is False)} 가 UNGROUNDED")
    print(f"  파싱 실패                          : {len(unparsed)}/{len(rows)}")
    print(f"  판정 지연 중앙값                  : "
          f"{round(statistics.median(secs), 2)}s (질의당 LLM 1회 추가 비용)")
    print("  기준선('항상 grounded') : 오경보 0, 검출 0 — 검출이 이걸 못 얹으면"
          " 비용만 쓰는 노드다.")

    RESULTS.write_text(json.dumps(
        {"run_metadata": env, "source_file": args.file,
         "source_model": src_model,
         "summary": {
             "false_alarm": rate(correct, lambda r: r["grounded"] is False),
             "detection": rate(wrong, lambda r: r["grounded"] is False),
             "correct_refusal_bucket": rate(
                 correct_ref, lambda r: r["grounded"] is False),
             "wrong_refusal_bucket": rate(
                 refusals, lambda r: r["grounded"] is False),
             "unparsed": len(unparsed),
             "median_sec": round(statistics.median(secs), 2)},
         "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
