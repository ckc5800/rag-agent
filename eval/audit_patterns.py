"""채점 패턴 감사 — 정답을 오답으로 세고 있지 않은가.

실제로 겪은 일: 팝 노이즈 질문의 패턴이 `Residual Buffer`만 허용했는데
문서에는 `잉여 버퍼`라는 표기도 있었다. 모델이 후자로 답하자 **근거에
충실한 정답이 오답으로 집계**됐다. 웹소켓 질문도 같은 이유로
(`password_changed_at` vs `pwd_changed_at`) 깎이고 있었다. 두 건이면
10문항 평가에서 20%p다 — 채점기가 모델보다 더 틀리고 있었던 셈이다.

기계가 대신 판단할 수 없는 문제라(동의어인지 다른 개념인지는 사람이
정한다) `audit_gold.py`와 같은 방침을 따른다: **후보만 제시하고 판단은
사람이 한다.** 다만 기계가 확정할 수 있는 것 하나는 실패로 처리한다 —
**코퍼스의 어떤 청크와도 매치되지 않는 패턴**은 정답이 존재할 수 없으므로
그 케이스는 영원히 실패한다. 그건 명백한 버그다.

    python eval/audit_patterns.py            # 리포트
    python eval/audit_patterns.py --strict   # 매치 0인 패턴이 있으면 exit 1 (CI용)
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"

# 추론형은 **답이 원문에 없는 게 정상**이다. "2023년 논문 몇 편?"의 답 '3편'은
# 표의 행을 세야 나오는 값이라 코퍼스 어디에도 그 문자열이 없다. 이 유형에
# 패턴-코퍼스 매치를 요구하면 정상 문항을 버그로 신고하게 된다.
# 대신 gold_anchors(정답 근거가 있는 청크를 가리키는 문자열)로 검증한다.
DERIVED_TYPES = {"aggregation", "comparison"}

# 같은 개념이 코퍼스에서 두 표기로 쓰이는지 보려고, 패턴에서 뽑은 핵심
# 토큰 주변을 사람이 눈으로 볼 수 있게 찍는다.
_TOKEN = re.compile(r"[0-9A-Za-z_가-힣]{3,}")

# 수치 패턴이 단위 여러 개를 한꺼번에 허용하면(예: `9\s*(개|건|편)`),
# 모델이 **다른 슬롯의 수치를 잘못 말한 것**이 우연히 매치될 수 있다.
# 실제 사례: "합치면 몇 개?"(정답 9개)에 모델이 "논문 수는 9편이고 …
# 총 11개"라고 답했는데 — 최종 결론은 11로 오답인데 — 잘못 센 "9편"이
# `9\s*(개|건|편)`의 편 분기에 걸려 PASS로 집계됐다. 질문이 묻는 단위
# 하나로 좁히면(`9\s*개`) 이 오매치가 구조적으로 막힌다.
_MULTI_UNIT = re.compile(r"\d[^|()]*\((?:개|건|편|번|회|명|대|배)"
                         r"(?:\|(?:개|건|편|번|회|명|대|배))+\)")


def load_chunks() -> list[dict]:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="채점 패턴 감사")
    ap.add_argument("--strict", action="store_true",
                    help="코퍼스와 매치되지 않는 패턴이 있으면 exit 1")
    args = ap.parse_args()

    chunks = load_chunks()
    corpus = "\n".join(c["page_content"] for c in chunks)
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    dead = []
    print(f"평가셋 {len(cases)}문항 · 코퍼스 {len(chunks)}청크\n")

    for case in cases:
        q = case["question"]
        if case.get("expect_refusal"):
            # 거부가 정답인 케이스는 코퍼스에 답이 **없어야** 정상이다.
            leaked = [p for p in case.get("answer_patterns", [])
                      if re.search(p, corpus)]
            state = "!! 코퍼스에 답이 있다" if leaked else "OK(거부 기대)"
            print(f"[{state:<16}] {q[:44]}")
            continue

        derived = case.get("type") in DERIVED_TYPES

        # 추론형은 앵커로 검증한다 — 근거 청크가 코퍼스에 실재해야 한다
        for a in case.get("gold_anchors", []):
            if not any(a in c["page_content"] for c in chunks):
                dead.append((q, f"anchor:{a}"))
                print(f"[!! 앵커 매치 0       ] {q[:40]}")
                print(f"      앵커 {a!r} 를 담은 청크가 없다 — 근거가 사라졌다")

        patterns = case.get("answer_patterns") or case.get("expected_keywords", [])
        for p in patterns:
            if _MULTI_UNIT.search(p):
                print(f"[?? 다단위 수치 패턴  ] {q[:40]}")
                print(f"      {p!r} — 단위 여러 개를 허용하면 다른 슬롯의"
                      " 오답 수치가 우연히 매치될 수 있다. 질문이 묻는"
                      " 단위 하나로 좁힐 것")
            hits = [c for c in chunks if re.search(p, c["page_content"])]
            if not hits:
                if derived:
                    # 세거나 비교해야 나오는 답이라 원문에 없는 게 맞다
                    print(f"[추론형(매치 0 정상) ] {q[:40]}")
                    print(f"      {p!r} — 답이 원문에 없다. 앵커로 검증됨")
                    continue
                dead.append((q, p))
                print(f"[!! 매치 0            ] {q[:40]}")
                print(f"      패턴 {p!r} 이 어떤 청크에도 없다 — 이 케이스는"
                      " 영원히 실패한다")
                continue

            # 사람이 동의 표기를 눈으로 찾을 수 있게, 매치 주변을 보여준다
            m = re.search(p, hits[0]["page_content"])
            s = max(0, m.start() - 30)
            snip = hits[0]["page_content"][s:m.end() + 30].replace("\n", " ")
            print(f"[{len(hits):>2}청크 매치         ] {q[:40]}")
            print(f"      {p!r}  …{snip}…")

    print("\n※ 이 스크립트는 '동의 표기가 있는지'를 대신 판단하지 않는다."
          "\n   위 스니펫을 보고, 같은 사실을 문서가 다른 말로도 쓰고 있다면"
          "\n   그 표기를 패턴에 추가할 것 (예: 'Residual Buffer|잉여 버퍼')."
          "\n   채점 기준을 바꾸면 이전 버전과의 비교가 끊기므로 README에"
          "\n   변경 시점을 남길 것.")

    if dead:
        print(f"\n[FAIL] 코퍼스와 매치되지 않는 패턴 {len(dead)}건")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
