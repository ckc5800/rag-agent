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

# 같은 개념이 코퍼스에서 두 표기로 쓰이는지 보려고, 패턴에서 뽑은 핵심
# 토큰 주변을 사람이 눈으로 볼 수 있게 찍는다.
_TOKEN = re.compile(r"[0-9A-Za-z_가-힣]{3,}")


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

        patterns = case.get("answer_patterns") or case.get("expected_keywords", [])
        for p in patterns:
            hits = [c for c in chunks if re.search(p, c["page_content"])]
            if not hits:
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
