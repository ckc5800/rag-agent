"""gold 라벨 누락 감사 — 손으로 붙인 라벨의 빈틈을 찾는다.

라벨링 기준은 "그 청크만 보고 질문에 답할 수 있는가"였다. 문제는 손으로
훑다 보면 **답이 있는데 빠뜨린다**는 것이다(편향보다 누락이 먼저 온다).
누락은 recall을 실제보다 낮게 만든다 — 정답이 상위에 와도 오답 처리되므로.

여기서는 질문별 정답 문자열로 전 청크를 훑어 gold 밖 후보를 뽑는다.
**자동으로 gold에 넣지 않는다** — 문자열이 있다고 답이 되는 건 아니기
때문이다(예: "논문 2편 게재(제1저자)"는 '제1저자'를 담고 있지만 전체
편수를 묻는 질문의 답이 아니다). 후보만 제시하고 판단은 사람이 한다.

    python eval/audit_gold.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

RETRIEVAL_SET = Path(__file__).parent / "retrieval_set.json"

# 질문별 "정답이 담겼다면 반드시 나타날" 표현. 표기 흔들림까지 포함한다
# (특허번호는 1025382250000 / 10-2538225-0000 두 표기가 섞여 있다).
PATTERNS = {
    "TTS 프로젝트에서 TTFB를 얼마나 개선했나요?": r"2292",
    "스트리밍 오디오의 팝 노이즈 문제를 어떻게 해결했나요?": r"Residual Buffer|잔여 버퍼|팝 노이즈|Pop-noise",
    "이윤선의 제1저자 논문은 몇 편인가요?": r"논문\s*7\s*편|7편\(제1저자\)",
    "등록된 특허 번호를 알려주세요.": r"10-?2538\s?2[23]5|1025382250000|1025382310000",
    "Kubernetes 인프라 구축에 어떤 CI/CD 도구를 사용했나요?": r"ArgoCD|Argo\s*CD",
    "우수 논문상을 받은 논문 제목은?": r"우수\s*논문상",
    "TTS 시스템의 동시 처리 채널은 몇 개로 확장했나요?": r"12\s*→\s*24|동시 처리 채널",
    "화자 분할에는 어떤 모델을 사용했나요?": r"yannote",
    "웹소켓 세션 탈취 문제는 어떻게 방어했나요?": r"password_changed_at|세션 탈취|Session Invalidation",
    "이윤선이 근무한 회사들을 알려주세요.": r"Experience Overview|근무 회사|경력 요약",
}


def main() -> int:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    md5s = [hashlib.md5(r["page_content"].encode("utf-8")).hexdigest()
            for r in rows]
    cases = json.loads(RETRIEVAL_SET.read_text(encoding="utf-8"))

    total_missing = 0
    for case in cases:
        pat = PATTERNS.get(case["question"])
        if not pat:
            continue
        gold = {g["md5"] for g in case["gold"]}
        found = [i for i, r in enumerate(rows)
                 if re.search(pat, r["page_content"])]
        missing = [i for i in found if md5s[i] not in gold]

        stale = [g for g in case["gold"] if g["md5"] not in md5s]
        status = []
        if missing:
            status.append(f"후보 {len(missing)}개")
        if stale:
            status.append(f"무효 라벨 {len(stale)}개")
        print(f"[{'  '.join(status) or 'OK':<12}] {case['question']}")

        for i in missing:
            total_missing += 1
            m = re.search(pat, rows[i]["page_content"])
            s = max(0, m.start() - 45)
            snip = rows[i]["page_content"][s:m.end() + 45].replace("\n", " ")
            print(f"     #{i:<3} [{rows[i]['metadata']['source']}] ...{snip}...")

    print(f"\ngold 밖 후보 {total_missing}건 — 사람이 보고 판단할 것.")
    print("문자열이 있다고 답이 되는 건 아니다 "
          '(예: "논문 2편 게재(제1저자)"는 전체 편수 질문의 답이 아님).')
    return 0


if __name__ == "__main__":
    sys.exit(main())
