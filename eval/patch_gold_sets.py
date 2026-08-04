"""eval_set.json 에 gold_anchor_sets(대안 근거 경로)를 붙인다.

**왜 필요한가**

지금 라벨은 `gold_anchors`(정답 문자열 목록) 하나뿐이고, 평가는 그걸
"이 중 하나라도 담은 청크를 회수했나"로 해석한다. 두 방향으로 틀린다.

1. **과대평가** — 집계 질문은 전수가 필요하다. "한국자동차공학회에 게재한
   논문은 몇 편인가"에 논문 목록 한 행만 회수해도 성공으로 세지만 셀 수 없다.

2. **과소평가** — 반대로 전수를 요구하면 요약 경로를 실패로 센다. 같은 질문에
   대해 resume 의 한 줄("논문 2편 게재(제1저자, 한국자동차공학회…)")은
   **혼자서 답이 된다**. 목록 두 행을 다 못 가져와도 이 청크만 있으면 맞다.

즉 근거는 **집합 하나가 아니라 대안 경로 여러 개**다. 그래서 스키마를 바꾼다:

    gold_anchor_sets: [ [앵커…],   # 경로 A — 이 앵커들이 **전부** 충족되면 답 가능
                        [앵커…] ]  # 경로 B — 대안

    판정: **어느 한 경로라도 완전히 충족**되면 검색 성공.

부수 효과로 "앵커 과잉" 문제가 사라진다. 'Qwen3-TTS' 처럼 흔한 문자열이
청크 10개에 걸려도, 경로는 앵커 1개짜리 집합이므로 그중 아무 청크나 하나
회수하면 충족이다 — 청크 10개를 다 가져오라고 요구하지 않는다.

경로를 명시하지 않은 문항은 `[[a] for a in gold_anchors]`로 유도된다
(= 앵커 하나당 경로 하나 = 기존 recall 의미론). 따라서 이 패치는 집계처럼
**전수가 실제로 필요한 문항에만** 경로를 손으로 적어 넣는다.

    python eval/patch_gold_sets.py --dry-run   # 무엇이 바뀌는지만 출력
    python eval/patch_gold_sets.py --write
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"

# 손으로 확인한 대안 경로. 각 항목은 [경로1, 경로2, …]이고 경로는 앵커 목록이다.
# 근거는 data/chunks.jsonl 을 직접 읽고 확인했다 (아래 주석의 chunk 번호).
PATHS: dict[str, list[list[str]]] = {
    # chunk24 요약 한 줄이 단독으로 답이 된다 / 목록 경로는 두 행 전수 필요
    "한국자동차공학회에 게재한 논문은 몇 편인가요?": [
        ["논문 2편 게재(제1저자, 한국자동차공학회"],                    # chunk24
        ["센서 퓨전 기반의 Trans-Unet을 활용한 2차원",                  # chunk18
         "자율 주행 도메인의 3차원 시맨틱 세그멘테이션을 위한"],        # chunk19
    ],
    # patents.md(chunk0)에 두 건이 같이 있다 / resume 요약도 단독 경로
    "등록된 특허는 총 몇 건인가요?": [
        ["특허 2건(제1발명자)"],                                        # chunk20
        ["1025382250000", "1025382310000"],                             # chunk0
    ],
    "등록된 특허 번호를 알려주세요.": [
        ["1025382250000", "1025382310000"],                             # chunk0
    ],
    # 2023년 게재 3편 — 목록 전수 경로만 존재한다(요약 문장이 없다)
    "2023년에 게재한 논문은 몇 편인가요?": [
        ["국방 데이터 확보를 위한 생성 모델",                           # chunk17
         "GAN을 활용한 데이터 생성 연구 동향",                          # chunk18
         "센서 퓨전 기반의 Trans-Unet을 활용한 2차원"],                 # chunk18
    ],
    # 'Qwen3-TTS'는 청크 10개에 걸리는 과잉 앵커였다 — 엔진을 실제로 명시한
    # 청크만 가리키도록 좁힌다. 어느 하나만 회수해도 답이 된다.
    "TTS 추론 엔진은 어떤 모델을 사용하나요?": [
        ["vLLM-Omni"],
        ["Qwen3-TTS-1.7B"],
    ],
    # 라벨이 resume 경로만 잡고 portfolio 경로를 빠뜨려 MISS 로 세고 있었다.
    # audit_coverage.py 의 "정답률 PASS 인데 커버리지 MISS" 경고로 발견했다 —
    # 모델은 chunk8 로 정확히 답하고 있었다.
    "TTS 프로젝트와 화자 분할 프로젝트 중 먼저 시작한 쪽의 시작 시점은 언제인가요?": [
        ["Speaker Diarization (2025.04"],      # chunk22 (resume.md)
        ["화자 분할(2025.04"],                  # chunk8  (portfolio.md)
    ],
}


def resolve(anchors: list[str], chunks: list[str]) -> set[int]:
    hits: set[int] = set()
    for a in anchors:
        hits |= {i for i, t in enumerate(chunks) if a in t}
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line)["page_content"] for line in f]
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    byq = {c["question"]: c for c in cases}

    bad = 0
    for q, paths in PATHS.items():
        if q not in byq:
            print(f"[error] eval_set 에 없는 질문: {q}")
            bad += 1
            continue
        print(f"\n{q}")
        for n, path in enumerate(paths, 1):
            ok = True
            parts = []
            for a in path:
                idx = sorted(resolve([a], chunks))
                if not idx:
                    ok = False
                parts.append(f"{a[:34]!r}→{idx if idx else 'NONE'}")
            mark = "  " if ok else "!!"
            print(f"  {mark} 경로{n}: " + " + ".join(parts))
            if not ok:
                bad += 1
        byq[q]["gold_anchor_sets"] = paths

    # 경로를 안 적은 문항은 앵커 하나당 경로 하나로 유도 (기존 의미론 유지)
    derived = 0
    for c in cases:
        if c.get("gold_anchors") and "gold_anchor_sets" not in c:
            c["gold_anchor_sets"] = [[a] for a in c["gold_anchors"]]
            derived += 1

    print(f"\n명시 경로 {len(PATHS)}문항 / 유도 {derived}문항 / 앵커 미해결 {bad}건")
    if bad:
        print("앵커가 어느 청크에도 없다 — 문자열을 고치기 전에는 쓰면 안 된다.")
        return 1
    if args.write:
        EVAL_SET.write_text(json.dumps(cases, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"기록: {EVAL_SET}")
    else:
        print("(--write 없이 실행 — 파일은 그대로다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
