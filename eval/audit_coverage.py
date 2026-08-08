"""커버리지 라벨 감사 — 지표가 검색 탓이 아닌 것을 검색 탓으로 세는지 검사.

**왜 필요한가 (실패 사례)**

eval_coverage.py 의 첫 구현은 gold 를 "앵커가 걸린 청크 **전부**"로 정의하고
그걸 다 회수하라고 요구했다. 그래서 이런 일이 벌어졌다:

    'Qwen3-TTS' → 청크 10개 매칭 → 10개를 다 가져와야 성공
    실제로는 그중 하나만 있으면 답이 된다 → 멀쩡한 검색이 22%p 깎였다

지표가 틀리면 그 위에서 내린 모든 판단이 틀린다. audit_patterns.py 가
채점 패턴에 대해 하는 일을 커버리지 라벨에 대해 한다.

**검사 항목**

  [FAIL] 앵커 미해결   — 어느 청크에도 없는 앵커. 라벨이 깨졌다
  [FAIL] 구조적 불가   — 경로를 충족하려면 청크 N개가 필요한데 N > TOP_K.
                         어떤 검색기도 통과할 수 없으니 지표가 아니라 라벨 문제다
  [WARN] 과잉 앵커     — 한 앵커가 너무 많은 청크에 걸린다. 지금 의미론에서는
                         벌점이 아니지만(하나만 맞으면 충족), 앵커가 질문을
                         변별하지 못한다는 신호다
  [WARN] 라벨 누락 의심 — 정답률 평가는 PASS 인데 커버리지는 MISS.
                         모델이 라벨 밖 청크로 답했다는 뜻이다
                         (results.json 이 있을 때만 검사)

    python eval/audit_coverage.py            # 경고까지 출력, FAIL 있으면 exit 1
    python eval/audit_coverage.py --strict   # WARN 도 실패로 취급
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"
COVERAGE = Path(__file__).parent / "results_coverage.json"
RESULTS = Path(__file__).parent / "results.json"

BROAD = 6          # 앵커 하나가 이보다 많은 청크에 걸리면 경고


def min_chunks(path: list[set[int]]) -> int:
    """경로를 충족하는 최소 청크 수 (작은 집합 커버 — 완전 탐색).

    앵커가 같은 청크를 공유하면 1개로 여러 앵커를 덮을 수 있다. 예를 들어
    특허 두 건은 patents.md 한 청크에 같이 있으므로 최소 1이다. 이걸 세지
    않고 앵커 수로 판단하면 멀쩡한 라벨을 '불가'로 몰게 된다.
    """
    universe = sorted(set().union(*path)) if path else []
    for size in range(1, len(path) + 1):
        for combo in combinations(universe, size):
            picked = set(combo)
            if all(cand & picked for cand in path):
                return size
    return len(path) + 1        # 도달 불가 (앵커 미해결이 있는 경우)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="WARN 도 실패 처리")
    args = ap.parse_args()

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line)["page_content"] for line in f]

    fails, warns = [], []
    checked = 0

    for c in cases:
        sets = c.get("gold_anchor_sets")
        if not sets:
            if not c.get("gold_anchors"):
                continue                      # refusal — gold 가 없는 게 정상
            sets = [[a] for a in c["gold_anchors"]]
        checked += 1
        q = c["question"]

        best = None
        for n, names in enumerate(sets, 1):
            path = [{i for i, t in enumerate(chunks) if a in t} for a in names]
            for cand, a in zip(path, names):
                if not cand:
                    fails.append(f"[앵커 미해결] {q}\n      경로{n} 앵커 {a!r} 가 "
                                 f"어느 청크에도 없다")
                if len(cand) > BROAD:
                    warns.append(f"[과잉 앵커] {q}\n      {a!r} → 청크 {len(cand)}개. "
                                 f"질문을 변별하지 못한다")
            if all(path):
                need = min_chunks(path)
                best = need if best is None else min(best, need)

        if best is not None and best > config.TOP_K:
            fails.append(f"[구조적 불가] {q}\n      가장 싼 경로도 청크 {best}개가 "
                         f"필요한데 TOP_K={config.TOP_K} 다. 검색기가 통과할 수 없다")

    # ---- 정답률 평가와 교차검증 (둘 다 있을 때만) ----
    if COVERAGE.exists() and RESULTS.exists():
        cov = {r["question"]: r for r in
               json.loads(COVERAGE.read_text(encoding="utf-8"))["cases"]}
        res = json.loads(RESULTS.read_text(encoding="utf-8"))
        for r in res.get("cases", res if isinstance(res, list) else []):
            q = r.get("question")
            if q in cov and r.get("pass") and not cov[q]["per_k"]["@6"]["satisfied"]:
                warns.append(f"[라벨 누락 의심] {q}\n      정답률 PASS 인데 커버리지 "
                             f"MISS — 라벨 밖 청크로 답했을 수 있다")
    else:
        print("(results_coverage.json / results.json 이 없어 교차검증은 건너뛴다)\n")

    for f in fails:
        print(f"FAIL {f}")
    for w in warns:
        print(f"WARN {w}")
    print(f"\n{checked}문항 감사 — FAIL {len(fails)}건 / WARN {len(warns)}건")
    if fails:
        return 1
    if warns and args.strict:
        print("--strict: WARN 을 실패로 처리한다")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
