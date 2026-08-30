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
  [WARN] 겉핥기 앵커   — 앵커가 걸린 청크 중 **정답 패턴이 없는 것**이 있다.
                         그 청크를 회수하면 커버리지는 충족으로 세는데 정작
                         답은 못 하므로, 지표가 그만큼 낙관적이 된다.
                         (예전에는 '앵커가 N개 넘는 청크에 걸림'만 봤는데,
                          그건 결함이 아니라 코퍼스의 성질이었다 — 문서
                          코퍼스는 같은 사실을 여러 곳에 적는다. 실측으로
                          갈라 보니 과잉 6건 중 5건은 걸린 청크 **전부가**
                          답을 담고 있었고, 진짜 결함은 1건이었다:
                          "TTS 추론 엔진은 어떤 모델?"의 앵커가 `vLLM-Omni`
                          였는데, 서빙 스택만 적힌 7청크는 모델명을 안 담아
                          답이 안 된다. 개수를 세는 규칙은 그 1건을 나머지
                          5건의 잡음에 묻고 있었다.)
                         집계·비교처럼 정답이 원문에 없는 유형은 이 검사를
                         적용할 수 없어 예전처럼 개수만 알린다
  [WARN] 라벨 누락     — PASS 인데 커버리지 MISS 이고, **컨텍스트에는 정답
                         패턴을 만족하는 청크가 있다**. 근거는 회수됐는데
                         라벨이 그 경로를 안 적어 둔 것 → 라벨을 고친다
  [WARN] 근거 없이 맞음 — PASS 인데 커버리지 MISS 이고, **컨텍스트 어디에도
                         정답 패턴이 없다**. 모델이 파라미터 지식으로 답한
                         것이다 → 라벨이 아니라 검색 문제이고, 그 PASS 는
                         지표를 부풀린다 (둘을 한 메시지로 묶으면 고칠 곳을
                         못 찾는다 — 실제로 둘 다 있었다)
                         (results.json 이 있을 때만 검사)

    python eval/audit_coverage.py            # 경고까지 출력, FAIL 있으면 exit 1
    python eval/audit_coverage.py --strict   # WARN 도 실패로 취급
"""
import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"
COVERAGE = Path(__file__).parent / "results_coverage.json"
RESULTS = Path(__file__).parent / "results.json"

BROAD = 6          # 앵커 하나가 이보다 많은 청크에 걸리면 경고


def shallow_warning(q: str, anchor: str, cand: set[int], chunks: list[str],
                    pats: list[str] | None) -> str | None:
    """넓게 걸린 앵커가 실제로 해로운지 가른다. 무해하면 None.

    판정선은 개수가 아니라 **걸린 청크가 답을 담는가**다. 같은 사실이 여러
    문서에 적힌 것(문서 코퍼스의 정상)과, 답이 없는 청크까지 근거로 세는 것
    (지표를 부풀리는 결함)은 다르다.

    정답 패턴이 어느 청크에도 없으면 집계·비교처럼 답이 원문에 없는
    유형이므로 이 검사를 적용할 수 없다 — 그 경우만 예전처럼 개수를 알린다.
    """
    pats = pats or []
    grounded = {i for i in cand
                if pats and all(re.search(p, chunks[i]) for p in pats)}
    if not grounded:
        return (f"[과잉 앵커·검사불가] {q}\n      {anchor!r} → 청크 {len(cand)}개. "
                f"정답이 원문에 없는 유형이라 겉핥기 판정 불가")
    shallow = cand - grounded
    if not shallow:
        return None              # 전부 답을 담는다 — 코퍼스가 반복할 뿐, 결함 아님
    return (f"[겉핥기 앵커] {q}\n      {anchor!r} → 청크 {len(cand)}개 중 "
            f"{len(shallow)}개가 정답 패턴을 안 담는다. 그 청크를 회수해도 "
            f"답이 안 되는데 커버리지는 충족으로 센다")


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
    by_q = {c["question"]: c for c in cases}
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
                    w = shallow_warning(q, a, cand, chunks,
                                        c.get("answer_patterns"))
                    if w:
                        warns.append(w)
            if all(path):
                need = min_chunks(path)
                best = need if best is None else min(best, need)

        if best is not None and best > config.TOP_K:
            fails.append(f"[구조적 불가] {q}\n      가장 싼 경로도 청크 {best}개가 "
                         f"필요한데 TOP_K={config.TOP_K} 다. 검색기가 통과할 수 없다")

    # ---- 정답률 평가와 교차검증 (둘 다 있고, 같은 인덱스에서 나왔을 때만) ----
    #
    # 지문 대조가 없던 시절 이 교차검증이 **58청크 시절 results.json 과 842청크
    # 커버리지**를 나란히 놓고 "라벨 누락 4건"을 보고하고 있었다. 낡은 답변
    # 기록에서 나온 경고라 그대로 라벨을 고쳤으면 멀쩡한 라벨을 망칠 뻔했다.
    # 인덱스↔chunks 결속(check_index_consistency)과 같은 방식으로 막는다.
    if COVERAGE.exists() and RESULTS.exists():
        res = json.loads(RESULTS.read_text(encoding="utf-8"))
        stamp = (res.get("env") or {}).get("index_chunks_md5")
        current = json.loads(
            Path(config.INDEX_MANIFEST).read_text(encoding="utf-8")
        ).get("chunks_md5") if Path(config.INDEX_MANIFEST).exists() else None
        if stamp != current:
            print(f"[skip] results.json 이 다른 인덱스에서 나왔다 "
                  f"(기록 {str(stamp)[:8]} vs 현재 {str(current)[:8]}) — "
                  f"라벨 누락·근거 없이 맞음 교차검증을 건너뛴다.\n"
                  f"       `python eval/evaluate.py` 로 현재 인덱스에서 다시 잰 뒤 "
                  f"이 감사를 돌릴 것.\n")
            res = None
    else:
        res = None

    if res is not None:
        cov = {r["question"]: r for r in
               json.loads(COVERAGE.read_text(encoding="utf-8"))["cases"]}
        # "PASS 인데 커버리지 MISS"는 **정반대인 두 상황**이 같은 모양으로 나온다.
        # 실제로 둘 다 있었고(2026-08), 하나로 묶으면 고칠 곳을 못 찾는다:
        #
        #   (a) 라벨 누락 — 검색된 청크 중에 정답 패턴을 만족하는 게 **있다**.
        #       근거는 회수됐는데 라벨이 그 경로를 안 적어 둔 것 → 라벨을 고친다.
        #   (b) 근거 없이 맞음 — 검색된 어느 청크도 정답 패턴을 만족하지 **않는다**.
        #       모델이 파라미터 지식으로 답한 것이다("Prometheus/Grafana" 같은
        #       흔한 도구명이 그렇다) → 라벨이 아니라 검색이 문제이고, 그 PASS는
        #       근거 없는 정답이라 지표를 부풀린다.
        #
        # 저장된 결과의 contexts(실제로 프롬프트에 들어간 청크 본문)로 가른다.
        for r in res.get("cases", res if isinstance(res, list) else []):
            q = r.get("question")
            if not (q in cov and r.get("pass")
                    and not cov[q]["per_k"]["@6"]["satisfied"]):
                continue
            case = by_q.get(q, {})
            pats = case.get("answer_patterns") or []
            ctxs = r.get("contexts") or []
            supported = bool(pats) and any(
                all(re.search(p, ctx) for p in pats) for ctx in ctxs)
            if supported:
                warns.append(
                    f"[라벨 누락] {q}\n      정답 패턴을 만족하는 청크가 "
                    f"컨텍스트에 있는데 gold_anchor_sets 에 그 경로가 없다 "
                    f"→ 라벨에 대안 경로를 추가할 것")
            elif ctxs:
                warns.append(
                    f"[근거 없이 맞음] {q}\n      컨텍스트 어디에도 정답 패턴이 "
                    f"없는데 PASS 다 — 모델이 파라미터 지식으로 답했을 수 있다. "
                    f"라벨이 아니라 **검색**을 봐야 하고, 이 PASS 는 지표를 부풀린다")
            else:
                warns.append(
                    f"[라벨 누락 의심] {q}\n      정답률 PASS 인데 커버리지 MISS. "
                    f"저장된 결과에 contexts 가 없어 원인을 가를 수 없다 "
                    f"(evaluate.py 를 다시 돌리면 갈린다)")
    elif not (COVERAGE.exists() and RESULTS.exists()):
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
