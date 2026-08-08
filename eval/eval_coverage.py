"""검색 커버리지 평가 — recall@k 가 숨기는 실패를 드러낸다.

**왜 이 스크립트가 따로 필요한가**

eval_retrieval.py 의 recall@k 는 "gold 청크를 **하나라도** top-k 안에
올렸는가"로 정의된다. fact 질문("TTFB 얼마나 개선했나")에는 맞다 — 근거
한 조각이면 답이 나온다. 그런데 집계 질문은 **전수**가 필요하다.
"한국자동차공학회에 게재한 논문은 몇 편인가"는 목록 두 행이 다 모여야
셀 수 있는데, 한 행만 회수해도 recall 은 성공으로 센다.

실제로 그 일이 일어나고 있었다:

    recall@3 = 100%(10문항 기준)  ← 검색은 완벽하다고 보고
    aggregation 정답률 = 50~83%   ← 실제로는 틀리고 있음

**그러나 "전수 요구"로 뒤집으면 이번엔 과소평가가 된다.** 같은 질문에 대해
resume 의 한 줄("논문 2편 게재(제1저자, 한국자동차공학회…)")은 **혼자서
답이 된다**. 목록을 못 가져와도 이 청크만 있으면 맞다.

즉 근거는 집합 하나가 아니라 **대안 경로 여러 개**다. 그래서 판정을
gold_anchor_sets(eval/patch_gold_sets.py 가 붙인다) 위에서 한다:

    gold_anchor_sets: [[앵커…],   # 경로 A — 앵커가 **전부** 충족되면 답 가능
                       [앵커…]]   # 경로 B — 대안

    satisfied@k     어느 한 경로라도 완전히 충족됐는가  ← 검색의 진짜 성공 조건
    best_cov@k      경로 중 최대 충족 비율             ← 부분 점수(회귀 감지용)
    any_anchor@k    앵커를 하나라도 건드렸는가          ← 기존 recall 과 같은 의미

앵커 하나는 여러 청크에 걸릴 수 있고, 그중 **아무거나 하나** 회수하면
그 앵커는 충족이다. 따라서 'Qwen3-TTS' 처럼 흔한 문자열이 청크 10개에
걸려도 벌점이 되지 않는다(예전 구현은 10개를 다 요구해서 과소평가했다).

거부(refusal) 문항은 코퍼스에 답이 없어 gold 가 정의되지 않으므로 제외한다.

    python eval/eval_coverage.py
"""
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from graph import hybrid_search  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results_coverage.json"
KS = [1, 3, 5, 6]


def anchor_chunks(anchor: str, chunks: list[str]) -> set[int]:
    return {i for i, t in enumerate(chunks) if anchor in t}


def score(paths: list[list[set[int]]], got: set[int]) -> tuple[bool, float]:
    """(어느 경로 하나가 완전 충족?, 경로별 충족 비율의 최댓값)."""
    best = 0.0
    done = False
    for path in paths:
        hit = sum(1 for cand in path if cand & got)
        frac = hit / len(path) if path else 0.0
        best = max(best, frac)
        done = done or frac == 1.0
    return done, best


def baseline(cases: list[dict], n: int) -> dict:
    """무작위로 k개를 뽑았을 때의 기대치.

    앵커별 충족 확률은 정확히 계산하고(1 - C(N-|S|,k)/C(N,k)), 경로 충족은
    앵커 간 독립을 가정해 곱한다 — 같은 청크를 공유하는 앵커가 있으면 실제
    값보다 낮게 잡히는 보수적 근사다. 기준선 용도로는 충분하다.
    """
    out = {}
    for k in KS:
        sat = cov = anyv = 0.0
        for c in cases:
            p_any_all = 1.0
            best_sat = best_cov = 0.0
            for path in c["paths"]:
                ps = []
                for cand in path:
                    g = len(cand)
                    p = 1.0 - (comb(n - g, k) / comb(n, k) if n - g >= k else 0.0)
                    ps.append(p)
                    p_any_all *= (1 - p)
                prod = 1.0
                for p in ps:
                    prod *= p
                best_sat = max(best_sat, prod)
                best_cov = max(best_cov, sum(ps) / len(ps))
            sat += best_sat
            cov += best_cov
            anyv += 1 - p_any_all
        m = len(cases)
        out[f"@{k}"] = {"satisfied": round(sat / m * 100),
                        "best_cov": round(cov / m * 100),
                        "any_anchor": round(anyv / m * 100)}
    return out


def main() -> int:
    raw = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line)["page_content"] for line in f]

    cases, skipped = [], 0
    for c in raw:
        sets = c.get("gold_anchor_sets")
        if not sets:
            if c.get("gold_anchors"):
                sets = [[a] for a in c["gold_anchors"]]
            else:
                skipped += 1          # refusal — 코퍼스에 답이 없다
                continue
        paths = [[anchor_chunks(a, chunks) for a in path] for path in sets]
        for path, names in zip(paths, sets):
            for cand, a in zip(path, names):
                if not cand:
                    print(f"[warn] 앵커가 어느 청크에도 없음: {a!r} ({c['question']})")
        cases.append({"question": c["question"], "type": c.get("type", "fact"),
                      "n_paths": len(sets), "paths": paths})

    print(f"{len(cases)}문항 (거부 {skipped}문항 제외 — gold 가 정의되지 않는다)\n")

    rows = []
    for c in cases:
        docs = hybrid_search(c["question"])
        got_all = [d.metadata.get("chunk_index") for d in docs]
        per_k = {}
        for k in KS:
            got = set(got_all[:k])
            sat, cov = score(c["paths"], got)
            anyv = any(cand & got for path in c["paths"] for cand in path)
            per_k[f"@{k}"] = {"satisfied": sat, "best_cov": cov, "any_anchor": anyv}
        rows.append({"question": c["question"], "type": c["type"],
                     "n_paths": c["n_paths"], "retrieved": got_all, "per_k": per_k})

        k6 = per_k["@6"]
        mark = "OK  " if k6["satisfied"] else ("부분" if k6["any_anchor"] else "MISS")
        extra = f" (경로 {c['n_paths']}개)" if c["n_paths"] > 1 else ""
        print(f"[{mark}] 최대충족 {k6['best_cov'] * 100:>3.0f}%{extra:<11} "
              f"| {c['question'][:44]}")

    def agg(sub: list[dict]) -> dict:
        n = len(sub)
        return {f"@{k}": {m: round(sum(r["per_k"][f"@{k}"][m] for r in sub) / n * 100)
                          for m in ("satisfied", "best_cov", "any_anchor")}
                for k in KS}

    overall, base = agg(rows), baseline(cases, len(chunks))
    print("\n===== 전체 =====")
    print(f"{'':8}{'satisfied@k':>13}{'best_cov@k':>13}{'any_anchor@k':>15}"
          f"   (무작위 기준선)")
    for k in KS:
        o, b = overall[f"@{k}"], base[f"@{k}"]
        print(f"  k={k:<5}{o['satisfied']:>11}%{o['best_cov']:>12}%"
              f"{o['any_anchor']:>14}%   "
              f"({b['satisfied']}% / {b['best_cov']}% / {b['any_anchor']}%)")

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    print("\n===== 유형별 (k=6) =====")
    print(f"{'유형':<14}{'n':>3}{'satisfied':>11}{'best_cov':>10}{'any':>7}")
    for t in sorted(by_type):
        a = agg(by_type[t])["@6"]
        print(f"  {t:<12}{len(by_type[t]):>3}{a['satisfied']:>10}%"
              f"{a['best_cov']:>9}%{a['any_anchor']:>6}%")

    RESULTS.write_text(json.dumps(
        {"overall": overall, "baseline": base,
         "by_type": {t: agg(s) for t, s in by_type.items()}, "cases": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
