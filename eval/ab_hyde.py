"""HYDE A/B — 어휘 격차(vocabulary gap)를 겨냥한 가상 답변 검색의 첫 실측.

README 한계에 "HyDE나 동의어 확장이 정공법이지만 둘 다 미측정"으로 남아
있던 항목이다. 표적 실패는 명확하다 — "세그멘테이션 모델 구현에 사용한
딥러닝 프레임워크는?"의 정답 청크는 "• Tensorflow 2.8.0"뿐이라 질의
토큰과 겹침이 0개(BM25 구조적 실패)고 임베딩도 안 붙는 유일한 MISS다.

검색 층만 잰다 — HYDE는 순수하게 검색 신호를 바꾸는 손잡이라, 검색이
안 움직이면 생성까지 잴 이유가 없다(움직이면 그때 evaluate.py로 간다).
두 지표를 같이 본다:

  1. retrieval_set.json (14문항, md5 gold) — gold_rank 이동을 문항별로
  2. eval_set.json 커버리지 (앵커 경로, 거부 제외) — satisfied@k/best_cov.
     recall류 지표는 "하나라도 회수"라 집계·열거의 실패를 숨기기 때문이다.

ON 조건은 질의당 LLM 호출 1회(가상 단락 생성)가 추가된다. temperature 0
이고 채점에 LLM이 없어 반복이 필요 없다 — 3B 생성 편차(±6%p)가 끼어들
자리가 검색 순위에는 없다. 가상 단락 자체를 결과 파일에 저장해 "왜
움직였나/안 움직였나"를 사후 감사할 수 있게 한다.

    python eval/ab_hyde.py
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from eval_coverage import anchor_chunks, score  # noqa: E402
from graph import hybrid_search, hypothetical_doc  # noqa: E402
from runmeta import run_metadata  # noqa: E402

RETRIEVAL_SET = Path(__file__).parent / "retrieval_set.json"
EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results_ab_hyde.json"
KS = [1, 3, 5, 6]


def retrieval_ranks(cases: list[dict]) -> tuple[dict[str, int | None], list[float]]:
    """질문 → gold 최고 순위(없으면 None), 검색 지연 목록."""
    ranks, secs = {}, []
    for case in cases:
        gold = {g["md5"] for g in case["gold"]}
        t0 = time.time()
        docs = hybrid_search(case["question"])
        secs.append(time.time() - t0)
        hashes = [hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
                  for d in docs]
        ranks[case["question"]] = next(
            (i + 1 for i, h in enumerate(hashes) if h in gold), None)
    return ranks, secs


def coverage_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        docs = hybrid_search(c["question"])
        got_all = [d.metadata.get("chunk_index") for d in docs]
        per_k = {}
        for k in KS:
            got = set(got_all[:k])
            sat, cov = score(c["paths"], got)
            per_k[f"@{k}"] = {"satisfied": sat, "best_cov": cov}
        rows.append({"question": c["question"], "type": c["type"],
                     "per_k": per_k})
    return rows


def agg_coverage(rows: list[dict]) -> dict:
    n = len(rows)
    return {f"@{k}": {
        "satisfied": round(sum(r["per_k"][f"@{k}"]["satisfied"] for r in rows) / n * 100),
        "best_cov": round(sum(r["per_k"][f"@{k}"]["best_cov"] for r in rows) / n * 100),
    } for k in KS}


def generate_ab(repeat: int, env: dict) -> None:
    """75문항 전체 파이프라인 A/B — off vs terms (생성 포함, 프로덕션 run() 경로).

    검색 층 결과(아래 main)가 좋아도 전체 정답률로 확인해야 채택할 수 있다 —
    Kiwi 토크나이저 때 검색 단독 지표와 전체 정답률이 반대로 갔던 전례.
    full 모드는 검색 층에서 이미 탈락(표적 못 맞히고 GPU 문항만 오염)이라
    여기서는 겨루지 않는다.
    """
    from evaluate import EVAL_SET, is_pass  # noqa: PLC0415
    import graph  # noqa: PLC0415

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    out = {"run_metadata": env, "repeat": repeat, "conditions": {}}
    for label, on in (("off", False), ("terms", True)):
        config.HYDE, config.HYDE_MODE = on, "terms"
        tally, secs = Counter(), []
        for i in range(repeat):
            for case in cases:
                t0 = time.time()
                r = graph.run(case["question"])
                secs.append(time.time() - t0)
                tally[case["question"]] += is_pass(r["answer"], case)
            n_pass = sum(
                1 for c in cases if tally[c["question"]] > i)
            print(f"  HYDE={label} {i + 1}/{repeat}: "
                  f"{n_pass}/{len(cases)}", flush=True)
        out["conditions"][label] = {
            "tally": dict(tally),
            "total": sum(tally.values()),
            "median_sec": round(sorted(secs)[len(secs) // 2], 2),
        }

    off, on = out["conditions"]["off"], out["conditions"]["terms"]
    n = len(cases) * repeat
    print(f"\n===== 전체 파이프라인 (75문항 × {repeat}회) =====")
    print(f"  off   {off['total']}/{n}  (지연 중앙값 {off['median_sec']}s)")
    print(f"  terms {on['total']}/{n}  (지연 중앙값 {on['median_sec']}s)")
    print("\n뒤집힌 문항:")
    for c in cases:
        q = c["question"]
        a, b = off["tally"].get(q, 0), on["tally"].get(q, 0)
        if a != b:
            print(f"  {a}/{repeat} → {b}/{repeat} | {q[:50]}")

    path = Path(__file__).parent / "results_ab_hyde_generate.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n저장: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="HYDE A/B")
    ap.add_argument("--generate", action="store_true",
                    help="검색 층 대신 75문항 전체 파이프라인 A/B (off vs terms)")
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    # 지문은 조건 루프가 config를 바꾸기 전에 찍는다 (audit_docs 관례)
    env = run_metadata()

    if args.generate:
        generate_ab(args.repeat, env)
        return 0

    ret_cases = json.loads(RETRIEVAL_SET.read_text(encoding="utf-8"))
    raw = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line)["page_content"] for line in f]

    cov_cases = []
    for c in raw:
        sets = c.get("gold_anchor_sets") or (
            [[a] for a in c["gold_anchors"]] if c.get("gold_anchors") else None)
        if not sets:
            continue                      # refusal — gold가 정의되지 않는다
        cov_cases.append({
            "question": c["question"], "type": c.get("type", "fact"),
            "paths": [[anchor_chunks(a, chunks) for a in path] for path in sets]})

    # (HYDE, HYDE_MODE) — full은 단락 전체로 벡터+BM25, terms는 질의에 없는
    # 영숫자 토큰만 BM25로. 같은 프로세스라 가상 단락 LLM 호출은 조건 간에
    # 캐시가 재사용된다(hypothetical_doc) — terms 조건은 LLM 비용이 0이다.
    conditions = [("off", False, "full"), ("full", True, "full"),
                  ("terms", True, "terms")]

    out = {"run_metadata": env, "conditions": {}}
    for label, on, mode in conditions:
        config.HYDE, config.HYDE_MODE = on, mode
        print(f"\n===== HYDE={label} =====", flush=True)
        ranks, secs = retrieval_ranks(ret_cases)
        rows = coverage_rows(cov_cases)
        out["conditions"][label] = {
            "ranks": ranks,
            "coverage": agg_coverage(rows),
            "coverage_rows": rows,
            "search_median_sec": round(statistics.median(secs), 2),
        }
        print(f"  검색 지연 중앙값 {out['conditions'][label]['search_median_sec']}s")

    # 가상 단락을 저장해 둔다 — "왜 순위가 움직였나"의 1차 근거
    out["hypothetical_docs"] = {
        c["question"]: hypothetical_doc(c["question"]) for c in ret_cases}

    off = out["conditions"]["off"]
    for label, _, _ in conditions[1:]:
        cond = out["conditions"][label]
        print(f"\n===== off vs {label} — 검색 단독 ({len(ret_cases)}문항, "
              f"gold_rank 이동) =====")
        moved = 0
        for q, r_off in off["ranks"].items():
            r_on = cond["ranks"][q]
            if r_off != r_on:
                moved += 1
                print(f"  {r_off or 'MISS':>4} → {r_on or 'MISS':<4} | {q[:44]}")
        if not moved:
            print("  이동 없음")

        print(f"===== off vs {label} — 커버리지 ({len(cov_cases)}문항) =====")
        print(f"{'':8}{'satisfied@k':>14}{'best_cov@k':>16}")
        for k in KS:
            o, n_ = off["coverage"][f"@{k}"], cond["coverage"][f"@{k}"]
            print(f"  k={k:<4}{o['satisfied']:>5}% → {n_['satisfied']:>3}%"
                  f"{o['best_cov']:>8}% → {n_['best_cov']:>3}%")
        print(f"검색 지연 중앙값: {off['search_median_sec']}s → "
              f"{cond['search_median_sec']}s (HYDE 조건은 질의당 LLM 1회 포함, "
              f"같은 프로세스 2회차부터는 캐시)")

    RESULTS.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
