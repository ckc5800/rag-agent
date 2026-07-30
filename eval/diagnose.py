"""실패 원인 분해 — 검색인가, 청킹인가, 생성인가.

정답률 하나로는 "어디서부터 문제인지" 알 수 없다. 이 스크립트는 41개
답변 가능 문항(gold_anchors 보유)에 대해 층을 나눠 잰다. **LLM을 쓰지
않는다** — 임베딩 호출만 있어 수십 초면 끝나고 결정적이다.

  1. 검색 층 — gold가 top-k에 오는가
       · hybrid  : 프로덕션과 동일 (FAISS + BM25 RRF)
       · vector  : 임베딩(bge-m3) 단독  ← 임베딩이 문제인지
       · bm25    : 키워드 단독          ← 키워드가 문제인지
     세 경로를 나란히 보면 "임베딩이 못 찾는데 BM25는 찾는다"처럼
     책임이 갈린다.

  2. 청킹 층 — 근거가 어떤 청크에 어떻게 들어 있는가
       · 앵커를 담은 청크가 몇 개인가 (0이면 청킹이 근거를 파괴한 것)
       · 그 청크의 길이·출처
       · **앵커가 청크 경계에 걸쳐 잘렸는가**는 앵커가 사라진 것으로 드러난다

  3. 생성 층 — 검색은 됐는데 답이 틀렸는가
     저장된 평가 결과(results.json)와 교차해 4분면으로 가른다.
       검색O·정답O = 정상 / 검색O·정답X = **생성 문제**
       검색X·정답X = **검색 문제** / 검색X·정답O = 모델이 알고 있었음

    python eval/diagnose.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import graph  # noqa: E402

EVAL_SET = Path(__file__).parent / "eval_set.json"
RESULTS = Path(__file__).parent / "results.json"
OUT = Path(__file__).parent / "results_diagnose.json"
KS = (1, 3, 6)


def rank_of(docs, anchors: list[str]) -> int | None:
    for i, d in enumerate(docs, 1):
        if any(a in d.page_content for a in anchors):
            return i
    return None


def main() -> int:
    cases = [c for c in json.loads(EVAL_SET.read_text(encoding="utf-8"))
             if c.get("gold_anchors")]
    chunks = [json.loads(line) for line
              in Path(config.CHUNKS_PATH).read_text(
                  encoding="utf-8").splitlines() if line.strip()]

    passed = {}
    if RESULTS.exists():
        saved = json.loads(RESULTS.read_text(encoding="utf-8"))
        passed = {r["question"]: r["pass"] for r in saved.get("cases", [])}

    store, bm25 = graph._load_indexes()

    rows = []
    print(f"{len(cases)}문항 · 코퍼스 {len(chunks)}청크 (LLM 미사용)\n")
    print(f"{'유형':<12} {'hyb':>4} {'vec':>4} {'bm25':>5} {'청크':>4}  질문")
    for c in cases:
        anchors = c["gold_anchors"]
        q = c["question"]

        hyb = rank_of(graph.hybrid_search(q), anchors)
        vec = rank_of(store.similarity_search(q, k=config.TOP_K), anchors)
        kw = rank_of(bm25.invoke(q), anchors)

        owning = [ch for ch in chunks
                  if any(a in ch["page_content"] for a in anchors)]

        def fmt(r):
            return str(r) if r else "-"

        print(f"{c.get('type', '?'):<12} {fmt(hyb):>4} {fmt(vec):>4} "
              f"{fmt(kw):>5} {len(owning):>4}  {q[:40]}")

        rows.append({
            "question": q, "type": c.get("type"),
            "anchors": anchors,
            "rank_hybrid": hyb, "rank_vector": vec, "rank_bm25": kw,
            "in_top3": bool(hyb and hyb <= 3),
            "gold_chunks": len(owning),
            "gold_chunk_lens": sorted(len(ch["page_content"]) for ch in owning),
            "gold_sources": sorted({ch["metadata"]["source"] for ch in owning}),
            "answer_pass": passed.get(q),
        })

    # ── 1. 검색 층 ──────────────────────────────────────
    def recall(key: str, k: int) -> float:
        hit = sum(1 for r in rows if r[key] and r[key] <= k)
        return hit / len(rows) * 100

    print("\n===== 1. 검색 층 =====")
    print(f"{'':<10} " + "".join(f"recall@{k:<5}" for k in KS))
    for key, label in (("rank_hybrid", "hybrid"), ("rank_vector", "vector"),
                       ("rank_bm25", "bm25")):
        print(f"{label:<10} " + "".join(
            f"{recall(key, k):>6.0f}%   " for k in KS))
    print("  (vector=임베딩 단독, bm25=키워드 단독. 두 수치가 갈리면"
          " 책임이 어느 쪽인지 보인다)")

    only_kw = [r for r in rows
               if not (r["rank_vector"] and r["rank_vector"] <= 3)
               and (r["rank_bm25"] and r["rank_bm25"] <= 3)]
    only_vec = [r for r in rows
                if (r["rank_vector"] and r["rank_vector"] <= 3)
                and not (r["rank_bm25"] and r["rank_bm25"] <= 3)]
    print(f"\n  임베딩은 놓쳤는데 BM25가 top-3에 올린 문항: {len(only_kw)}")
    for r in only_kw:
        print(f"     · {r['question'][:52]}")
    print(f"  BM25는 놓쳤는데 임베딩이 top-3에 올린 문항: {len(only_vec)}")
    for r in only_vec:
        print(f"     · {r['question'][:52]}")

    # ── 2. 청킹 층 ──────────────────────────────────────
    print("\n===== 2. 청킹 층 =====")
    lost = [r for r in rows if r["gold_chunks"] == 0]
    print(f"  근거 앵커를 담은 청크가 0개인 문항: {len(lost)}"
          "   ← 청킹이 근거를 잘라 없앤 경우")
    for r in lost:
        print(f"     · {r['question'][:52]}  {r['anchors']}")
    spread = [r for r in rows if r["gold_chunks"] >= 3]
    print(f"  근거가 3개 이상 청크에 흩어진 문항: {len(spread)}"
          "   ← 열거·집계가 어려워지는 조건")
    for r in spread:
        print(f"     · {r['gold_chunks']}청크 {r['gold_sources']} "
              f"{r['question'][:40]}")

    # ── 3. 생성 층 (저장된 정답률과 교차) ────────────────
    if passed:
        print("\n===== 3. 검색 × 생성 4분면 =====")
        quad = {"검색O·정답O": [], "검색O·정답X": [],
                "검색X·정답O": [], "검색X·정답X": []}
        for r in rows:
            if r["answer_pass"] is None:
                continue
            key = ("검색O" if r["in_top3"] else "검색X") + \
                  ("·정답O" if r["answer_pass"] else "·정답X")
            quad[key].append(r)
        for k, v in quad.items():
            print(f"  {k}: {len(v)}건")
        print("\n  [검색O·정답X] = 생성 문제 (근거를 받고도 틀림)")
        for r in quad["검색O·정답X"]:
            print(f"     · ({r['type']}) rank {r['rank_hybrid']}  "
                  f"{r['question'][:46]}")
        print("\n  [검색X·정답X] = 검색 문제 (근거가 생성기에 안 감)")
        for r in quad["검색X·정답X"]:
            print(f"     · ({r['type']}) hybrid rank {r['rank_hybrid']} / "
                  f"vec {r['rank_vector']} / bm25 {r['rank_bm25']}  "
                  f"{r['question'][:40]}")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
