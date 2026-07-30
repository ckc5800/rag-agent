"""검색 품질 단독 평가 — retrieval recall@k / MRR.

답변 정확도(evaluate.py)는 검색과 생성의 실패가 섞여서, 어느 쪽이
문제인지 지표만으로는 알 수 없었다. 이 스크립트는 질문별로 정답이 담긴
청크(gold)를 라벨링해 두고(retrieval_set.json, 내용 md5로 고정),
프로덕션과 같은 hybrid_search가 gold를 top-k 안에 올리는지만 잰다.

LLM 생성이 없어서 결정적이고, 임베딩 호출만 있어 수 초 안에 끝난다.
청킹/검색 파라미터를 바꿀 때 이 지표부터 확인하면 실패 원인을
검색/생성으로 분리할 수 있다.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
from graph import hybrid_search  # noqa: E402

RETRIEVAL_SET = Path(__file__).parent / "retrieval_set.json"
RESULTS = Path(__file__).parent / "results_retrieval.json"
KS = [1, 3, 6]  # 6 = config.TOP_K (generate에 전달되는 전부)


def random_baseline(cases: list[dict]) -> dict:
    """무작위로 top-k를 뽑았을 때의 기대 recall.

    gold를 질문당 여러 개 두면 우연히 맞을 확률이 올라간다. 기준선을
    같이 보지 않으면 recall 수치가 혼자서는 아무것도 증명하지 못한다.
    """
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        n_chunks = sum(1 for _ in f)
    out = {}
    for k in KS:
        total = 0.0
        for case in cases:
            g = len(case["gold"])
            p_miss = 1.0
            for j in range(k):                      # gold를 하나도 못 뽑을 확률
                p_miss *= max(0.0, (n_chunks - g - j)) / (n_chunks - j)
            total += 1 - p_miss
        out[f"recall@{k}"] = round(total / len(cases) * 100)
    return out


def main():
    cases = json.loads(RETRIEVAL_SET.read_text(encoding="utf-8"))

    rows = []
    for case in cases:
        gold_md5 = {g["md5"] for g in case["gold"]}
        docs = hybrid_search(case["question"])
        ranks = [hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
                 for d in docs]
        first = next((i + 1 for i, h in enumerate(ranks) if h in gold_md5), None)
        rows.append({
            "question": case["question"],
            "gold_rank": first,
            "hits": {f"@{k}": (first is not None and first <= k) for k in KS},
        })
        mark = f"rank {first}" if first else "MISS"
        print(f"[{mark:>6}] {case['question']}")

    summary = {
        f"recall@{k}": round(
            sum(r["hits"][f"@{k}"] for r in rows) / len(rows) * 100) for k in KS}
    summary["mrr"] = round(
        sum(1 / r["gold_rank"] for r in rows if r["gold_rank"]) / len(rows), 3)
    summary["random_baseline"] = random_baseline(cases)

    print("\n===== 검색 단독 평가 =====")
    for k in KS:
        base = summary["random_baseline"].get(f"recall@{k}")
        print(f"recall@{k} : {summary[f'recall@{k}']}%"
              f"   (무작위 기준선 {base}%)")
    print(f"MRR      : {summary['mrr']}")

    RESULTS.write_text(json.dumps({"summary": summary, "cases": rows},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")


if __name__ == "__main__":
    main()
