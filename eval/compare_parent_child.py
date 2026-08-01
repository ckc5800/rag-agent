"""기본(단일 청크) vs Parent-Child — 검색 recall과 실제 답변을 나란히 비교.

gold 라벨(md5)은 기본 인덱스의 청크 경계에 묶여 있어 child 인덱스(118개,
전혀 다른 경계)에는 그대로 못 쓴다. sweep_chunk_size.py와 같은 방식으로
**정답 앵커 문자열**을 gold로 써서 두 구조를 공평하게 비교한다.

recall만으로는 parent-child의 요점(생성이 받는 맥락의 완결성)이 안
드러나므로, 실제 생성 답변도 나란히 찍는다.

    python eval/compare_parent_child.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

ANCHORS = {
    "TTS 프로젝트에서 TTFB를 얼마나 개선했나요?": ["2292"],
    "스트리밍 오디오의 팝 노이즈 문제를 어떻게 해결했나요?": ["Residual Buffer", "팝 노이즈"],
    "이윤선의 제1저자 논문은 몇 편인가요?": ["제1저자"],
    "등록된 특허 번호를 알려주세요.": ["2538225"],
    "Kubernetes 인프라 구축에 어떤 CI/CD 도구를 사용했나요?": ["ArgoCD"],
    "우수 논문상을 받은 논문 제목은?": ["우수논문상", "우수 논문상"],
    "화자 분할에는 어떤 모델을 사용했나요?": ["yannote"],
    "웹소켓 세션 탈취 문제는 어떻게 방어했나요?": ["password_changed_at", "세션 탈취"],
    "이윤선이 근무한 회사들을 알려주세요.": ["Experience Overview"],
}
KS = [1, 3, 6]


def random_baseline(n_chunks: int, gold_counts: list[int]) -> dict:
    base = {}
    for k in KS:
        tot = 0.0
        for g in gold_counts:
            miss = 1.0
            for j in range(k):
                miss *= max(0.0, n_chunks - g - j) / (n_chunks - j)
            tot += 1 - miss
        base[k] = round(tot / len(gold_counts) * 100)
    return base


def eval_retrieval(search_fn, all_chunks: list) -> dict:
    hits, ranks = {k: 0 for k in KS}, []
    for q, anchors in ANCHORS.items():
        docs = search_fn(q)
        rank = next((i + 1 for i, d in enumerate(docs)
                     if any(a in d.page_content for a in anchors)), None)
        ranks.append(rank)
        for k in KS:
            hits[k] += rank is not None and rank <= k
    n = len(ANCHORS)
    gold_counts = [sum(1 for c in all_chunks
                       if any(a in c.page_content for a in anc))
                   for anc in ANCHORS.values()]
    return {
        **{f"recall@{k}": round(hits[k] / n * 100) for k in KS},
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "baseline": random_baseline(len(all_chunks), gold_counts),
    }


def main():
    import json

    import graph
    import graph_parent_child as gpc

    if not Path(config.PARENT_DB_DIR).exists():
        raise SystemExit(
            "parent-child 인덱스가 없다. 먼저: python src/ingest_parent_child.py")

    base_chunks = [json.loads(l) for l in open(config.CHUNKS_PATH, encoding="utf-8")]
    base_chunks = [type("D", (), {"page_content": c["page_content"]})()
                   for c in base_chunks]
    child_chunks = [json.loads(l) for l in
                    open(config.PARENT_CHUNKS_PATH, encoding="utf-8")]
    child_chunks = [type("D", (), {"page_content": c["page_content"]})()
                    for c in child_chunks]

    print(f"기본 인덱스: {len(base_chunks)}청크 / child 인덱스: {len(child_chunks)}청크\n")

    print("=== 1. 검색 recall 비교 ===")
    r1 = eval_retrieval(graph.hybrid_search, base_chunks)
    r2 = eval_retrieval(gpc.hybrid_search_child, child_chunks)
    print(f"{'':<20}{'기본(단일 청크)':<20}{'parent-child(child 검색)'}")
    for k in KS:
        b1, b2 = r1["baseline"][k], r2["baseline"][k]
        print(f"  recall@{k:<12}{r1[f'recall@{k}']}% ({b1}%){'':<8}"
              f"{r2[f'recall@{k}']}% ({b2}%)")
    print(f"  MRR{'':<17}{r1['mrr']}{'':<20}{r2['mrr']}")

    print("\n=== 2. 실제 생성 답변 비교 (컨텍스트 완결성 확인용) ===")
    for q in list(ANCHORS)[:3]:
        print(f"\nQ: {q}")
        a1 = graph.ask(q)
        a2 = gpc.ask(q)
        print(f"  [기본]         {a1['answer'][:150]}")
        print(f"  [parent-child] {a2['answer'][:150]}")


if __name__ == "__main__":
    main()
