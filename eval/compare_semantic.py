"""base(고정 800자) vs 시맨틱 청킹 — 검색 recall 비교.

sweep_chunk_size.py와 같은 앵커 기반 gold를 재사용한다(청크 경계가
전혀 달라 md5 gold를 못 쓴다). base 인덱스는 그대로 두고 시맨틱
인덱스만 별도로 읽는다 — sweep_chunk_size.py와 달리 이 스크립트는
아무 것도 덮어쓰지 않는다.

    python eval/compare_semantic.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

# sweep_chunk_size.py와 동일한 앵커 — 두 실험 결과를 나란히 볼 수 있게.
ANCHORS = {
    "TTS 프로젝트에서 TTFB를 얼마나 개선했나요?": ["2292"],
    "스트리밍 오디오의 팝 노이즈 문제를 어떻게 해결했나요?": ["Residual Buffer", "팝 노이즈"],
    "이윤선의 제1저자 논문은 몇 편인가요?": ["제1저자"],
    "등록된 특허 번호를 알려주세요.": ["2538225"],
    "Kubernetes 인프라 구축에 어떤 CI/CD 도구를 사용했나요?": ["ArgoCD"],
    "우수 논문상을 받은 논문 제목은?": ["우수논문상", "우수 논문상"],
    "TTS 시스템의 동시 처리 채널은 몇 개로 확장했나요?": ["12 → 24"],
    "화자 분할에는 어떤 모델을 사용했나요?": ["yannote"],
    "웹소켓 세션 탈취 문제는 어떻게 방어했나요?": ["password_changed_at", "세션 탈취"],
    "이윤선이 근무한 회사들을 알려주세요.": ["Experience Overview"],
}
KS = [1, 3, 6]


def hybrid_search_semantic(query: str, store, bm25):
    vec_docs = store.similarity_search(query, k=config.TOP_K)
    kw_docs = bm25.invoke(query)
    K = 60
    scores, by_key = {}, {}
    for docs in (vec_docs, kw_docs):
        for rank, doc in enumerate(docs):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_key[k] for k in ranked[:config.TOP_K]]


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
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings

    import graph

    if not Path(config.SEMANTIC_DB_DIR).exists():
        raise SystemExit(
            "시맨틱 인덱스가 없다. 먼저: python src/ingest_semantic.py")

    base_chunks = [json.loads(l) for l in
                   open(config.CHUNKS_PATH, encoding="utf-8")]
    base_chunks = [type("D", (), {"page_content": c["page_content"]})()
                   for c in base_chunks]
    sem_rows = [json.loads(l) for l in
                open(config.SEMANTIC_CHUNKS_PATH, encoding="utf-8")]
    sem_chunks = [type("D", (), {"page_content": c["page_content"]})()
                  for c in sem_rows]

    sem_store = FAISS.load_local(
        config.SEMANTIC_DB_DIR, OllamaEmbeddings(model=config.EMBED_MODEL),
        allow_dangerous_deserialization=True)
    sem_docs = [Document(page_content=r["page_content"], metadata=r["metadata"])
                for r in sem_rows]
    sem_bm25 = BM25Retriever.from_documents(sem_docs, preprocess_func=graph.bm25_tokenize)
    sem_bm25.k = config.TOP_K

    print(f"기본 인덱스: {len(base_chunks)}청크 / 시맨틱 인덱스: {len(sem_chunks)}청크\n")

    r1 = eval_retrieval(graph.hybrid_search, base_chunks)
    r2 = eval_retrieval(
        lambda q: hybrid_search_semantic(q, sem_store, sem_bm25), sem_chunks)

    print(f"{'':<20}{'기본(고정 800자)':<20}{'시맨틱 청킹'}")
    for k in KS:
        b1, b2 = r1["baseline"][k], r2["baseline"][k]
        print(f"  recall@{k:<12}{r1[f'recall@{k}']}% ({b1}%){'':<8}"
              f"{r2[f'recall@{k}']}% ({b2}%)")
    print(f"  MRR{'':<17}{r1['mrr']}{'':<20}{r2['mrr']}")

    out = Path(__file__).parent / "results_semantic.json"
    out.write_text(json.dumps({"base": r1, "semantic": r2}, ensure_ascii=False,
                              indent=2), encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
