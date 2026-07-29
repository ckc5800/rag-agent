"""청크 크기를 바꿔가며 검색 성능을 잰다.

지금까지 800자는 **튜닝한 값이 아니라 합리적인 기본값**이었다. 실제로
스윕한 적이 없어서, 더 잘게/크게 자르면 어떻게 되는지 확인한다.

문제: gold 라벨이 청크 내용 md5라 크기를 바꾸면 전부 무효화된다.
해결: 크기와 무관한 **정답 앵커 문자열**로 gold를 정의한다. 어떤 크기로
      잘라도 "그 문자열을 담은 청크"가 정답이므로 라벨링이 필요 없다.
      앵커는 각각 현재 인덱스에서 1~4청크에만 등장하도록 좁게 골랐다.

    python eval/sweep_chunk_size.py                    # 기본 스윕
    python eval/sweep_chunk_size.py --sizes 400 800    # 일부만
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

# 질문 → 정답이 반드시 담겨 있어야 하는 문자열(하나라도 포함하면 gold)
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


def build_and_eval(size: int, overlap: int) -> dict:
    """청크 크기를 바꿔 재인제스트하고 검색만 평가한다."""
    config.CHUNK_SIZE = size
    config.CHUNK_OVERLAP = overlap

    # config를 바꾼 뒤 매번 새로 임포트해야 반영된다
    for m in ("ingest", "graph", "vectorstore"):
        sys.modules.pop(m, None)
    import ingest  # noqa: PLC0415
    import graph   # noqa: PLC0415

    # 다이어그램은 크기를 바꿔도 항상 같은 통짜 청크라, 넣으면 median/mean이
    # "산문이 이 크기에서 어떤가"가 아니라 다이어그램 유무로 흔들린다.
    # recall 비교에는 넣되(검색 대상이어야 하므로) 길이 통계에서는 뺀다.
    docs, diagrams = ingest.load_documents()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    prose_chunks = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    ).split_documents(docs)
    prose_chunks = [c for c in prose_chunks
                    if len(c.page_content) >= config.MIN_CHUNK_CHARS]
    chunks = prose_chunks + diagrams

    import vectorstore as vs  # noqa: PLC0415
    vs.build(chunks)

    # BM25도 같은 청크로 다시 만들어야 하므로 chunks.jsonl 갱신
    import json
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"page_content": c.page_content,
                                "metadata": c.metadata}, ensure_ascii=False) + "\n")

    graph._vectorstore = None      # 캐시 무효화
    graph._bm25 = None

    lengths = sorted(len(c.page_content) for c in prose_chunks)  # 산문만
    hits, ranks = {k: 0 for k in KS}, []
    for q, anchors in ANCHORS.items():
        docs_out = graph.hybrid_search(q)
        rank = next((i + 1 for i, d in enumerate(docs_out)
                     if any(a in d.page_content for a in anchors)), None)
        ranks.append(rank)
        for k in KS:
            hits[k] += rank is not None and rank <= k

    n = len(ANCHORS)

    # 무작위 기준선 — 청크가 커지면 개수가 줄어 우연히 맞을 확률이 올라간다.
    # 이걸 같이 보지 않으면 "큰 청크가 유리하다"는 착시에 빠진다.
    N = len(chunks)
    gold_counts = [sum(1 for c in chunks
                       if any(a in c.page_content for a in anc))
                   for anc in ANCHORS.values()]
    base = {}
    for k in KS:
        tot = 0.0
        for g in gold_counts:
            miss = 1.0
            for j in range(k):
                miss *= max(0.0, N - g - j) / (N - j)
            tot += 1 - miss
        base[f"recall@{k}"] = round(tot / n * 100)

    return {
        "size": size, "chunks": N,
        "median": lengths[len(lengths) // 2], "mean": sum(lengths) // len(lengths),
        **{f"recall@{k}": round(hits[k] / n * 100) for k in KS},
        "baseline": base,
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "misses": [q[:22] for q, r in zip(ANCHORS, ranks) if r is None or r > 3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[300, 500, 800, 1200, 1600])
    args = ap.parse_args()

    rows = []
    for size in args.sizes:
        overlap = max(20, size // 8)          # 원래 비율(800:100)을 유지
        print(f"청크 {size}자(overlap {overlap}) 구축·평가 중...")
        rows.append(build_and_eval(size, overlap))

    print("\n===== 청크 크기 스윕 (검색 단독, 앵커 기반 gold) =====")
    print("  괄호 안은 무작위 기준선 — 청크가 커지면 개수가 줄어 기준선이 올라간다.")
    print("  크기  청크수  중앙값 |   recall@1      recall@3      recall@6    MRR")
    for r in rows:
        b = r["baseline"]
        print("  {:>4}  {:>5}  {:>5} | {:>3}% ({:>2}%)  {:>3}% ({:>2}%)  {:>3}% ({:>2}%)  {:.3f}".format(
            r["size"], r["chunks"], r["median"],
            r["recall@1"], b["recall@1"], r["recall@3"], b["recall@3"],
            r["recall@6"], b["recall@6"], r["mrr"]))

    print("\n  top-3 밖으로 밀린 질문 (generate가 못 받는 것):")
    for r in rows:
        print("   {:>4}자: {}".format(r["size"], ", ".join(r["misses"]) or "없음"))

    import json
    out = Path(__file__).parent / "results_chunk_sweep.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")
    print("주의: 이 스크립트는 인덱스와 chunks.jsonl을 덮어쓴다. "
          "끝나면 `python src/ingest.py`로 기본 설정(800자) 복구할 것.")


if __name__ == "__main__":
    main()
