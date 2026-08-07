"""Graph RAG 인덱스 구축 — 트리플 추출(LLM) → 그래프 저장.

전제: `python src/ingest.py`가 먼저 실행되어 data/chunks.jsonl이 있어야
한다. 청킹은 base 그대로 재사용한다 — 이건 청킹 실험이 아니라 검색 방식
실험이라, 변수를 하나로 유지한다.

사용: python src/ingest_kg.py
"""
import json
import time

from langchain_core.documents import Document

import config
import kg


def load_chunks() -> list[Document]:
    docs = []
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            docs.append(Document(page_content=d["page_content"], metadata=d["metadata"]))
    return docs


def main():
    chunks = load_chunks()
    print(f"{len(chunks)}개 청크 로드 — 트리플 추출 시작 "
          f"(LLM {config.LLM_MODEL}, 청크당 1회 호출)")

    t0 = time.time()
    g, stats = kg.build_graph(chunks)
    elapsed = time.time() - t0

    empty_pct = stats["chunks_empty"] / max(stats["chunks_seen"], 1) * 100
    print(f"청크 {stats['chunks_seen']}개 처리, 트리플 0개인 청크 "
          f"{stats['chunks_empty']}개 ({empty_pct:.0f}%)")
    print(f"트리플 {stats['triples']}개 → 노드 {g.number_of_nodes()}개, "
          f"엣지 {g.number_of_edges()}개 ({elapsed:.1f}초)")

    if g.number_of_edges() == 0:
        raise SystemExit("트리플이 하나도 추출되지 않았습니다 — LLM_MODEL·Ollama 상태를 확인하세요.")

    kg.save(g)
    print(f"그래프 저장: {kg.KG_PATH}")


if __name__ == "__main__":
    main()
