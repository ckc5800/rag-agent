"""시맨틱 청킹 인제스트 — 별도 인덱스로 기본 파이프라인과 나란히 비교한다.

정제(clean_markdown)·다이어그램 분리(extract_diagrams)는 ingest.py 것을
그대로 재사용한다 — 콘텐츠 정제는 청킹 전략과 무관한 별개 관심사다
(ingest_parent_child.py와 같은 원칙).

    python src/ingest_semantic.py
"""
import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config
import vectorstore
from ingest import clean_markdown, extract_diagrams
from semantic_chunk import chunk_semantically


def main():
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)

    docs, diagrams = [], []
    for path in sorted(Path(config.DOCS_DIR).glob("*.md")):
        text = clean_markdown(path.read_text(encoding="utf-8"))
        body, diag_docs = extract_diagrams(text)
        docs.append(Document(page_content=body, metadata={"source": path.name}))
        for d in diag_docs:
            d.metadata["source"] = path.name
        diagrams.extend(diag_docs)
    print(f"{len(docs)}개 문서 로드 (다이어그램 {len(diagrams)}개 분리)")

    chunks: list[Document] = []
    for doc in docs:
        texts = chunk_semantically(
            doc.page_content, embeddings.embed_documents,
            percentile=config.SEMANTIC_PERCENTILE,
            max_chars=config.SEMANTIC_MAX_CHARS,
            min_chars=config.SEMANTIC_MIN_CHARS)
        for t in texts:
            chunks.append(Document(page_content=t, metadata=dict(doc.metadata)))
    print(f"산문 {len(chunks)}개 청크")
    if chunks:
        lens = sorted(len(c.page_content) for c in chunks)
        print(f"  길이: 최소 {lens[0]} / 중앙값 {lens[len(lens)//2]} / "
              f"최대 {lens[-1]}")

    chunks += diagrams   # 다이어그램은 시맨틱 분할 대상이 아니다 (통짜 보존)
    print(f"다이어그램 포함 총 {len(chunks)}개 청크")

    # 이 실험은 청킹 전략 비교가 목적이라 벡터 저장소를 FAISS로 고정한다
    # (parent-child와 같은 이유 — Qdrant 컬렉션 충돌 방지).
    config.DB_DIR = config.SEMANTIC_DB_DIR
    path = vectorstore.build(chunks, kind="faiss")
    print(f"인덱스 저장: {path}")

    with open(config.SEMANTIC_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"page_content": c.page_content, "metadata": c.metadata},
                ensure_ascii=False) + "\n")
    print(f"청크 저장: {config.SEMANTIC_CHUNKS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
