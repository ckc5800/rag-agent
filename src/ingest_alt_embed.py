"""대안 임베딩 모델(config.ALT_EMBED_MODEL)로 별도 인덱스를 만든다.

청킹(800자 고정)은 base와 완전히 동일하게 두고 임베딩 모델만 바꾼다 —
변수 하나만 바꿔야 recall 차이가 청킹이 아니라 임베딩 때문이라고 말할
수 있다.

    ollama pull nomic-embed-text     # 먼저 모델을 받아야 한다
    python src/ingest_alt_embed.py
"""
import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import vectorstore
from ingest import clean_markdown, extract_diagrams


def main():
    docs, diagrams = [], []
    for path in sorted(Path(config.DOCS_DIR).glob("*.md")):
        text = clean_markdown(path.read_text(encoding="utf-8"))
        body, diag_docs = extract_diagrams(text)
        docs.append(Document(page_content=body, metadata={"source": path.name}))
        for d in diag_docs:
            d.metadata["source"] = path.name
        diagrams.extend(diag_docs)
    print(f"{len(docs)}개 문서 로드 (다이어그램 {len(diagrams)}개 분리)")

    # base와 완전히 같은 청킹 — 임베딩 모델만 변수로 남긴다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "])
    chunks = [c for c in splitter.split_documents(docs)
             if len(c.page_content) >= config.MIN_CHUNK_CHARS]
    chunks += diagrams
    print(f"총 {len(chunks)}개 청크 (base와 동일한 청킹)")

    config.EMBED_MODEL = config.ALT_EMBED_MODEL
    config.DB_DIR = config.ALT_EMBED_DB_DIR
    path = vectorstore.build(chunks, kind="faiss")
    print(f"[{config.ALT_EMBED_MODEL}] 인덱스 저장: {path}")

    with open(config.ALT_EMBED_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"page_content": c.page_content, "metadata": c.metadata},
                ensure_ascii=False) + "\n")
    print(f"청크 저장: {config.ALT_EMBED_CHUNKS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
