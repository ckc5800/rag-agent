"""대안 임베딩 모델(config.ALT_EMBED_MODEL)로 별도 인덱스를 만든다.

청킹(800자 고정)은 base와 완전히 동일하게 두고 임베딩 모델만 바꾼다 —
변수 하나만 바꿔야 recall 차이가 청킹이 아니라 임베딩 때문이라고 말할
수 있다.

    ollama pull nomic-embed-text     # 먼저 모델을 받아야 한다
    python src/ingest_alt_embed.py
"""
import json
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import vectorstore
from ingest import load_documents


def main():
    # base(ingest.py)와 같은 load_documents()를 그대로 재사용한다 — PDF·DOCX·
    # HWPX 로더 추가(loaders.py) 이후 여기 자체 markdown-only 로직이 낡아
    # base(58청크)와 청크 수가 어긋난 채(51청크) 비교하던 결함을 고쳤다.
    # 임베딩 모델 외의 변수(코퍼스 구성)가 섞이면 recall 차이의 원인을
    # 임베딩으로 단정할 수 없다.
    docs, diagrams = load_documents()
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
