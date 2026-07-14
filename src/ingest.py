"""문서 인제스트 파이프라인: Markdown 로드 → 청킹 → 임베딩 → FAISS 인덱스 저장."""
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(Path(config.DOCS_DIR).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit(f"문서가 없습니다: {config.DOCS_DIR}")
    print(f"{len(docs)}개 문서 로드")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)}개 청크 생성")

    # 기존 DB 삭제 후 재구축 (멱등성 보장)
    if Path(config.DB_DIR).exists():
        shutil.rmtree(config.DB_DIR)

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config.DB_DIR)
    print(f"FAISS 인덱스 저장 완료: {config.DB_DIR}")

    # BM25(키워드 검색) 재구축용 청크 원문 저장
    import json
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"page_content": c.page_content, "metadata": c.metadata},
                ensure_ascii=False) + "\n")
    print(f"청크 저장 완료: {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
