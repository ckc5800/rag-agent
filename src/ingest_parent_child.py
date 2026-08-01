"""Parent-Child 청킹 인제스트 — 검색은 작게, 생성은 크게.

기본 파이프라인(ingest.py)은 검색과 생성이 같은 청크 크기(800자)를
공유한다. 검색엔 작을수록 정밀하고(주제 하나만 담겨 매칭이 명확) 생성엔
클수록 유리한데(문맥이 안 잘림), 하나의 크기로는 둘 다 타협하게 된다.

Parent-Child는 이 타협을 없앤다:
    문서 → parent(2000자, 큰 문맥 단위) → child(300자, 검색용 정밀 단위)
    검색은 child 벡터로 하고, 찾은 child의 parent_id로 parent 원문을 가져와
    생성에 넘긴다.

기존 hybrid_search()·grade()·generate() 로직은 그대로 두고, "검색 결과를
parent로 확장하는" 레이어만 이 파일과 graph_parent_child.py에 얹는다.
정제(clean_markdown)와 다이어그램 분리(extract_diagrams)는 ingest.py 것을
그대로 재사용한다 — 콘텐츠 정제는 청킹 전략과 무관한 별개 관심사다.

    python src/ingest_parent_child.py
"""
import json
import sys
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import vectorstore
from ingest import clean_markdown, extract_diagrams

SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]


def build_parent_child(docs: list[Document]) -> tuple[list[Document], list[Document]]:
    """(parents, children) — child.metadata['parent_id']로 짝을 찾는다."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_SIZE, chunk_overlap=config.PARENT_OVERLAP,
        separators=SEPARATORS)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_SIZE, chunk_overlap=config.CHILD_OVERLAP,
        separators=SEPARATORS)

    parents = parent_splitter.split_documents(docs)
    parents = [p for p in parents if len(p.page_content) >= config.MIN_CHUNK_CHARS]

    children = []
    for parent in parents:
        parent_id = str(uuid.uuid4())
        parent.metadata["parent_id"] = parent_id
        for child in child_splitter.split_documents([parent]):
            if len(child.page_content) < config.MIN_CHUNK_CHARS:
                continue
            child.metadata["parent_id"] = parent_id
            children.append(child)
    return parents, children


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

    parents, children = build_parent_child(docs)

    # 다이어그램은 parent-child 분할 대상이 아니다 — 이미 통짜 청크이므로
    # parent 취급하되(부모=자신) child도 자기 자신 하나로 둔다. 검색 정밀도를
    # 얻을 여지가 없는 콘텐츠라 child를 더 쪼개는 이득이 없다.
    for d in diagrams:
        d.metadata["parent_id"] = str(uuid.uuid4())
        parents.append(d)
        children.append(d)

    print(f"parent {len(parents)}개 → child {len(children)}개")
    plen = sorted(len(p.page_content) for p in parents)
    clen = sorted(len(c.page_content) for c in children)
    print(f"  parent 길이: 최소 {plen[0]} / 중앙값 {plen[len(plen)//2]} / 최대 {plen[-1]}")
    print(f"  child  길이: 최소 {clen[0]} / 중앙값 {clen[len(clen)//2]} / 최대 {clen[-1]}")

    # child만 임베딩해서 검색 인덱스에 넣는다. 이 실험은 청킹 전략 비교가
    # 목적이라 벡터 저장소는 FAISS로 고정한다 — VECTOR_STORE=qdrant인 채로
    # 돌리면 기본 파이프라인의 Qdrant 컬렉션과 경로가 겹칠 위험이 있다.
    config.DB_DIR = config.PARENT_DB_DIR
    path = vectorstore.build(children, kind="faiss")
    print(f"child 검색 인덱스 저장: {path}")

    # BM25 재구축용 — child 원문 (기본 파이프라인의 chunks.jsonl과 같은 형식)
    with open(config.PARENT_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps(
                {"page_content": c.page_content, "metadata": c.metadata},
                ensure_ascii=False) + "\n")
    print(f"child 청크 저장: {config.PARENT_CHUNKS_PATH}")

    # parent_id → 원문 조회용. child 검색 결과를 이걸로 확장한다.
    with open(config.PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(
                {"parent_id": p.metadata["parent_id"],
                 "page_content": p.page_content, "metadata": p.metadata},
                ensure_ascii=False) + "\n")
    print(f"parent 저장소: {config.PARENT_STORE_PATH}")


if __name__ == "__main__":
    sys.exit(main())
