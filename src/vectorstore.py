"""벡터 저장소 교체 레이어 — FAISS / Qdrant.

같은 임베딩(bge-m3)에 같은 exact 검색이므로 **recall은 동일해야 한다.**
바꿔 낀 이유는 성능이 아니라 기능 차이를 직접 확인하기 위해서다:

    FAISS   파일 2개(index.faiss/index.pkl)로 끝난다. 가장 가볍다.
            메타데이터 필터가 없다 → 검색 후 파이썬에서 걸러야 하고,
            그러면 top-k가 필터 후 부족해질 수 있다.
    Qdrant  payload에 메타데이터를 같이 저장하고 필터를 검색에 넣는다.
            "resume.md 안에서만" 같은 질의가 되고, 필터를 적용한 채로
            top-k를 채운다. 서버 없이 로컬 경로로도 돈다(임베디드).

교체는 환경변수 하나다:  VECTOR_STORE=qdrant python src/ingest.py
"""
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL)


# ── 구축 ────────────────────────────────────────────────

def build(chunks: list[Document], kind: str | None = None) -> str:
    """청크를 임베딩해 저장소에 적재하고, 저장 위치를 반환한다."""
    kind = kind or config.VECTOR_STORE
    if kind == "faiss":
        import shutil
        from pathlib import Path

        from langchain_community.vectorstores import FAISS

        if Path(config.DB_DIR).exists():
            shutil.rmtree(config.DB_DIR)          # 멱등성
        FAISS.from_documents(chunks, _embeddings()).save_local(config.DB_DIR)
        return config.DB_DIR

    if kind == "qdrant":
        import shutil
        from pathlib import Path

        from langchain_qdrant import QdrantVectorStore

        if Path(config.QDRANT_PATH).exists():
            shutil.rmtree(config.QDRANT_PATH)
        QdrantVectorStore.from_documents(
            chunks,
            _embeddings(),
            path=config.QDRANT_PATH,              # 서버 대신 로컬 경로(임베디드)
            collection_name=config.QDRANT_COLLECTION,
        )
        return config.QDRANT_PATH

    raise ValueError(f"알 수 없는 VECTOR_STORE: {kind}")


# ── 로드 ────────────────────────────────────────────────

def load(kind: str | None = None):
    kind = kind or config.VECTOR_STORE
    if kind == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.load_local(
            config.DB_DIR, _embeddings(),
            allow_dangerous_deserialization=True,  # 로컬에서 직접 만든 인덱스만
        )

    if kind == "qdrant":
        from langchain_qdrant import QdrantVectorStore

        return QdrantVectorStore.from_existing_collection(
            path=config.QDRANT_PATH,
            collection_name=config.QDRANT_COLLECTION,
            embedding=_embeddings(),
        )

    raise ValueError(f"알 수 없는 VECTOR_STORE: {kind}")


# ── 검색 ────────────────────────────────────────────────

def search(store, query: str, k: int, source: str | None = None) -> list[Document]:
    """의미 검색. source를 주면 그 문서 안에서만 찾는다.

    필터가 두 저장소의 실질적 차이다. FAISS는 필터를 못 받으므로 넉넉히
    뽑아 파이썬에서 걸러야 하고(그래도 k를 못 채울 수 있다), Qdrant는
    필터를 검색에 넣어 **필터를 적용한 채로 k를 채운다.**
    """
    if source is None:
        return store.similarity_search(query, k=k)

    if config.VECTOR_STORE == "qdrant":
        from qdrant_client.http import models as rest

        flt = rest.Filter(must=[rest.FieldCondition(
            key="metadata.source", match=rest.MatchValue(value=source))])
        return store.similarity_search(query, k=k, filter=flt)

    # FAISS: 사후 필터링 — k를 못 채울 수 있다는 게 이 방식의 한계
    docs = store.similarity_search(query, k=k * 10)
    return [d for d in docs if d.metadata.get("source") == source][:k]
