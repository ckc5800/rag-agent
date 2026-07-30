"""프로젝트 전역 설정."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"  # BM25 재구축용 청크 저장

# 벡터 저장소 — faiss | qdrant (환경변수 VECTOR_STORE로 교체)
#
# FAISS는 파일 2개(index.faiss/index.pkl)로 끝나 가장 가볍지만 메타데이터
# 필터가 없다. Qdrant는 payload 필터를 지원해 "resume.md 안에서만 검색"
# 같은 질의가 되고, 서버 없이 로컬 경로로도 돈다(임베디드 모드).
# 같은 임베딩·같은 exact 검색이라 recall은 동일해야 한다 — 차이는 기능이다.
VECTOR_STORE = os.environ.get("VECTOR_STORE", "faiss")

DB_DIR = str(BASE_DIR / "data" / "faiss_index")
QDRANT_PATH = str(BASE_DIR / "data" / "qdrant_index")
QDRANT_COLLECTION = "portfolio"

# Ollama 모델 (GPU 환경이면 qwen2.5:7b 권장, CPU 환경은 3b)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")

# 청킹 / 검색 파라미터
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6

# 이보다 짧은 청크는 인덱싱하지 않는다. 마크다운 구분선('---')이나 제목 줄만
# 남은 조각이 인덱스 자리를 차지하는 것을 막는다 (검수로 발견 — inspect_data.py).
MIN_CHUNK_CHARS = 30

# 검색 결과가 부족할 때 질문을 재작성해 재검색하는 최대 횟수.
# 0으로 두면 grade·rewrite 노드가 통째로 빠져 순수 RAG가 된다 —
# corrective 루프가 실제로 값을 하는지 재는 A/B에 쓴다(eval/ab_rewrite.py).
MAX_REWRITES = int(os.environ.get("MAX_REWRITES", "1"))
