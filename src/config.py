"""프로젝트 전역 설정."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
DB_DIR = str(BASE_DIR / "data" / "faiss_index")
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"  # BM25 재구축용 청크 저장

# Ollama 모델 (GPU 환경이면 qwen2.5:7b 권장, CPU 환경은 3b)
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"

# 청킹 / 검색 파라미터
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6

# 이보다 짧은 청크는 인덱싱하지 않는다. 마크다운 구분선('---')이나 제목 줄만
# 남은 조각이 인덱스 자리를 차지하는 것을 막는다 (검수로 발견 — inspect_data.py).
MIN_CHUNK_CHARS = 30

# 검색 결과가 부족할 때 질문을 재작성해 재검색하는 최대 횟수
MAX_REWRITES = 1
