"""프로젝트 전역 설정."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
DB_DIR = str(BASE_DIR / "data" / "faiss_index")

# Ollama 모델 (GPU 환경이면 qwen2.5:7b 권장, CPU 환경은 3b)
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"

# 청킹 / 검색 파라미터
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6

# 검색 결과가 부족할 때 질문을 재작성해 재검색하는 최대 횟수
MAX_REWRITES = 1
