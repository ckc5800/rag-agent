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

# generate·grade가 실제로 보는 상위 N개.
#
# 오래 3이었다(소형 모델의 lost-in-the-middle 대응). 진단에서 검색 기인
# 실패 6건의 gold rank가 전부 4~6으로 나왔다 — recall@6는 95%인데 generate가
# 3개만 봐서 근거를 못 받고 있었다. 3/4/5/6을 51문항×2회로 재본 결과:
#
#     top-3  68%   top-4  70%   top-5  75%   top-6  75%
#
# 5와 6이 같지만 6에서 부작용이 나타난다 — 거부 20/20 → 19/20(환각 발생),
# temporal 7/8 → 6/8. generate가 랭크 역순으로 배치해 rank 6 근거가 긴
# 컨텍스트의 맨 앞(모델이 가장 못 보는 자리)에 놓이기 때문이다. 그래서 5.
# 편차가 ±6%p라 5 vs 6은 사실상 동률이고, 3보다 낫다는 것이 결론이다.
GENERATE_TOP_N = int(os.environ.get('GENERATE_TOP_N', '5'))

# 이보다 짧은 청크는 인덱싱하지 않는다. 마크다운 구분선('---')이나 제목 줄만
# 남은 조각이 인덱스 자리를 차지하는 것을 막는다 (검수로 발견 — inspect_data.py).
MIN_CHUNK_CHARS = 30

# 검색 결과가 부족할 때 질문을 재작성해 재검색하는 최대 횟수.
# 0으로 두면 grade·rewrite 노드가 통째로 빠져 순수 RAG가 된다 —
# corrective 루프가 실제로 값을 하는지 재는 A/B에 쓴다(eval/ab_rewrite.py).
MAX_REWRITES = int(os.environ.get("MAX_REWRITES", "1"))
