"""프로젝트 전역 설정."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.jsonl"  # BM25 재구축용 청크 저장
# 인덱스와 chunks.jsonl이 같은 인제스트 산출물인지 대조하는 지문.
# 어긋난 채로 검색하면 서로 다른 청킹을 RRF로 섞게 되는데 증상이 없다.
INDEX_MANIFEST = BASE_DIR / "data" / "index_manifest.json"

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
# 각 검색기가 내놓는 후보 수이자 RRF 융합 결과의 길이.
#
# 15로 올리면 하이브리드 recall@5가 90% → 95%로 오른다(결정적 측정).
# 그런데 **정답률은 76% → 74%로 오르지 않았다**(51문항×2회, 편차 ±6%p).
# 이유: generate가 랭크 역순으로 배치하므로 새로 들어온 gold(대개 rank 4~5)가
# 프롬프트 맨 앞, 소형 모델의 주의가 가장 약한 위치에 놓인다. top-6에서
# rank 6이 살아나지 않았던 것과 같은 현상이다.
# recall이 올라도 그 자리를 모델이 못 쓰면 소용없다 → 6 유지.
# (측정: eval/sweep_top_k.py)
TOP_K = int(os.environ.get("TOP_K", "6"))

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

# 이웃 확장: 검색된 청크의 앞뒤 N개를 함께 프롬프트에 넣는다 (0이면 끔).
#
# 착안점: 근거를 더 주려고 **하위 랭크 청크를 추가**하면 generate가 랭크
# 역순 배치라 그게 프롬프트 맨 앞, 모델의 주의가 가장 약한 자리에 놓여
# 소용이 없다(top-6, TOP_K=15 둘 다 recall만 오르고 정답률은 제자리).
# 이웃 확장은 **상위 랭크 청크의 자리를 유지한 채** 주변 문맥만 붙인다.
#
# 실측 (51문항 × 2회를 두 번, GENERATE_TOP_N=3 + W=1 조건):
#     1차  현재 75% → 80%      2차  현재 74% → 75%
# **전체 정답률에서는 재현되지 않았다.** 편차 ±6%p 안이라 개선이라 할 수
# 없다. 그래서 기본값은 0(끔)이다.
#
# 다만 유형별로는 두 실행 모두 같은 방향이었다:
#     aggregation  6/12 → 10/12,  6/12 → 9/12   (세는 질문은 문맥이 이어져야)
#     temporal     8/8  →  6/8,   8/8  → 7/8
# 집계 유형에 한해 켜볼 만한 신호다. 반복을 늘려 다시 재는 것이 다음 순서.
NEIGHBOR_WINDOW = int(os.environ.get("NEIGHBOR_WINDOW", "0"))

# generate 프롬프트에 청크를 배치하는 순서.
#   reversed  1위가 질문 바로 앞 (기존 — 소형 모델의 끝부분 주의가 강하다는 직관)
#   ranked    1위가 맨 앞
#   sandwich  1위를 맨 앞과 맨 뒤 양쪽에 (가운데가 약하다는 가정)
#
# 세 실험이 전부 "근거는 들어갔는데 위치 때문에 못 쓴다"로 끝났는데, 정작
# 이 순서는 직관으로 정하고 잰 적이 없었다. 컨텍스트·지연 비용이 0이다.
#
# 실측 (51문항 × 2회):
#     역순 74% / 정순 74% / 샌드위치 78%
# 샌드위치가 4%p 높지만 편차 ±6%p 안이고, 유형별로 심하게 갈린다:
#     comparison   역순 4/4 → 샌드위치 2/4   (악화)
#     enumeration  역순 7/16 → 샌드위치 12/16 (개선)
#     refusal      20/20 → 19/20            (환각 1건)
# 단일 실행이고 비교 유형이 반토막 나므로 기본값은 바꾸지 않는다.
# 반복을 늘려 다시 재는 것이 다음 순서.
CONTEXT_ORDER = os.environ.get("CONTEXT_ORDER", "reversed")

# generate 프롬프트 변형: base | targeted
# targeted는 실측된 두 실패(열거 목록의 첫 항목만 답함, 같은 대상의
# 숫자가 여러 개일 때 범위 혼동)를 겨냥한 지시 두 줄을 더한다.
# 프롬프트를 늘려 소형 모델이 나빠지는 경우를 이미 겪었으므로
# 기본은 base로 두고 잰다.
GENERATE_PROMPT_VARIANT = os.environ.get("GENERATE_PROMPT_VARIANT", "base")

# 다이어그램 청크를 generate 컨텍스트에서 제외할지.
# 산문의 6.4배(4,395~5,079자)라 41문항 중 13문항에서 컨텍스트의 63~94%를
# 차지한다. 빼면 컨텍스트가 4,638자 → 2,758자(-41%)로 줄고 지연도 준다.
#
# 실측 (51문항 × 2회): 정답률 74% → 74%. **차이가 없다.**
# 즉 다이어그램은 정확도에 해도 득도 없이 컨텍스트만 먹고 있다. 지연·비용을
# 줄이려면 켤 만하지만, 정확도를 근거로 켤 이유는 없다. 다만 fact 유형이
# 28/38 → 24/38로 떨어져(단일 실행) 무해하다고 단정하기도 이르다.
EXCLUDE_DIAGRAMS = os.environ.get("EXCLUDE_DIAGRAMS", "0") == "1"

# 이보다 짧은 청크는 인덱싱하지 않는다. 마크다운 구분선('---')이나 제목 줄만
# 남은 조각이 인덱스 자리를 차지하는 것을 막는다 (검수로 발견 — inspect_data.py).
MIN_CHUNK_CHARS = 30

# 검색 결과가 부족할 때 질문을 재작성해 재검색하는 최대 횟수.
# 0으로 두면 grade·rewrite 노드가 통째로 빠져 순수 RAG가 된다 —
# corrective 루프가 실제로 값을 하는지 재는 A/B에 쓴다(eval/ab_rewrite.py).
MAX_REWRITES = int(os.environ.get("MAX_REWRITES", "1"))
