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
#
# **위 이유는 재측정에서 기각됐다.** 값(6)은 유지하되 근거가 다르다.
# GPU(RTX 5080)에서 7B로 재보니 TOP_K=15 가 85%로 기본(85.3%)과 동일했다 —
# 3B 때처럼 떨어지지도, 오르지도 않는다. "모델이 그 자리를 못 쓴다"가
# 맞다면 큰 모델에서는 올라야 했다.
#
# 진짜 이유는 **새로 들어올 gold 가 없다**는 것이다. 경로 기반 커버리지로
# 재보면 이 코퍼스는 satisfied@6 이 97~100% 라 후보를 15개로 넓혀도 추가로
# 건질 근거가 존재하지 않는다(eval/eval_coverage.py). 즉 TOP_K 는 이 코퍼스
# 에서 **돌릴 대상이 없는 손잡이**다. 코퍼스가 커지면 다시 재야 한다.
TOP_K = int(os.environ.get("TOP_K", "6"))

# RRF(Reciprocal Rank Fusion) 완충 상수 — score = Σ 1/(K+rank). graph.py의
# hybrid_search와 kg.py의 fused_search가 공유한다. 도입 당시 "표준값"으로
# 60을 그냥 썼고 스윕한 적이 없었다.
#
# eval/sweep_rrf_k.py(원 코퍼스 14문항)로 재보니 K=1이 recall@1 57%로
# K=60(50%)보다 높아 보였지만, 문항이 14개뿐이라 이건 질문 1개가 뒤집힌
# 것뿐이었다(±7%p = 1/14). KLUE-RE(120문항)로 다시 재니 K=5~200 구간에서
# recall@1 73%·MRR 0.812로 완전히 동일하고, K=1만 오히려 더 낮다(70%,
# MRR 0.792) — 작은 표본의 신호와 방향이 반대였다. 60은 "실측 없이 표준값
# 이라 썼다"가 아니라 실측으로도 평평한 최적 구간 안에 있는 값으로 확인됨.
RRF_K = int(os.environ.get("RRF_K", "60"))

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
#
# **위 스윕은 전부 qwen2.5:3b(CPU) 관측이다.** 모델을 키우면 전제가 바뀐다 —
# 51문항 3회 기준 3B 80.0% / 7B 85.3% / 14B 94.0% 로, 모델 크기가 이
# 프로젝트에서 가장 큰 레버였다(코드 변경 없이 환경변수 하나). "긴 컨텍스트를
# 소형 모델이 못 본다"를 근거로 정한 값이므로 14B 를 기본으로 쓸 거라면
# 3/4/5/6 을 다시 스윕해야 한다. 값은 재측정 전까지 5로 둔다.
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
#
# **재측정 완료(eval/ab_neighbor_window.py, 55문항 × 2회, GENERATE_TOP_N=5·
# TYPE_ROUTING 끈 조건, 2026-08)** — "다음 순서"로 남겨 뒀던 TODO를 닫는다.
# 전체 정답률은 다시 한 번 동률이었다(W=0 78% vs W=1 77%). 그런데 유형별
# 분해는 예전보다 훨씬 선명하게 갈렸다:
#     enumeration  12/18 → 18/18   (만점 — 가장 큰 신호)
#     comparison    1/4  →  3/4
#     aggregation   5/12 →  6/12
#     fact         40/44 → 33/44   (가장 큰 손해)
#     refusal      20/20 → 18/20   (환각 저항이 무너진다 — 반복 관찰된 패턴)
#     temporal      6/8  →  5/8
# 지연도 2.9s → 4.6s(1.6배)로 는다. **전역 기본값은 계속 0이 맞다** —
# fact가 전체 문항의 40%라 전역으로 켜면 손해가 이득을 덮는다. 대신 이
# 측정이 route.py의 유형별 오버라이드를 정당화한다(TYPE_ROUTING 참고).
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
#
# **실측 완료(eval/ab_prompt_variant.py, 55문항, 2026-08)** — 반복 횟수가
# 결론을 뒤집은 사례라 과정을 남긴다:
#     2회 반복: base 75%(83/110) vs targeted 80%(88/110)  → +5%p, 유망해 보임
#     4회 반복: base 78%(171/220) vs targeted 78%(171/220) → **완전 동률**
# 2회에서 보였던 +5%p는 노이즈였다(문서화된 편차 ±6%p 안). 채택 전에
# 반복을 늘려보길 잘한 경우다.
#
# 4회 반복의 "동률"이 상쇄로 보이긴 했다:
#     aggregation  10/24 → 13/24  (개선)
#     fact         76/88 → 71/88  (악화 — 긴 프롬프트가 단순 조회를 방해)
# 특히 "한국자동차공학회에 게재한 논문은 몇 편?"이 0/4 → 4/4로 구제된 게
# targeted의 두 번째 규칙(범위 확인)이 정확히 겨냥한 실패라, NEIGHBOR_WINDOW
# 처럼 유형별 라우팅으로 aggregation에만 주면 되겠다는 가설을 세웠다.
#
# **그 가설은 실측에서 기각됐다**(eval/ab_route_variants.py,
# enum_w1_agg_targeted, 15문항 × 5회). aggregation은 14/30 → 17/30(+10%p)
# 인데 대조군(enumeration, 두 조합의 설정이 동일)이 40/45 → 38/45(-4.4%p)로
# 함께 흔들려 효과가 노이즈의 2배 남짓에 그쳤다. 결정적인 건 **구제 근거로
# 삼았던 바로 그 문항이 여기서는 0/5 대 0/5**라는 점이다 — 거의 같은
# 조건에서 0/4→4/4와 0/5→0/5가 갈리므로 그 "구제"는 재현되지 않는
# 노이즈였다. 근거가 무너졌으므로 라우팅에 넣지 않는다. 전역 기본도 base 유지.
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

# 질문 유형별로 NEIGHBOR_WINDOW·CONTEXT_ORDER를 다르게 적용할지(src/route.py).
# NEIGHBOR_WINDOW·CONTEXT_ORDER 실험 둘 다 "어떤 유형엔 좋고 어떤 유형엔
# 나쁘다"로 끝나 전역 기본값을 못 바꿨다 — 유형별로 다르게 주면 상충을 풀 수
# 있는지가 가설이었다.
#
# eval/ab_type_routing.py(aggregation·enumeration 15문항 × 3회, 2026-08
# 재측정 — Kiwi 토크나이저·SEED_MATCH_RATIO=0.7 반영 후)로 채택 여부를
# 결정했다: OFF 69% → ON 78%, 구제 2 / 악화 0 / 변화없음 13. 악화가
# 하나도 없어 트레이드오프가 아니라 순개선이다. 기본을 켠다.
TYPE_ROUTING = os.environ.get("TYPE_ROUTING", "1") == "1"

# 멀티홉 질문만 team.py(Planner→Workers→Synthesizer)로 보낼지(route.should_use_team).
#
# 측정은 있었는데 **배선이 없었다** — ab_team_routing.py가 routed 42/44(95%)
# vs single 34/44(77%)로 이 저장소 최대 개선폭(+18%p)을 냈는데, 정작
# should_use_team()을 부르는 건 그 평가 스크립트뿐이었고 api.py·cli.py는
# 단일 그래프로만 갔다. 값이 증명된 기능이 제품에 없던 셈이라 연결한다.
#
# **기본 조건(3b)에서 재측정했고, 켜지 않기로 했다.** 그 +18%p는
# qwen2.5:14b 기준이었는데 3b에서는 재현되지 않는다.
#
# 1차(75문항×2회)는 routed 121/150 vs single 108/150 으로 +9%p처럼 보였다.
# 그런데 should_use_team()은 comparison만 팀으로 보내므로 **나머지 유형은
# single과 완전히 같은 코드 경로**다 — 그 대조군이 100 → 109로 +9판정이나
# 흔들렸다. 즉 헤드라인 +13판정 중 9가 노이즈, 실제 팀 효과는 +4였다.
#
# 2차로 comparison만 격리해 5회 반복(10문항×5회=50판정):
#     single    20/50 (40%)  2.9s
#     routed    22/50 (44%)  3.9s   ← +2판정
#     all_team  20/50 (40%)  3.8s   ← routed와 정의상 동일 조건
# comparison만 돌리면 routed와 all_team은 같은 설정인데 22 vs 20으로 갈린다
# — **효과(+2)와 노이즈(2)가 같은 크기**다. 게다가 지연은 2.9s → 3.9s(+34%).
#
# 문항별로 보면 분산이 크다 — 팀이 고치는 것도 있고("등록특허 vs 2023년 논문"
# 0/5 → 5/5) 부수는 것도 있다("석사와 학사 중 나중에 취득한 학위" 5/5 → 0/5).
# 3b는 planner가 하위 질문을 잘 못 쪼개고 synthesizer도 약해서, 분해가
# 만드는 오류가 분해로 얻는 이득을 상쇄한다. **멀티에이전트의 값어치는
# 모델 크기에 의존한다**는 게 이 실험의 결론이다 — 14b를 기본으로 쓸 거면
# 다시 켜고 재측정할 것.
TEAM_ROUTING = os.environ.get("TEAM_ROUTING", "0") == "1"

# RRF 순위를 LLM으로 재정렬할지(graph.rerank). diagnose.py가 지목한 병목
# (검색은 top-3에 근거를 올렸는데 생성이 놓치는 10건)을 겨냥한 시도다.
# LLM 호출이 질의당 1회 늘어(retrieve 직후) 지연이 커진다.
#
# **서로 다른 두 세션이 독립적으로 재서 같은 결론에 도달했다** — 모델도
# 표본도 다른데 방향이 같아 신뢰도가 높다:
#
#     7B · 51문항×2회   85%   vs 기본 85.3%   지연 1.7s → 2.9~3.7s
#     3B · 55문항×2회   78%(86/110) vs 78%(86/110)  지연 2.6s → 5.1s
#
# 두 조건 모두 **정확도는 동률인데 지연만 약 2배**다(3B 쪽은 뒤집힌 8문항이
# 상승 4/하락 4로 방향성도 없다). 끄는 것이 맞다.
#
# 원인까지 짚으면 "리랭커가 나빠서"가 아니라 **재정렬할 대상이 없어서**다 —
# 이 코퍼스는 satisfied@6 이 97~100%라 상위 6개 안에 답이 이미 다 들어와
# 있다(eval/eval_coverage.py). 후보가 수백 개가 되는 규모에서는 결론이
# 달라질 수 있다.
RERANK = os.environ.get("RERANK", "0") == "1"

# generate 직후 답변이 근거 문서에 실제로 기반하는지 사후 확인할지
# (graph.verify). GENERATE_PROMPT가 "지어내지 마세요"라고 지시하지만 그
# 지시를 따랐는지 확인하는 단계가 없었다. 지금은 State에 기록만 하고
# 라우팅은 안 바꾼다(fail-open) — LLM 호출이 하나 더 늘어 지연이 커지고,
# 아직 A/B 전이라 기본은 꺼둔다.
VERIFY_GROUNDING = os.environ.get("VERIFY_GROUNDING", "0") == "1"

# ── Parent-Child 청킹 (실험) ──────────────────────────────
#
# 기본 청킹(위 CHUNK_SIZE)은 검색과 생성이 같은 크기의 청크를 공유한다.
# 정확한 검색에 맞는 크기(작음)와 완결된 답변에 맞는 크기(큼)는 원래 다른데,
# 하나의 크기로 타협한 것이다. Parent-Child는 이 타협을 없앤다 — child(작음)
# 로 검색하고 그 child가 속한 parent(큼)를 생성에 준다. 검색은 정밀하게,
# 답변은 맥락을 잃지 않게.
# PARENT_SIZE는 원래 2000이었다가 800으로 줄였다. 2000자에서 "이윤선의
# 제1저자 논문은 몇 편?" 질문이 7B로도, 프롬프트를 고쳐도 안 풀렸다 —
# 정답 문장("학술 성과 총계: 논문 7편")이 parent 안에서는 여전히 앞쪽에 있고,
# 뒤로 TTFB·채널 수치 등 무관한 숫자가 1300자 넘게 딸려오며 희석됐다.
# base(단일 청크, 800자 상한)의 같은 문장은 그 상한 덕에 무관한 숫자
# 섹션에 닿기 전에 청크가 끊겨 우연히 희석을 피했다. 그래서 parent도
# 800으로 맞춰 같은 보호를 받게 했다 — child(300)보다는 크게 유지해
# parent-child의 취지(검색은 정밀하게, 생성은 맥락 있게)는 살린다.
PARENT_SIZE = 800
PARENT_OVERLAP = 100
CHILD_SIZE = 300
CHILD_OVERLAP = 30
PARENT_STORE_PATH = str(BASE_DIR / "data" / "parents.jsonl")
PARENT_DB_DIR = str(BASE_DIR / "data" / "parent_faiss_index")
PARENT_CHUNKS_PATH = BASE_DIR / "data" / "parent_child_chunks.jsonl"

# ── 시맨틱 청킹 (실험) ─────────────────────────────────────
#
# 고정 글자 수 대신 문장 임베딩의 유사도가 급격히 떨어지는 지점(주제 전환)
# 에서 자른다. 800자 상한은 여전히 안전판으로 강제한다(무한정 안 잘리는
# 구간 방지) — src/semantic_chunk.py 참고.
SEMANTIC_PERCENTILE = 95.0       # 이 percentile을 넘는 인접-문장 거리만 breakpoint
SEMANTIC_MAX_CHARS = 800         # 안전판 상한 (기본 CHUNK_SIZE와 동일하게 맞춤)
SEMANTIC_MIN_CHARS = 30
SEMANTIC_DB_DIR = str(BASE_DIR / "data" / "semantic_faiss_index")
SEMANTIC_CHUNKS_PATH = BASE_DIR / "data" / "semantic_chunks.jsonl"

# ── 임베딩 모델 비교 (실측 완료, 미채택) ──────────────────────
#
# README 한계에 "임베딩은 bge-m3 하나만 써봤다. 비교 대상이 없다"고
# 정직하게 적어 뒀던 것을 겨냥했다. nomic-embed-text는 다국어 지원을
# 내세우는 가벼운 모델(~137M)이라 비교 대상으로 골랐다 — bge-m3(다국어,
# 한국어 강세)와 정반대 크기대의 선택. 실측(compare_embeddings.py, 같은
# 58청크 코퍼스): recall@1 70%→20%, MRR 0.85→0.533으로 bge-m3가 크게
# 앞선다. 한국어 코퍼스에서는 경량 다국어 모델이 밀린다는 뚜렷한 신호라
# 기본은 bge-m3 유지.
ALT_EMBED_MODEL = "nomic-embed-text"
ALT_EMBED_DB_DIR = str(BASE_DIR / "data" / "alt_embed_faiss_index")
ALT_EMBED_CHUNKS_PATH = BASE_DIR / "data" / "alt_embed_chunks.jsonl"
