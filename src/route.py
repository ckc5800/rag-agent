"""질문 유형별 컨텍스트 전략 — 하나의 전역 설정으로는 유형마다 다른
요구를 동시에 만족시키지 못한다는 게 config.py의 실측 주석들(NEIGHBOR_WINDOW,
CONTEXT_ORDER)에 이미 나와 있다:

    aggregation — 이웃 확장(NEIGHBOR_WINDOW=1)이 두 번의 실행 모두 개선
                  (6/12→10/12, 6/12→9/12). 세는 질문은 문맥이 이어져야 한다.
    enumeration — 샌드위치 배치가 두 번의 실행 모두 개선(7/16→12/16 등).
    comparison  — 샌드위치는 오히려 악화(4/4→2/4)라 기본(역순)을 유지해야 한다.
    temporal    — 이웃 확장이 두 번 다 악화(8/8→6/8, 8/8→7/8), 손대지 않는다.

전역 기본값(reversed, NEIGHBOR_WINDOW=0)은 이 세 유형 각각에는 최선이
아니지만, 하나를 켜면 다른 유형이 나빠져 기본값을 못 바꾸고 있었다.
질문마다 다르게 적용하면 이 상충을 풀 수 있다는 게 이 모듈의 가설이다.

**분류는 LLM이 아니라 규칙 기반이다.** pdm-agent의 원칙과 같다 — "탐지는
결정적으로, 판단이 필요한 것만 모델에게." 정규식 오분류의 대가가 비대칭이라
정밀도를 리콜보다 우선한다: fact·temporal·refusal 질문을 aggregation·
enumeration으로 잘못 분류하면 원래 좋던 설정(reversed)을 나쁜 설정으로
바꿔버리는데, 반대로 진짜 aggregation·enumeration 질문을 놓쳐 fact
취급하면 그냥 기본값을 쓰는 것뿐이라 손해가 없다. 그래서 신호가 확실할
때만 분류하고 애매하면 fact(기본값)로 둔다.

51문항 평가셋으로 정밀도·재현율을 확인했다 — fact 19/19·temporal 4/4·
refusal 10/10 오탐 없음, aggregation 6/6·comparison 2/2 전량 포착,
enumeration은 8개 중 4개만 포착(나머지는 안전하게 fact로 폴백).

**63문항 확장 후 재검증** — comparison이 2→8문항이 되자 2건을 놓쳤다.
둘 다 "중 나중에" 형태였다("…중 나중에 시작한 쪽", "…중 나중에 취득한").
기존 패턴은 "중 더/먼저"만 봤는데, 비교의 방향이 뒤쪽일 때(나중·늦게)를
빠뜨린 것이다. 오탐은 여전히 0이므로 리콜만 넓힌다. 재검증 결과:
fact 19/19·temporal 4/4·refusal 10/10 오탐 없음, comparison 8/8 포착.
"""
import re

_COMPARISON = re.compile(r"중\s*(더|먼저|나중|늦게|어느|어떤|무엇)")
_AGGREGATION = re.compile(r"몇\s*(편|건|배)|(총|합치면|모두)\s*몇")
_ENUMERATION = re.compile(
    r"들을?\s*(알려|무엇|모두)|모두\s*(알려|나열)|몇\s*가지|들을?\s*어떤\s*순서")

# 순서가 중요하다 — comparison이 가장 구체적인 신호("A와 B 중")라 먼저 본다.
# 세 패턴이 겹치는 실측 사례는 없었지만, 겹치면 comparison을 우선한다
# (comparison을 다른 유형으로 잘못 분류하면 샌드위치가 적용돼 악화되므로
# 더 보수적인 쪽을 우선).
_CLASSIFIERS = (
    ("comparison", _COMPARISON),
    ("aggregation", _AGGREGATION),
    ("enumeration", _ENUMERATION),
)


def classify_question_type(question: str) -> str:
    """규칙 기반 분류. 신호가 없으면 'fact'(= 라우팅 없음, 기본 설정)."""
    for label, pattern in _CLASSIFIERS:
        if pattern.search(question):
            return label
    return "fact"


# 유형 → config 오버라이드. 여기 없는 유형(fact/temporal/trap/refusal)은
# 오버라이드 없이 전역 기본값을 그대로 쓴다.
#
# aggregation에 GENERATE_TOP_N=3을 같이 묶는 이유 — 이웃 확장의 개선
# 신호(6/12→10/12 두 번 재현)는 **GENERATE_TOP_N=3 + W=1** 조건에서
# 측정된 것이다(config.py NEIGHBOR_WINDOW 주석). 그 뒤 base가 top-5로
# 올라갔는데, 처음엔 W=1만 얹었더니 top-5 × W=1이라는 **미측정 조합**이
# 됐고, 컨텍스트가 3,522자 → 7,742자(2.2배)로 부풀며 "제1저자 논문 몇
# 편?"(base 3/3)을 0/3으로 깨뜨렸다 — 실측으로 확인. "컨텍스트를 키우면
# 환각 저항이 먼저 무너진다"(README §9)의 세 번째 재현이다. 그래서
# 오버라이드를 개선이 실제로 측정된 좌표(top-3 + W=1 ≈ 4.3k자)로 되돌린다.
#
# enumeration의 sandwich는 base가 이미 top-5일 때 측정된 값이라(배치 순서
# 실험, 51문항×2회) 그대로 둔다.
ROUTES: dict[str, dict[str, object]] = {
    "aggregation": {"NEIGHBOR_WINDOW": 1, "GENERATE_TOP_N": 3},
    "enumeration": {"CONTEXT_ORDER": "sandwich"},
}


def should_use_team(question: str) -> bool:
    """멀티에이전트 팀(team.py)으로 보낼 질문인가.

    측정 근거(eval/ab_team_routing.py, 22문항×2회, qwen2.5:14b):

        single    35/44 (80%)   기준
        routed    40/44 (91%)   +11%p   comparison 만 팀
        all_team  30/44 (68%)   -11%p   전부 팀

    **전부 팀으로 보내면 기준선보다 나쁘다** — aggregation 10/12→6/12,
    enumeration 16/16→10/16. 분해는 서로 다른 청크의 사실을 엮을 때 이기고,
    한 청크에 요약이 이미 있을 때는 그 근거를 잃는다. 그래서 팀이 좋은 게
    아니라 **멀티홉에만** 좋다 — 분류기가 이득의 원천이다.

    aggregation 신호를 배제하는 이유 — 두 신호를 동시에 가진 문항이 있다:

        "제1저자 논문 편수와 등록 특허 건수 중 더 많은 쪽은 몇 편인가요?"
          단일 2/2 — 요약 청크("논문 7편(제1저자), 특허 2건")로 바로 답
          팀   0/2 — 하위 질문으로 쪼개 목록을 직접 세다가 5편으로 오답

    "중 더"(comparison)와 "몇 편"(aggregation)이 겹치는데 성격은 집계다.
    classify_question_type 은 comparison 을 우선하지만(컨텍스트 전략에서는
    그게 맞다), 팀 라우팅에서는 반대로 aggregation 이 있으면 보내지 않는다.

    **한계**: 이 가드의 근거는 위 1문항이다. 겹치는 문항이 더 모이기 전에는
    가설로 취급할 것 — 지금은 그 1건에서 routed 40/44 → 42/44 가 된다.
    """
    if _AGGREGATION.search(question):
        return False
    return classify_question_type(question) == "comparison"
