"""Agent가 선택 호출하는 도구 정의."""
import ast
import operator
from datetime import date

from langchain_core.tools import tool

import graph as rag

# ── 안전한 수식 평가 (eval 금지, AST 화이트리스트 방식) ──
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


# 거듭제곱 상한. AST 화이트리스트는 eval을 막아 주지만 자원 고갈은 못 막는다 —
# '9**9**9**9' 한 줄이면 파이썬이 수십억 자리 정수를 만들려다 프로세스가 멈춘다.
# 입력이 3B 모델의 출력이라 악의보다는 오조립으로 나올 수 있는 형태다.
MAX_EXPONENT = 64


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"지수가 너무 큽니다 (최대 {MAX_EXPONENT})")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("허용되지 않는 표현식입니다 (사칙연산만 가능)")


@tool
def search_portfolio(query: str) -> str:
    """이윤선의 경력, 프로젝트, 논문, 특허, 기술 스택에 대한 정보를 포트폴리오 문서에서 검색한다.

    포트폴리오/경력 관련 질문에는 반드시 이 도구로 근거를 찾은 뒤 답해야 한다.
    """
    # RAG 경로와 **같은 것을 본다**. 예전엔 상위 4개를 600자로 잘라
    # 넘겼는데 셋 다 문제였다:
    #   · 개수 4 — 측정으로 정한 GENERATE_TOP_N(5)과 어긋난다.
    #   · 600자 — 산문 청크가 800자 상한이라 25%가 잘려 나간다.
    #   · 통짜 청크 — 표(1,773자)·다이어그램(최대 5,079자)이 뭉개져,
    #     "논문 몇 편"처럼 표 전체가 있어야 답이 되는 질문이 원리적으로
    #     불가능해진다. 본 파이프라인에서 표를 통짜로 만든 작업이 이
    #     레이어에는 닿지 않고 있었다.
    # context_docs는 라우팅 전략·이웃 확장까지 본 파이프라인과 같은 규칙을
    # 쓰므로, 재사용하면 두 경로가 갈라지지 않는다.
    docs = rag.context_docs(rag.hybrid_search(query))
    return "\n---\n".join(
        f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs)


@tool
def calculate(expression: str) -> str:
    """사칙연산 수식을 계산한다. 예: '(2292-334)/2292*100'"""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return str(round(result, 4))
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def get_current_date() -> str:
    """오늘 날짜를 YYYY-MM-DD 형식으로 반환한다. '올해', '현재', '몇 년째' 등 시점 계산이 필요할 때 사용한다."""
    return date.today().isoformat()


TOOLS = [search_portfolio, calculate, get_current_date]
