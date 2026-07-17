"""Tool-Calling Agent: LLM이 도구(검색/계산/날짜)를 선택 호출하는 ReAct 루프.

실행 흐름:
    agent(LLM) ─ tool_calls 있음 → tools 실행 → agent (반복)
              └ tool_calls 없음 → 최종 답변 → END

기존 Corrective-RAG(graph.py)의 하이브리드 검색을 search_portfolio 도구로
래핑하여, 질문 성격에 따라 검색·계산·날짜 조회를 multi-step으로 조합한다.
"""
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from tools import TOOLS

# 소형 모델(3B)은 시스템 프롬프트가 길면 tool calling이 무력화되는 문제가 있어
# 최소한의 지시만 유지한다 (상세 사용 조건은 각 도구의 docstring이 전달).
SYSTEM_PROMPT = (
    "이윤선의 포트폴리오 어시스턴트입니다. "
    "필요하면 도구를 사용해 정확히 답하세요. "
    "경력/프로젝트 질문은 search_portfolio로 근거를 찾으세요."
)


def build_agent():
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0.1).bind_tools(TOOLS)

    def agent_node(state: MessagesState) -> dict:
        messages = state["messages"]
        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm.invoke(messages)

        # 결정적 폴백: 소형 모델이 도구도 답변도 내지 않는 경우(무응답),
        # 첫 턴이면 원 질문으로 search_portfolio를 강제 호출해 근거를 확보한다.
        already_used_tool = any(getattr(m, "tool_calls", None) for m in messages)
        if (not response.tool_calls and not response.content.strip()
                and not already_used_tool):
            question = next(
                m.content for m in messages if isinstance(m, HumanMessage)
            )
            response = AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_portfolio",
                    "args": {"query": question},
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                }],
            )
        return {"messages": [response]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # tool_calls 유무로 tools/END 분기
    g.add_edge("tools", "agent")
    return g.compile()


def run(question: str) -> dict:
    agent = build_agent()
    result = agent.invoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 12},  # 도구 호출 무한 루프 방지
    )
    messages = result["messages"]
    tool_calls = [
        tc["name"]
        for m in messages
        if getattr(m, "tool_calls", None)
        for tc in m.tool_calls
    ]
    return {"answer": messages[-1].content, "tool_calls": tool_calls}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else \
        "TTFB가 2292ms에서 334ms로 줄었는데 몇 퍼센트 개선인지 계산해줘"
    out = run(q)
    print("사용한 도구:", out["tool_calls"])
    print("답변:", out["answer"])
