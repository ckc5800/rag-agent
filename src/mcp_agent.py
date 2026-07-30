"""MCP 클라이언트 에이전트 — portfolio-mcp 서버의 도구를 그대로 쓰는 ReAct 루프.

agent.py의 도구(검색/계산/날짜)는 이 저장소 안에 하드코딩되어 있다.
이 파일은 같은 에이전트 구조에서 도구만 MCP(Model Context Protocol)로
바꾼 버전이다 — 별도 저장소의 portfolio-mcp 서버를 stdio로 띄우고,
서버가 광고하는 도구 4개(profile/projects/publications/BM25 검색)를
langchain-mcp-adapters로 LangChain 도구로 변환해 에이전트에 물린다.

도구 정의가 에이전트 코드 밖(서버)에 있으므로, 서버가 도구를 추가하면
에이전트는 코드 수정 없이 그대로 쓴다. MCP 서버(제공자)와 MCP 클라이언트
(소비자)를 양쪽 다 구현해 보는 것이 목적.

실행 조건: 형제 디렉토리에 portfolio-mcp가 클론되어 있어야 한다
(경로는 PORTFOLIO_MCP_SERVER 환경변수로 변경 가능).
"""
import asyncio
import os
import sys
import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config

MCP_SERVER = os.environ.get(
    "PORTFOLIO_MCP_SERVER",
    str(Path(__file__).resolve().parents[2] / "portfolio-mcp" / "server.py"),
)

# agent.py와 동일한 원칙: 3B 모델은 시스템 프롬프트가 길면 tool calling이
# 무력화되므로 최소 지시만 유지한다.
SYSTEM_PROMPT = (
    "이윤선의 포트폴리오 어시스턴트입니다. "
    "필요하면 도구를 사용해 정확히 답하세요. "
    "경력/프로젝트 질문은 portfolio_search로 근거를 찾으세요."
)


def build_agent(tools):
    llm = ChatOllama(model=config.LLM_MODEL, temperature=0.1).bind_tools(tools)

    async def agent_node(state: MessagesState) -> dict:
        messages = state["messages"]
        if not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = await llm.ainvoke(messages)

        # 결정적 폴백(agent.py와 동일): 무응답이면 검색을 강제 호출
        already_used_tool = any(getattr(m, "tool_calls", None) for m in messages)
        if (not response.tool_calls and not response.content.strip()
                and not already_used_tool):
            question = next(
                m.content for m in messages if isinstance(m, HumanMessage)
            )
            response = AIMessage(
                content="",
                tool_calls=[{
                    "name": "portfolio_search",
                    "args": {"query": question},
                    "id": str(uuid.uuid4()),
                    "type": "tool_call",
                }],
            )
        return {"messages": [response]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


async def run(question: str) -> dict:
    if not Path(MCP_SERVER).exists():
        raise SystemExit(
            f"MCP 서버를 찾을 수 없습니다: {MCP_SERVER}\n"
            "형제 디렉토리에 portfolio-mcp를 클론하거나 "
            "PORTFOLIO_MCP_SERVER 환경변수로 경로를 지정하세요.")

    client = MultiServerMCPClient({
        "portfolio": {
            "command": sys.executable,
            "args": ["-X", "utf8", MCP_SERVER],
            "transport": "stdio",
        }
    })
    # get_tools()는 세션을 들고 있지 않는다 — 도구 호출마다 stdio 세션을
    # 새로 열고 닫는다(langchain-mcp-adapters 규약). 즉 자식 프로세스는
    # 남지 않으므로 여기서 세션을 따로 관리하지 않는다.
    tools = await client.get_tools()
    agent = build_agent(tools)
    result = await agent.ainvoke(
        {"messages": [("user", question)]},
        config={"recursion_limit": 12},
    )
    messages = result["messages"]
    tool_calls = [
        tc["name"]
        for m in messages
        if getattr(m, "tool_calls", None)
        for tc in m.tool_calls
    ]
    return {"answer": messages[-1].content, "tool_calls": tool_calls,
            "mcp_tools": [t.name for t in tools]}


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "우수 논문상을 받은 논문 제목이 뭐야?"
    out = asyncio.run(run(q))
    print("MCP 서버가 광고한 도구:", out["mcp_tools"])
    print("사용한 도구:", out["tool_calls"])
    print("답변:", out["answer"])
