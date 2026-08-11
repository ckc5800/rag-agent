"""Multi-Agent Orchestration: Planner → RAG Workers → Synthesizer.

단일 RAG는 "A와 B를 비교해줘" 같은 멀티홉 질문에 약하다 — 한 번의 검색으로
서로 다른 두 정보를 모두 가져오기 어렵기 때문이다. 이를 해결하기 위해:

    질문 → planner    질문을 독립적인 sub-질문 목록으로 분해 (단순 질문이면 1개)
        → workers    sub-질문마다 기존 Corrective-RAG(graph.py)를 worker로 실행
        → synthesizer sub-답변들을 종합해 최종 답변 작성

소형 모델의 JSON 출력이 불안정하므로, planner 파싱 실패 시 원 질문을
단일 worker로 처리하는 결정적 폴백을 둔다.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

import config
from graph import build_graph, get_llm


class TeamState(TypedDict):
    question: str
    sub_questions: list[str]
    sub_answers: list[dict]   # {question, answer, sources}
    answer: str


# planner가 낼 수 있는 sub-질문 상한. 프롬프트 문구·파서 상한·
# 스레드 수가 모두 이 값을 따른다 — 예전엔 셋에 3이 따로 적혀 있어서,
# 프롬프트만 고치면 파서가 조용히 잘라내는 상태가 될 수 있었다.
MAX_SUB_QUESTIONS = 3

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    "질문에 답하는 데 필요한 사실(fact)들을 문서에서 찾기 위한 검색 질문으로 분해하세요.\n"
    "규칙:\n"
    "- 각 검색 질문은 문서에서 찾을 수 있는 사실 하나를 겨냥할 것\n"
    "- 원 질문을 변형하거나 의견을 묻는 질문을 만들지 말 것\n"
    f"- 하나의 사실로 답할 수 있으면 질문 1개만, 최대 {MAX_SUB_QUESTIONS}개\n"
    "- JSON 배열만 출력\n\n"
    "- 각 검색 질문은 완전한 의문문으로 쓸 것\n"
    "예시 1) 질문: A 프로젝트와 B 프로젝트 중 어느 것을 먼저 시작했어?\n"
    '출력: ["A 프로젝트는 언제 시작했어?", "B 프로젝트는 언제 시작했어?"]\n'
    "예시 2) 질문: 특허 등록번호 알려줘\n"
    '출력: ["특허 등록번호는 무엇인가?"]\n'
    "예시 3) 질문: 논문 수와 특허 수를 합치면 몇 개야?\n"
    '출력: ["논문은 몇 편인가?", "특허는 몇 건인가?"]\n\n'
    "질문: {question}"
)

SYNTH_PROMPT = ChatPromptTemplate.from_template(
    "아래는 원 질문을 분해해 각각 검색·답변한 결과입니다.\n"
    "이 결과들에 근거해서만 원 질문에 한국어로 답하세요. "
    "근거가 부족한 부분은 모른다고 하세요.\n\n"
    "원 질문: {question}\n\n{sub_results}\n\n최종 답변:"
)


def _parse_sub_questions(text: str, fallback: str) -> list[str]:
    """LLM 출력에서 JSON 배열 추출. 실패 시 원 질문 1개로 폴백."""
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group())
            items = [s.strip() for s in items if isinstance(s, str) and s.strip()]
            if items:
                return items[:MAX_SUB_QUESTIONS]
        except json.JSONDecodeError:
            pass
    return [fallback]


def build_team():
    rag = build_graph()  # 기존 Corrective-RAG를 worker로 재사용

    def planner(state: TeamState) -> dict:
        out = (PLANNER_PROMPT | get_llm()).invoke(
            {"question": state["question"]}).content
        return {"sub_questions": _parse_sub_questions(out, state["question"])}

    def _one_worker(sq: str) -> dict:
        # planner가 이미 검색용으로 정제한 질문이므로 worker의 자체 재작성은
        # 끈다 (rewrites를 상한으로 시작) — 이중 가공이 검색 믹스를 흐트러뜨림
        r = rag.invoke({"question": sq, "query": sq,
                        "rewrites": config.MAX_REWRITES})
        return {"question": sq, "answer": r["answer"], "sources": r["sources"]}

    def workers(state: TeamState) -> dict:
        """sub-질문을 동시에 처리한다. 순서는 planner가 낸 순서 그대로.

        멀티에이전트 구조인데 for 루프로 순차 실행하고 있었다 — sub-질문이
        3개면 지연도 3배다. README의 팀 다이어그램은 worker 2개가 병렬인
        것처럼 그려져 있어 구현과도 어긋났다.

        **답은 바뀌지 않는다.** worker끼리 독립이고(공유 상태 없음), 자체
        재작성이 꺼져 있어 temperature 0 경로만 타며, executor.map이 입력
        순서를 보존한다. 그래서 품질 A/B 없이 넣을 수 있는 변경이다 —
        남는 이득은 지연뿐이고, 그건 Ollama의 동시 처리 능력에 달렸으므로
        실제 머신에서 확인할 것(CPU 추론이면 이득이 작을 수 있다).

        planner가 sub-질문을 MAX_SUB_QUESTIONS개로 제한하므로 스레드도
        그만큼이 상한이다.
        """
        subs = state["sub_questions"]
        if len(subs) == 1:                      # 스레드풀 만들 이유가 없다
            return {"sub_answers": [_one_worker(subs[0])]}
        with ThreadPoolExecutor(max_workers=len(subs)) as pool:
            return {"sub_answers": list(pool.map(_one_worker, subs))}

    def synthesizer(state: TeamState) -> dict:
        subs = state["sub_answers"]
        # sub-질문이 1개면 종합 호출 없이 그대로 반환 (불필요한 LLM 호출 제거)
        if len(subs) == 1:
            return {"answer": subs[0]["answer"]}
        sub_results = "\n\n".join(
            f"[sub-질문 {i}] {s['question']}\n[답변] {s['answer']}"
            for i, s in enumerate(subs, 1))
        out = (SYNTH_PROMPT | get_llm()).invoke(
            {"question": state["question"], "sub_results": sub_results}).content
        return {"answer": out.strip()}

    g = StateGraph(TeamState)
    g.add_node("planner", planner)
    g.add_node("workers", workers)
    g.add_node("synthesizer", synthesizer)
    g.add_edge(START, "planner")
    g.add_edge("planner", "workers")
    g.add_edge("workers", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def ask_team(question: str) -> dict:
    team = build_team()
    r = team.invoke({"question": question})
    return {
        "answer": r["answer"],
        "sub_questions": r["sub_questions"],
        "sources": sorted({s for sa in r["sub_answers"] for s in sa["sources"]}),
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else \
        "TTS 프로젝트와 Kubernetes 프로젝트 중 어느 것을 먼저 시작했어?"
    out = ask_team(q)
    print("sub-질문:", out["sub_questions"])
    print("답변:", out["answer"])
    print("출처:", out["sources"])
