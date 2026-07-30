"""대화형 CLI: 터미널에서 포트폴리오에 대해 질문."""
from graph import build_graph, warmup
from preflight import check_all


def main():
    check_all()          # 인덱스·Ollama 확인 (실패 시 안내 후 종료)
    graph = build_graph()
    warmup()             # 첫 질문 지연 제거
    print("포트폴리오 RAG Agent (종료: q)")
    while True:
        q = input("\n질문> ").strip()
        if not q or q.lower() == "q":
            break
        result = graph.invoke({"question": q, "query": q, "rewrites": 0})
        print(f"\n{result['answer']}")
        print(f"\n[출처: {', '.join(result['sources'])}"
              f" | 질문 재작성 {result['rewrites']}회]")


if __name__ == "__main__":
    main()
