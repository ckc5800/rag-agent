"""대화형 CLI: 터미널에서 포트폴리오에 대해 질문."""
from graph import build_graph


def main():
    graph = build_graph()
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
