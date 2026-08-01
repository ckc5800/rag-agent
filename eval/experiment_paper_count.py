""""논문 몇 편?" 실패를 두 축으로 분리 실험 — 모델 크기 vs 프롬프트.

repeat_parent_child.py로 확인한 것: qwen2.5:3b는 "이 프로젝트에서 논문 2편
게재(제1저자, 전체 경력 논문 7편 중 일부)"처럼 명시적으로 조건절을 달아도
표면적으로 가까운 숫자(2)를 답으로 낸다 — 부분/전체 관계를 못 푼다.

이게 (a) 모델 용량 문제인지 (b) 프롬프트가 그 판단 규칙을 안 알려줘서인지
분리해서 재본다:

    A. qwen2.5:7b + 기존 프롬프트   → 모델을 키우면 풀리는가
    B. qwen2.5:3b + 개선 프롬프트   → 규칙을 명시하면 풀리는가

둘 다 5회 반복(7B는 느려서 3회)해서 n=1의 함정(이 프로젝트에서 이미 한 번
걸렸다)에 다시 안 걸리게 한다.

    python eval/experiment_paper_count.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

Q = "이윤선의 제1저자 논문은 몇 편인가요?"
OK_PATTERNS = ["7편", "7 편"]

# 기존 프롬프트에 "합계 vs 부분" 판단 규칙 한 줄을 추가한 버전.
IMPROVED_RULE = (
    "참고: 숫자가 여러 번 언급되면 '총', '전체', '총계'가 붙은 숫자가 합계이고, "
    "'이 프로젝트에서', '이 연구에서'처럼 범위가 좁혀진 숫자는 그 합계의 일부입니다. "
    "'몇 개/편/건'처럼 전체를 묻는 질문에는 반드시 합계(총/전체) 쪽 숫자로 답하세요.\n"
)


def check(answer: str) -> bool:
    return any(p in answer for p in OK_PATTERNS)


def run_condition(label: str, model: str, use_improved_prompt: bool, repeat: int):
    import config
    config.LLM_MODEL = model

    for m in ("graph", "graph_parent_child", "vectorstore"):
        sys.modules.pop(m, None)
    import graph
    import graph_parent_child as gpc

    if use_improved_prompt:
        from langchain_core.prompts import ChatPromptTemplate
        # ChatPromptTemplate은 .template이 아니라 .messages[0].prompt.template에
        # 원문 문자열이 있다 (HumanMessagePromptTemplate으로 감싸져 있어서).
        old_template = graph.GENERATE_PROMPT.messages[0].prompt.template
        new_template = old_template.replace(
            "참고: 항목 옆", IMPROVED_RULE + "참고: 항목 옆")
        assert new_template != old_template, "치환 대상 문자열을 못 찾음"
        new_prompt = ChatPromptTemplate.from_template(new_template)
        # graph_parent_child.py는 `from graph import GENERATE_PROMPT`로 값을
        # 복사해 가져갔다 — graph.GENERATE_PROMPT만 바꾸면 gpc.generate()는
        # 여전히 옛 프롬프트를 쓴다. 두 모듈 다 갱신해야 한다.
        graph.GENERATE_PROMPT = new_prompt
        gpc.GENERATE_PROMPT = new_prompt

    print(f"\n{'='*70}\n{label}  (model={model}, improved_prompt={use_improved_prompt})\n{'='*70}")
    for name, ask_fn in [("기본       ", graph.ask), ("parent-child", gpc.ask)]:
        n_ok = 0
        for i in range(repeat):
            out = ask_fn(Q)
            ok = check(out["answer"])
            n_ok += ok
            print(f"  [{name}] {i+1}/{repeat} {'OK  ' if ok else 'FAIL'}  "
                  f"{out['answer'][:70].strip()}")
        print(f"  [{name}] → {n_ok}/{repeat} 정답")


def main():
    import config
    if not Path(config.PARENT_DB_DIR).exists():
        raise SystemExit("parent-child 인덱스가 없다: python src/ingest_parent_child.py")

    run_condition("A. 7B + 기존 프롬프트 (모델 용량 가설)", "qwen2.5:7b",
                  use_improved_prompt=False, repeat=3)
    run_condition("B. 3B + 개선 프롬프트 (프롬프트 가설)", "qwen2.5:3b",
                  use_improved_prompt=True, repeat=5)


if __name__ == "__main__":
    main()
