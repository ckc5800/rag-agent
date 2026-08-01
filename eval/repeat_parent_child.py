"""기본 vs Parent-Child를 실제 파이프라인(grade→rewrite 루프 포함)으로
여러 번 반복해 비교한다.

교훈: 격리된 단일 호출(hybrid_search_child + generate를 직접 부르는 것)로
"PARENT_TOP_N=1이 오답을 고쳤다"고 판단했다가, 전체 파이프라인으로 다시
돌리니 결과가 뒤집혔다. temperature=0이어도 이 3B 모델은 완전히 결정적이지
않고(동점 로짓·부동소수점), 격리 호출은 grade/rewrite 루프를 안 타 실제
경로와 다르다. eval_tool_chain.py에서 배운 것과 같은 교훈 — n=1로 결론
내리면 안 되고, 실제로 쓰이는 전체 경로로 반복해야 한다.

    python eval/repeat_parent_child.py --repeat 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

QUESTIONS = {
    "TTS 프로젝트에서 TTFB를 얼마나 개선했나요?": ["2292", "334"],
    "이윤선의 제1저자 논문은 몇 편인가요?": ["7편", "7 편"],
}


def check(answer: str, patterns: list[str]) -> bool:
    return any(p in answer for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    import config

    if not Path(config.PARENT_DB_DIR).exists():
        raise SystemExit("parent-child 인덱스가 없다: python src/ingest_parent_child.py")

    import graph
    import graph_parent_child as gpc

    for q, patterns in QUESTIONS.items():
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        for label, ask_fn in [("기본       ", graph.ask), ("parent-child", gpc.ask)]:
            n_ok = 0
            for i in range(args.repeat):
                out = ask_fn(q)
                ok = check(out["answer"], patterns)
                n_ok += ok
                mark = "OK " if ok else "FAIL"
                print(f"  [{label}] {i+1}/{args.repeat} {mark}  "
                      f"{out['answer'][:70].strip()}")
            print(f"  [{label}] → {n_ok}/{args.repeat} 정답")


if __name__ == "__main__":
    main()
