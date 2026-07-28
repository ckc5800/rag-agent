"""도구 체이닝 한계 측정 — 모델 크기를 바꿔가며 같은 코드로 잰다.

agent.py는 ReAct 루프(agent → tools → agent)라 구조적으로는 여러 도구를
이어 쓸 수 있다. 하지만 "구조가 지원한다"와 "이 모델이 실제로 해낸다"는
다르다. 그래서 필요한 도구 수를 1→2→3단으로 늘려가며 어디서 깨지는지 잰다.

측정 대상은 정답이 아니라 **호출된 도구 목록**이다. 답이 맞았는지는
모델의 산술 능력까지 섞여 들어가지만, 도구를 불렀는지 여부는 명확하다.

    python eval/eval_tool_chain.py --model qwen2.5:3b
    python eval/eval_tool_chain.py --model qwen2.5:7b

주의: 런마다 결과가 흔들린다. 처음 1회씩만 돌렸을 때 "3B는 2단 실패,
7B는 성공"으로 보였는데, 반복해 보니 3B도 2단을 해냈다. tool calling은
temperature 0.1에서도 재현되지 않으므로 --repeat으로 여러 번 돌려
성공 횟수로 봐야 한다. 1회 결과로 모델 간 차이를 말하면 틀린다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RESULTS = Path(__file__).parent / "results_tool_chain.json"

# 필요한 도구를 1개씩 늘려가는 세 질문.
# expected는 "이 도구들을 불러야 답할 수 있다"는 뜻이다.
CASES = [
    {"level": 1, "label": "계산만",
     "question": "TTFB가 2292ms에서 334ms로 줄었는데 몇 퍼센트 개선인지 계산해줘",
     "expected": ["calculate"]},
    {"level": 2, "label": "검색→계산",
     "question": "TTS 프로젝트에서 TTFB가 개선 전후로 몇 ms였는지 문서에서 찾아서, "
                 "몇 퍼센트 개선인지 계산해줘",
     "expected": ["search_portfolio", "calculate"]},
    {"level": 3, "label": "검색→날짜→계산",
     "question": "이윤선이 지금 회사에서 몇 년째 일하고 있어? 올해 기준으로 계산해줘",
     "expected": ["search_portfolio", "get_current_date", "calculate"]},
]


def main():
    ap = argparse.ArgumentParser(description="도구 체이닝 한계 측정")
    ap.add_argument("--model", default=None, help="config.LLM_MODEL 덮어쓰기")
    ap.add_argument("--repeat", type=int, default=3,
                    help="단계별 반복 횟수 (tool calling이 재현되지 않으므로 기본 3)")
    args = ap.parse_args()

    import config
    if args.model:
        config.LLM_MODEL = args.model
    import agent  # config를 바꾼 뒤 import해야 반영된다

    print(f"모델: {config.LLM_MODEL} (각 {args.repeat}회)\n")
    summary = []
    for case in CASES:
        runs = []
        for i in range(args.repeat):
            t0 = time.time()
            try:
                out = agent.run(case["question"])
                called, answer, err = out["tool_calls"], out["answer"], None
            except Exception as e:                   # noqa: BLE001
                called, answer, err = [], "", f"{type(e).__name__}: {e}"
            elapsed = time.time() - t0

            # 기대 도구를 전부 불렀는가 (순서·중복은 보지 않는다)
            ok = set(case["expected"]).issubset(set(called))
            runs.append({
                "called": called,
                "missing": [t for t in case["expected"] if t not in called],
                "pass": ok, "sec": round(elapsed),
                "answer": answer, "error": err,
            })
            print(f"  [{case['level']}단 {i+1}/{args.repeat}] "
                  f"{'PASS' if ok else 'FAIL'} ({elapsed:.0f}s)  {called}")

        n_pass = sum(r["pass"] for r in runs)
        med = sorted(r["sec"] for r in runs)[len(runs) // 2]
        print(f"[{case['level']}단 · {case['label']}] "
              f"{n_pass}/{args.repeat} 성공, 중앙값 {med}s\n")
        summary.append({
            "level": case["level"], "label": case["label"],
            "expected": case["expected"],
            "passed": n_pass, "of": args.repeat, "median_sec": med,
            "runs": runs,
        })

    print(f"===== {config.LLM_MODEL} =====")
    for s in summary:
        print(f"  {s['level']}단 {s['label']:<14} {s['passed']}/{s['of']} "
              f"(중앙값 {s['median_sec']}s)")

    # 모델별로 누적 저장 (다른 모델 결과를 덮어쓰지 않는다)
    all_results = {}
    if RESULTS.exists():
        all_results = json.loads(RESULTS.read_text(encoding="utf-8"))
    all_results[config.LLM_MODEL] = {"repeat": args.repeat, "levels": summary}
    RESULTS.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"\n저장: {RESULTS}")


if __name__ == "__main__":
    main()
