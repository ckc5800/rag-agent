"""corrective 루프 A/B — 본 평가셋 층화 표본(유형별 2문항)으로 재검증.

ab_rewrite.py의 원래 A/B(10문항, OFF 50%→ON 60%)는 10문항 시절 평가셋
기준이라 comparison·aggregation 유형이 아예 없었다. 평가셋 전체
(51×2조건×repeat)는 CPU 추론 기준 수 시간이 걸려 이번엔 유형별 2문항씩
층화 표본(7유형×2=14문항)으로 시간을 줄이되 유형 다양성은 확보한다.

전체 검증이 아니라 **층화 표본**이라는 것을 결과에 명시한다.
(문항 수는 eval_set.json 에서 읽는다 — 51→63→71 로 늘어났으므로
 숫자를 문자열에 박아두면 곧 거짓이 된다.)

    python eval/ab_rewrite_stratified.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_rewrite import run_condition  # noqa: E402
from runmeta import run_metadata  # noqa: E402

RESULTS = Path(__file__).parent / "results_ab_rewrite_stratified.json"
EVAL_SET = Path(__file__).parent / "eval_set.json"
PER_TYPE = 2


def stratified_sample(cases: list[dict], per_type: int) -> list[dict]:
    by_type = defaultdict(list)
    for c in cases:
        by_type[c.get("type", "?")].append(c)
    sample = []
    for t in sorted(by_type):
        sample.extend(by_type[t][:per_type])
    return sample


def main() -> int:
    import config

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    n_total = len(cases)
    sample = stratified_sample(cases, PER_TYPE)
    types = sorted({c.get("type", "?") for c in sample})
    print(f"층화 표본 {len(sample)}문항 (유형 {len(types)}종 × 최대 {PER_TYPE}개: {types})\n")

    repeat = 3
    total_calls = len(sample) * 2 * repeat
    print(f"{len(sample)}문항 × 2조건 × {repeat}회 = RAG 실행 {total_calls}회\n")

    print("── OFF: MAX_REWRITES=0 (grade·rewrite 없음) ──")
    off = run_condition(sample, 0, repeat)
    print("\n── ON: MAX_REWRITES=1 (현재 동작) ──")
    on = run_condition(sample, 1, repeat)

    print("\n===== 문항별 비교 (통과 횟수) =====")
    rescued, broken, unchanged = [], [], 0
    by_type_delta = defaultdict(lambda: [0, 0])  # [rescued, broken]
    for a, b, case in zip(off, on, sample):
        delta = b["passed"] - a["passed"]
        t = case.get("type", "?")
        if delta > 0:
            tag, rescued = "UP", rescued + [a["question"]]
            by_type_delta[t][0] += 1
        elif delta < 0:
            tag, broken = "DN", broken + [a["question"]]
            by_type_delta[t][1] += 1
        else:
            tag, unchanged = "  ", unchanged + 1
        print(f"  {tag} [{t:<11}] OFF {a['passed']}/{a['of']} -> ON {b['passed']}/{b['of']}"
              f"  (ON 재작성 {b['rewrote']}/{b['of']}회) {a['question'][:36]}")

    off_rate = sum(r["passed"] for r in off) / max(sum(r["of"] for r in off), 1)
    on_rate = sum(r["passed"] for r in on) / max(sum(r["of"] for r in on), 1)

    # ab_rewrite.py와 같은 분리 — ON에서 재작성이 안 일어난 문항은 generate가
    # 받는 문서가 OFF와 같아 사실상 같은 경로다. 그 열의 흔들림이 노이즈다.
    var_idx = [i for i, b in enumerate(on) if b["rewrote"] > 0]
    ctl_idx = [i for i, b in enumerate(on) if b["rewrote"] == 0]

    def tally(idx, rows):
        return sum(rows[i]["passed"] for i in idx), sum(rows[i]["of"] for i in idx)

    v_off, v_tot = tally(var_idx, off)
    v_on, _ = tally(var_idx, on)
    c_off, c_tot = tally(ctl_idx, off)
    c_on, _ = tally(ctl_idx, on)

    print(f"\n정답률(층화표본, 섞은 값)   OFF {off_rate * 100:.0f}%  ->  ON {on_rate * 100:.0f}%")
    print(f"변수(실제 재작성 {len(var_idx)}문항)   : {v_off}/{v_tot} -> {v_on}/{v_tot}"
          f"  ({v_on - v_off:+d}판정)")
    print(f"대조군(재작성 0회 {len(ctl_idx)}문항): {c_off}/{c_tot} -> {c_on}/{c_tot}"
          f"  ({c_on - c_off:+d}판정)  <- 노이즈")
    print(f"구제 {len(rescued)}개 / 악화 {len(broken)}개 / 변화없음 {unchanged}개")
    print("유형별 구제/악화:")
    for t in sorted(by_type_delta):
        up, dn = by_type_delta[t]
        print(f"  {t:<12} 구제 {up} / 악화 {dn}")
    if rescued:
        print("  구제 질문: " + ", ".join(q[:30] for q in rescued))
    if broken:
        print("  악화 질문: " + ", ".join(q[:30] for q in broken))
    print(f"\n주의: 전체 {n_total}문항이 아니라 유형별 {PER_TYPE}문항 층화 표본"
          f"({len(sample)}문항)이다. 10문항 원 A/B(50%->60%)와 직접 비교 불가"
          " — 채점 기준·평가셋이 다르다.")

    RESULTS.write_text(json.dumps({
        "sample_size": len(sample), "per_type": PER_TYPE, "repeat": repeat,
        "model": config.LLM_MODEL,
        "env": run_metadata(),
        # 합계는 참고용 — 변수와 대조군이 섞여 있다.
        "off_accuracy": round(off_rate * 100, 1),
        "on_accuracy": round(on_rate * 100, 1),
        # 판단은 이 둘로 한다.
        "rewritten": {"off": v_off, "on": v_on, "judgements": v_tot,
                      "questions": len(var_idx)},
        "control_no_rewrite": {"off": c_off, "on": c_on, "judgements": c_tot,
                               "questions": len(ctl_idx)},
        "rescued": rescued, "broken": broken,
        "by_type_delta": dict(by_type_delta),
        "off": off, "on": on,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
