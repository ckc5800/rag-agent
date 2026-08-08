"""README와 코드가 어긋나지 않았는지 기계적으로 확인 (CI용).

문서 드리프트를 실제로 여러 번 겪었다 — README가 top-3을 설명하는데 코드는
top-5였고, "10문항"이라고 쓰인 채 51문항이 됐고, evaluate_team.py는 구조
트리에 아예 빠져 있었다. 사람이 눈으로 잡을 수 있는 종류가 아니다.

확인 항목:
  1. 구조 트리와 실제 파일 목록이 일치하는가 (양방향)
  2. README가 적어둔 실행 명령의 스크립트가 존재하는가
  3. README가 전제한 기본값이 config와 같은가
  4. 평가셋 문항 수 표기가 맞는가
  5. README가 언급한 환경변수 손잡이가 실제로 있는가

    python eval/audit_docs.py
"""
import pathlib
import re
import sys

# Windows 콘솔(cp949) 기본 인코딩에서 em dash(—) 등을 출력하면 UnicodeEncodeError로
# 죽는다. CI(Ubuntu)는 영향 없지만 로컬 실행(PYTHONUTF8 미설정)을 위해 강제한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path(__file__).resolve().parent.parent
readme = (REPO / "README.md").read_text(encoding="utf-8")
problems = []

# 1. 구조 트리에 적힌 파일이 실제로 있는가 / 있는 파일이 트리에 있는가
tree = readme.split("## 구조")[1].split("```")[1]
listed_src = set(re.findall(r"├──\s+(\w+\.py)|└──\s+(\w+\.py)", tree))
listed = {a or b for a, b in listed_src}
actual_src = {p.name for p in (REPO / "src").glob("*.py")}
actual_eval = {p.name for p in (REPO / "eval").glob("*.py")}
actual = actual_src | actual_eval

for f in sorted(actual - listed):
    problems.append(f"[트리 누락] {f} 가 저장소에 있는데 README 구조에 없음")
for f in sorted(listed - actual):
    problems.append(f"[트리 유령] {f} 가 README에 있는데 파일이 없음")

# 2. README가 언급한 실행 명령의 스크립트가 존재하는가
for m in re.finditer(r"python (src|eval)/(\w+\.py)", readme):
    if not (REPO / m.group(1) / m.group(2)).exists():
        problems.append(f"[명령 유령] {m.group(0)} — 파일 없음")

# 3. README가 주장하는 기본값이 config와 일치하는가
sys.path.insert(0, str((REPO / "src").resolve()))
import config  # noqa: E402

claims = {
    "GENERATE_TOP_N": 5,     # §5 "기본값을 top-5로 변경했다"
    "TOP_K": 6,              # §9 "TOP_K 6 → 15 미채택"
    "NEIGHBOR_WINDOW": 0,    # §9 "기본 꺼둠"
    "MAX_REWRITES": 1,
}
for k, v in claims.items():
    if getattr(config, k) != v:
        problems.append(
            f"[값 불일치] README는 {k}={v} 전제인데 config는 "
            f"{getattr(config, k)}")
if config.CONTEXT_ORDER != "reversed":
    problems.append("[값 불일치] README는 배치 순서 기본값을 역순으로 적었다")
if config.EXCLUDE_DIAGRAMS:
    problems.append("[값 불일치] README는 다이어그램 제외 기본 꺼짐으로 적었다")

# 4. 평가셋 문항 수 주장
import json  # noqa: E402
n = len(json.loads((REPO / "eval/eval_set.json").read_text(encoding="utf-8")))
if f"{n}문항" not in readme:
    problems.append(f"[수치 불일치] 평가셋은 {n}문항인데 README에 그 표기가 없음")

# 5. 환경변수 손잡이가 실제로 존재하는가
for name in re.findall(r"`(TOP_K|GENERATE_TOP_N|NEIGHBOR_WINDOW|CONTEXT_ORDER|"
                       r"EXCLUDE_DIAGRAMS|GENERATE_PROMPT_VARIANT|MAX_REWRITES|"
                       r"LLM_MODEL)`", readme):
    if not hasattr(config, name):
        problems.append(f"[손잡이 없음] README가 {name}을 언급하는데 config에 없음")

# 6. A/B·스윕 스크립트가 환경 지문(runmeta)을 기록하는가
#
# 이 저장소는 "90% vs 74%가 비교 가능한지 판단할 근거가 결과 파일에 없다"로
# 한 번 데였고 runmeta.py를 만들었다. 그런데 실제로 쓰는 건 evaluate.py
# 하나뿐이었고 A/B·스윕 9개가 전부 빠져 있었다 — 3B 결과와 14B 결과가 한
# 폴더에 섞여도 파일만 보고는 구분할 수 없었다. 관례를 사람 기억이 아니라
# 빌드로 강제한다.
for p in sorted((REPO / "eval").glob("ab_*.py")) + sorted((REPO / "eval").glob("sweep_*.py")):
    src = p.read_text(encoding="utf-8")
    if "run_metadata" not in src:
        problems.append(
            f"[지문 누락] {p.name} 이 결과를 저장하는데 runmeta.run_metadata()를 "
            "기록하지 않음 (모델·인덱스를 모르면 나중에 비교 불가)")

if problems:
    print("문서-코드 불일치:")
    for p in problems:
        print("  " + p)
    raise SystemExit(1)
print(f"문서-코드 일치 확인 — src {len(actual_src)}개 / eval {len(actual_eval)}개, "
      f"평가셋 {n}문항, 기본값 {len(claims) + 2}개 대조 통과")
