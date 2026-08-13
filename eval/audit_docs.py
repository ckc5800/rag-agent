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

# config.knobs()가 손잡이 전부를 돌려주므로 손으로 고른 몇 개가 아니라
# 전수로 확인한다. 예전엔 4개만 적어 뒀고, 그 목록 자체가 드리프트했다
# (cache.py·runmeta.py가 같은 병으로 실제 버그를 냈다).
for name, value in sorted(config.knobs().items()):
    if name not in readme:
        problems.append(
            f"[손잡이 미문서화] {name}(기본 {value}) 가 README에 없음 — "
            "손잡이를 노출했으면 무엇인지·왜 그 기본값인지 적어야 한다")

# 3b. A/B 스크립트의 실행 경로가 명시돼 있는가
#
# 프로덕션(api/cli)은 graph.run/ask로 TYPE_ROUTING 오버라이드를 거친다.
# A/B가 graph.invoke를 직접 부르면 그 경로를 우회하는데, 우회가 암묵적이면
# 나중에 run()으로 바꾸는 순간 조용히 다른 것을 재게 된다 — ab_rewrite는
# 실제로 경로를 바꾸자 결론의 부호가 뒤집혔다. invoke를 쓰는 스크립트는
# 격리를 코드로 선언하게 강제한다.
for script in sorted((REPO / "eval").glob("ab_*.py")):
    body = script.read_text(encoding="utf-8")
    if '.invoke({"question"' not in body:
        continue                      # ask/run 사용 또는 질의를 안 돌리는 스크립트
    if "TYPE_ROUTING = False" not in body:
        problems.append(
            f"[격리 미선언] {script.name} 이 graph.invoke로 프로덕션 경로를 "
            "우회하는데 config.TYPE_ROUTING = False 선언이 없다")

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

# 6-b. **산출물에도 지문이 실제로 들어갔는가.**
#
# 위 검사는 스크립트 소스만 본다 — 그래서 "게이트는 통과하는데 결과 파일엔
# 지문이 없는" 구멍이 남았다(실측: 14개 중 8개). 지문을 넣기 **전에** 돌려
# 둔 결과가 그대로 남아 3b·7b·14b 결과와 한 폴더에 섞이는데, 정작 파일만
# 봐서는 구분할 수 없다 — 지문을 도입한 이유가 바로 그거였다.
#
# 새로 만드는 결과에는 지문을 강제하되, 지문 도입 이전 산출물은 **명시적으로
# 열거**해 둔다. 지우거나 되돌려 쓰지 않는 이유는 이 저장소가 측정 기록을
# 증거로 보존하기 때문이고, 목록에 적어 두면 "빠졌다"가 눈에 보인다.
# 목록을 늘리려면 그 실험을 다시 돌려 지문과 함께 저장하는 것이 원칙이다.
LEGACY_NO_FINGERPRINT = {
    "results_ab_hyde.json", "results_ab_hyde_generate.json",
    "results_ab_neighbor_window.json", "results_ab_prompt_variant.json",
    "results_ab_rerank.json", "results_ab_route_variants.json",
    "results_ab_top_n.json", "results_seed_match_ratio.json",
    "results_top_k.json",
}
for p in sorted((REPO / "eval").glob("ab_*.py")) + sorted((REPO / "eval").glob("sweep_*.py")):
    src = p.read_text(encoding="utf-8")
    m = re.search(r"(results_[a-z0-9_]+\.json)", src)
    if not m:
        continue
    out = REPO / "eval" / m.group(1)
    if not out.exists() or m.group(1) in LEGACY_NO_FINGERPRINT:
        continue
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        problems.append(f"[산출물 손상] {m.group(1)} 을 JSON 으로 읽을 수 없음")
        continue
    if not (isinstance(data, dict) and data.get("env")):
        problems.append(
            f"[산출물 지문 누락] {m.group(1)} 에 env 가 없다 — {p.name} 은 "
            "run_metadata() 를 부르는데 저장물엔 안 들어갔다. 지문 도입 이전 "
            "산출물이면 audit_docs.py 의 LEGACY_NO_FINGERPRINT 에 적을 것")

if problems:
    print("문서-코드 불일치:")
    for p in problems:
        print("  " + p)
    raise SystemExit(1)
print(f"문서-코드 일치 확인 — src {len(actual_src)}개 / eval {len(actual_eval)}개, "
      f"평가셋 {n}문항, 손잡이 {len(config.knobs())}개 문서화 확인")
