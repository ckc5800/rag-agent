"""인덱싱 전 데이터 검수 — 파이프라인의 0단계.

이 스크립트가 없어서 겪은 일: 노션 내보내기의 `$\\color{...}$` 장식이
경력 요약 청크의 임베딩을 오염시켜 "근무한 회사들" 질문이 계속 실패했다.
원인을 찾은 건 일주일 뒤 검색/생성 분리 평가였지만, 청크를 몇 개만
열어봤으면 5분이면 보였을 문제였다.

노이즈를 미리 다 예측하는 건 불가능하다. 대신 인덱싱할 텍스트를
"반드시 한 번은 눈으로 보게" 만드는 것이 이 스크립트의 목적이다.

    python src/inspect_data.py           # 리포트 출력
    python src/inspect_data.py --strict  # 잔여 아티팩트가 있으면 exit 1 (CI용)
"""
import argparse
import re
import sys
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingest import load_documents

# 실제 http(s) URL. 인코딩 검사에서 제외하려고 먼저 지운다 —
# 논문 링크(...articleTitle=GAN%EC%9D%84+...)의 인코딩은 정상이다.
_REAL_URL = re.compile(r"https?://\S+")

# 정제가 놓치면 안 되는 것들. --strict에서 CI를 실패시킨다.
DEFECTS = {
    "LaTeX 색상 매크로": re.compile(r"\$\\color\{"),
    "이미지 임베드": re.compile(r"!\[[^\]]*\]\("),
    "HTML 태그": re.compile(r"</?(?:br|div|span|aside|img|p)\b[^>]*>", re.I),
    "노션 내부 링크(인코딩 경로)": re.compile(r"%[0-9A-Fa-f]{2}"),
    "3줄 이상 연속 공백": re.compile(r"\n{3,}"),
}

# 노이즈인지 신호인지 기계가 정할 수 없는 것들. 보고만 하고 통과시킨다.
# 검수기의 역할은 후보를 찾아 주는 것이지 대신 판단하는 것이 아니다.
REVIEW = {
    "mermaid 다이어그램": re.compile(
        r"```mermaid|^\s*(?:graph|sequenceDiagram|flowchart)\s", re.M),
}

# 인코딩 검사만 URL을 걷어낸 텍스트에 적용한다.
_URL_SENSITIVE = {"노션 내부 링크(인코딩 경로)"}


def scan(text: str, table: dict) -> list[str]:
    """텍스트에서 발견된 항목 이름 목록."""
    stripped = _REAL_URL.sub(" ", text)
    return [name for name, pat in table.items()
            if pat.search(stripped if name in _URL_SENSITIVE else text)]


def bar(count: int, width: int = 30) -> str:
    return "#" * min(count, width)


def main() -> int:
    ap = argparse.ArgumentParser(description="인덱싱 전 데이터 검수")
    ap.add_argument("--strict", action="store_true",
                    help="잔여 아티팩트가 있으면 exit 1 (CI용)")
    ap.add_argument("--show", type=int, default=5,
                    help="짧은 청크를 몇 개까지 보여줄지")
    args = ap.parse_args()

    docs, diagrams = load_documents()  # clean_markdown + 다이어그램 분리가 적용된 상태
    if not docs:
        print(f"[FAIL] 검수할 문서가 없습니다: {config.DOCS_DIR}")
        return 1
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs) + diagrams

    print(f"문서 {len(docs)}개 → 청크 {len(chunks)}개 "
          f"(상한 {config.CHUNK_SIZE}자 / overlap {config.CHUNK_OVERLAP}자, "
          f"다이어그램 {len(diagrams)}개는 상한 미적용 통짜 청크)\n")

    # ── 1. 길이 분포 ── 다이어그램은 상한 미적용 통짜라 별도 콘텐츠 종류로
    # 취급한다. 같이 섞으면 "상한 초과"가 결함처럼 보고돼 오해를 산다.
    prose = [c for c in chunks if c.metadata.get("kind") != "diagram"]
    lengths = sorted(len(c.page_content) for c in prose)
    total = sum(lengths)
    n = len(lengths)
    print("── 길이 분포 (산문 청크만; 다이어그램 {}개는 제외) ──".format(len(diagrams)))
    if n:
        edges = [0, 100, 200, 300, 400, 500, 600, 700, config.CHUNK_SIZE + 1]
        for lo, hi in zip(edges, edges[1:]):
            c = sum(1 for x in lengths if lo <= x < hi)
            print(f"  {lo:>4}~{hi - 1:<4} {bar(c):<30} {c:>3}개")
        print(f"\n  최소 {lengths[0]} / 중앙값 {lengths[n // 2]} / "
              f"평균 {total // n} / 최대 {lengths[-1]}")
        over = [x for x in lengths if x > config.CHUNK_SIZE]
        print(f"  상한 초과: {len(over)}개")
        tiny = [x for x in lengths if x < config.MIN_CHUNK_CHARS]
        print(f"  최소 길이({config.MIN_CHUNK_CHARS}자) 미만: {len(tiny)}개 "
              f"— 전체 글자의 {sum(tiny) / total * 100:.1f}%")
    else:
        print("  산문 청크 없음 (다이어그램만 있음)")
    if diagrams:
        dlen = sorted(len(d.page_content) for d in diagrams)
        print(f"  다이어그램 {len(diagrams)}개 — 최소 {dlen[0]} / 최대 {dlen[-1]}자 "
              f"(상한 미적용, 통짜 보존이 의도)\n")
    else:
        print()

    # ── 2. 가장 짧은 청크 (여기에 쓰레기가 모인다) ──────
    print(f"── 가장 짧은 청크 {args.show}개 ──")
    for c in sorted(chunks, key=lambda c: len(c.page_content))[:args.show]:
        body = c.page_content.replace("\n", "\\n")[:70]
        mark = "  DROP" if len(c.page_content) < config.MIN_CHUNK_CHARS else "  keep"
        print(f"{mark} {len(c.page_content):>4}자 [{c.metadata['source']}] {body!r}")
    print()

    # ── 3. 잔여 아티팩트 스캔 ─────────────────────────
    def report(title: str, table: dict) -> Counter:
        print(f"── {title} ──")
        hits = Counter()
        examples: dict[str, tuple[str, str]] = {}
        for c in chunks:
            for name in scan(c.page_content, table):
                hits[name] += 1
                examples.setdefault(name, (c.metadata["source"], c.page_content))
        if not hits:
            print("  없음")
        for name, cnt in hits.most_common():
            src, text = examples[name]
            # 스캔한 것과 같은 텍스트에서 위치를 잡아야 스니펫이 실제 원인을 가리킨다
            probe = _REAL_URL.sub(" ", text) if name in _URL_SENSITIVE else text
            m = table[name].search(probe)
            snippet = probe[max(0, m.start() - 25):m.start() + 45].replace("\n", "\\n")
            print(f"  [{cnt:>2}청크] {name}")
            print(f"           최초 발견 {src}: ...{snippet}...")
        print()
        return hits

    defects = report("결함 — 정제가 놓친 것", DEFECTS)
    review = report("판단 필요 — 사람이 정할 것", REVIEW)
    if review:
        print("  ※ mermaid는 문법(graph/subgraph/화살표)이 노이즈지만 노드 라벨은"
              "\n     실제 기술 키워드다. 지우면 키워드까지 잃으므로 남겨 두었다.\n")

    print(f"출처별 청크: {dict(Counter(c.metadata['source'] for c in chunks))}")

    if args.strict and defects:
        print(f"\n[FAIL] 결함 {len(defects)}종이 인덱싱 대상에 남아 있다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
