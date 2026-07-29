"""문서 인제스트 파이프라인: Markdown 로드 → 정제 → 청킹 → 임베딩 → FAISS 인덱스 저장."""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import vectorstore

# 노션 내보내기 특유의 노이즈. 특히 $\color{...}{About}$ 같은 LaTeX 장식이
# 핵심 청크(About Me = 경력 요약)의 임베딩을 오염시켜 검색 순위를 떨어뜨렸다.
_COLOR_MACRO = re.compile(r"\$\\color\{[^}]*\}\{([^}]*)\}\$")
_IMAGE_EMBED = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_EXTRA_BLANK = re.compile(r"\n{3,}")

# 아래 둘은 inspect_data.py(검수)로 뒤늦게 발견한 것들이다.
# 노션은 내부 페이지 링크를 퍼센트 인코딩된 한글 파일명으로 내보낸다
# ([Experience](%EC%9D%B4%EC%9C%A4%EC%84%A0%20...)). 링크 '텍스트'는 의미가
# 있지만 인코딩된 경로는 순수 노이즈이고, 이것만으로 코퍼스의 9.4%를 먹고 있었다.
# 주의: 경로에 괄호가 한 겹 들어가는 경우가 있고(...%EC%9B%90(kisti)%2020a...),
# 외부 http(s) URL에도 인코딩이 섞인다. 후자는 "깃허브 주소" 같은 질문에
# 답해야 하므로 반드시 남긴다 — 그래서 단순 치환이 아니라 함수로 판별한다.
_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)")
_HTML_TAG = re.compile(r"</?(?:br|div|span|aside|img|p)\b[^>]*/?>", re.I)

# 중첩된 링크([[A](url)(kisti) ](내부경로))는 바깥 텍스트에 ']'가 들어 있어
# _MD_LINK가 못 잡는다. 링크가 정리되고 나서 남은 '](인코딩 경로)' 조각을
# 마저 지운다 — 여기까지 와서 남아 있다면 짝을 잃은 잔여물이다.
_LEFTOVER_TARGET = re.compile(
    r"\]\((?!https?://)(?:[^()]|\([^()]*\))*%[0-9A-Fa-f]{2}(?:[^()]|\([^()]*\))*\)")

# 청크 평균 길이가 594자로 산문(평균 545자)보다 커 보이길래 재보니, 원인은
# portfolio.md에 박힌 ASCII 아키텍처 다이어그램(```markdown 펜스 안의 박스
# 그림)이었다 — 5개 펜스, 코퍼스 글자의 32%. 텍스트 스플리터가 800자 상한을
# 다이어그램 도중에 그어서 박스가 반토막 나는 청크가 나왔다(예: 파이프라인이
# "text_processor.py ← ITN" 줄에서 잘림). 문장 단위가 없는 콘텐츠라 문자
# 기준 분할이 애초에 안 맞는다 — 다이어그램은 통째로 하나의 청크로 둔다.
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_BOX_CHARS = re.compile(r"[┌┐└┘│─┼┬┴├┤═║╔╗╚╝▼►]")


def _is_diagram(fence: str) -> bool:
    """박스 그림 문자가 5% 넘게 섞인 펜스만 다이어그램으로 본다.

    같은 코드펜스라도 API 경로 예시(```POST /api/...```)처럼 검색 가치가
    있는 짧은 텍스트가 있어, "코드펜스면 무조건 분리"는 과하다.
    """
    return len(_BOX_CHARS.findall(fence)) / max(len(fence), 1) > 0.05


_DIAGRAM_PLACEHOLDER = "\n\n[다이어그램: {n}]\n\n"


def extract_diagrams(text: str) -> tuple[str, list[Document]]:
    """다이어그램 펜스를 본문에서 떼어내 플레이스홀더로 치환하고,
    떼어낸 것들은 각각 하나의 통짜 청크(Document)로 돌려준다.

    본문 청킹(RecursiveCharacterTextSplitter)이 이 청크들을 다시 자르지
    못하도록 ingest.main()에서 최종 chunks 리스트에 그대로 이어붙인다.
    """
    diagrams = []

    def _replace(m: re.Match) -> str:
        fence = m.group(0)
        if not _is_diagram(fence):
            return fence          # API 예시 등은 본문에 그대로 둔다
        diagrams.append(fence)
        return _DIAGRAM_PLACEHOLDER.format(n=len(diagrams))

    body = _CODE_FENCE.sub(_replace, text)
    # 플레이스홀더 앞뒤에 원래 있던 빈 줄과 합쳐져 3줄 이상 공백이 재발할 수
    # 있다 — clean_markdown의 공백 정리를 여기서 한 번 더 적용한다.
    body = _EXTRA_BLANK.sub("\n\n", body)
    docs = [Document(page_content=d, metadata={"source": "", "kind": "diagram"})
            for d in diagrams]
    return body, docs


def _strip_notion_link(m: re.Match) -> str:
    """노션 내부 링크는 텍스트만 남기고, 외부 URL 링크는 그대로 둔다."""
    full = m.group(0)
    target = full[full.rindex("](") + 2:-1]
    if target.startswith(("http://", "https://")):
        return full          # 실제 URL — 정보가 있으므로 보존
    if "%" in target:
        return m.group(1)    # 인코딩된 내부 경로 — 텍스트만 남긴다
    return full


def clean_markdown(text: str) -> str:
    """노션 내보내기 노이즈 제거.

    색상 매크로·인코딩 링크는 안의 텍스트만 남기고, 이미지 임베드와
    HTML 태그는 지운다. 정제 규칙을 늘릴 때는 inspect_data.py로
    "무엇이 남아 있는지" 먼저 확인하는 것이 순서다.
    """
    text = _COLOR_MACRO.sub(r"\1", text)
    text = _IMAGE_EMBED.sub("", text)
    text = _MD_LINK.sub(_strip_notion_link, text)
    text = _LEFTOVER_TARGET.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    return _EXTRA_BLANK.sub("\n\n", text)


def load_documents() -> tuple[list[Document], list[Document]]:
    """(본문 문서, 다이어그램 청크) — 다이어그램은 이미 청크 단위로 완성돼 있다."""
    docs, diagrams = [], []
    for path in sorted(Path(config.DOCS_DIR).glob("*.md")):
        text = clean_markdown(path.read_text(encoding="utf-8"))
        body, diag_docs = extract_diagrams(text)
        docs.append(Document(page_content=body, metadata={"source": path.name}))
        for d in diag_docs:
            d.metadata["source"] = path.name
        diagrams.extend(diag_docs)
    return docs, diagrams


def main():
    docs, diagrams = load_documents()
    if not docs:
        raise SystemExit(f"문서가 없습니다: {config.DOCS_DIR}")
    print(f"{len(docs)}개 문서 로드 (다이어그램 {len(diagrams)}개 분리)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)

    # 구분선('---')이나 제목 줄만 남은 조각은 검색에 걸릴 일이 없으면서
    # 인덱스 자리만 차지한다. 검수(inspect_data.py)로 발견해 걸러낸다.
    kept = [c for c in chunks
            if len(c.page_content) >= config.MIN_CHUNK_CHARS]
    dropped = len(chunks) - len(kept)
    chunks = kept

    # 산문 통계를 다이어그램과 분리해서 찍는다 — 다이어그램은 800자 상한이
    # 적용 안 된 통짜(수 KB)라 섞으면 "평균/최대"가 산문 실태를 안 보여준다.
    lengths = sorted(len(c.page_content) for c in chunks)
    print(f"{len(chunks)}개 산문 청크 "
          f"(최소 길이 {config.MIN_CHUNK_CHARS}자 미만 {dropped}개 제외) — "
          f"길이 최소 {lengths[0]} / 중앙값 {lengths[len(lengths) // 2]} / "
          f"평균 {sum(lengths) // len(lengths)} / 최대 {lengths[-1]}")

    # 다이어그램은 스플리터가 안 건드린 통짜 청크로 그대로 합류시킨다.
    # 800자 상한에 걸려 도중에 잘리던 것(예: 파이프라인이 중간 줄에서 끊김)을
    # 여기서 막는다 — 문장 단위가 없는 콘텐츠라 문자 기준 분할이 안 맞는다.
    if diagrams:
        dlen = sorted(len(d.page_content) for d in diagrams)
        print(f"+ 다이어그램 {len(diagrams)}개 (상한 미적용, 통짜 보존) — "
              f"길이 {dlen[0]}~{dlen[-1]}자")
    chunks += diagrams
    print(f"총 {len(chunks)}개 청크")

    path = vectorstore.build(chunks)
    print(f"{config.VECTOR_STORE} 인덱스 저장 완료: {path}")

    # BM25(키워드 검색) 재구축용 청크 원문 저장
    import json
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"page_content": c.page_content, "metadata": c.metadata},
                ensure_ascii=False) + "\n")
    print(f"청크 저장 완료: {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
