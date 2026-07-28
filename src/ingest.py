"""문서 인제스트 파이프라인: Markdown 로드 → 정제 → 청킹 → 임베딩 → FAISS 인덱스 저장."""
import re
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

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


def load_documents() -> list[Document]:
    docs = []
    for path in sorted(Path(config.DOCS_DIR).glob("*.md")):
        text = clean_markdown(path.read_text(encoding="utf-8"))
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit(f"문서가 없습니다: {config.DOCS_DIR}")
    print(f"{len(docs)}개 문서 로드")

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

    lengths = sorted(len(c.page_content) for c in chunks)
    print(f"{len(chunks)}개 청크 생성 "
          f"(최소 길이 {config.MIN_CHUNK_CHARS}자 미만 {dropped}개 제외) — "
          f"길이 최소 {lengths[0]} / 중앙값 {lengths[len(lengths) // 2]} / "
          f"평균 {sum(lengths) // len(lengths)} / 최대 {lengths[-1]}")

    # 기존 DB 삭제 후 재구축 (멱등성 보장)
    if Path(config.DB_DIR).exists():
        shutil.rmtree(config.DB_DIR)

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(config.DB_DIR)
    print(f"FAISS 인덱스 저장 완료: {config.DB_DIR}")

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
