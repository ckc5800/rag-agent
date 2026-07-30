"""문서 인제스트 파이프라인: Markdown 로드 → 정제 → 청킹 → 임베딩 → FAISS 인덱스 저장."""
import hashlib
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


def chunk_fingerprint(path) -> str:
    """chunks.jsonl 내용 해시. 인덱스와의 결속 확인에 쓴다."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


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
        # 다이어그램을 **먼저** 떼어내고 본문만 정제한다. 반대 순서면 정제
        # 규칙(이미지·마크다운 링크·HTML 태그)이 펜스 안의 그림까지 건드린다 —
        # 통짜로 보존하는 게 다이어그램 청크의 존재 이유인데 그게 깨진다.
        # (현재 코퍼스의 다이어그램 3개는 정제 규칙에 걸리는 게 없어 이 순서
        #  변경으로 청크 내용은 바뀌지 않는다. 앞으로를 위한 방어다.)
        body, diag_docs = extract_diagrams(path.read_text(encoding="utf-8"))
        body = clean_markdown(body)
        docs.append(Document(page_content=body, metadata={"source": path.name}))
        for d in diag_docs:
            d.metadata["source"] = path.name
        diagrams.extend(diag_docs)
    return docs, diagrams


def annotate_positions(chunks: list[Document]) -> None:
    """청크에 위치 메타데이터를 붙인다 (page_content는 건드리지 않는다).

    지금까지 메타데이터가 {source, kind}뿐이라 할 수 없던 것들이 있다:

      · **이웃 확장** — 검색된 청크의 앞뒤를 함께 넘기는 방식. 근거를 더
        넣으려고 하위 랭크 청크를 추가하면 프롬프트 맨 앞(모델이 가장 못 보는
        자리)에 놓여 소용이 없다는 것을 두 번 확인했다(top-6, TOP_K=15).
        이웃 확장은 **상위 랭크 청크의 자리를 유지한 채** 그 주변만 넓힌다.
      · 정밀 인용 — 지금은 출처가 'resume.md'까지고 문서 어디인지 알 수 없다.

    부모를 '섹션'으로 잡는 방식은 이 코퍼스에서 안 된다. resume.md·
    publications.md·patents.md에 마크다운 헤딩이 각 1개(제목)뿐이다.
    그래서 문서 내 순번을 쓴다 — 헤딩 유무와 무관하게 항상 성립한다.

    page_content를 바꾸지 않으므로 gold 라벨(내용 md5)은 무효화되지 않는다.
    """
    per_source: dict[str, int] = {}
    for i, c in enumerate(chunks):
        src = c.metadata.get("source", "?")
        idx = per_source.get(src, 0)
        c.metadata["chunk_index"] = i          # 전역 순번 (이웃 확장용)
        c.metadata["doc_index"] = idx          # 문서 내 순번 (인용용)
        per_source[src] = idx + 1
    for c in chunks:
        c.metadata["doc_total"] = per_source[c.metadata.get("source", "?")]


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

    if not chunks and not diagrams:
        raise SystemExit(
            f"인덱싱할 청크가 없습니다 — {dropped}개가 전부 "
            f"MIN_CHUNK_CHARS({config.MIN_CHUNK_CHARS}자) 미만입니다.")

    # 산문 통계를 다이어그램과 분리해서 찍는다 — 다이어그램은 800자 상한이
    # 적용 안 된 통짜(수 KB)라 섞으면 "평균/최대"가 산문 실태를 안 보여준다.
    if chunks:
        lengths = sorted(len(c.page_content) for c in chunks)
        print(f"{len(chunks)}개 산문 청크 "
              f"(최소 길이 {config.MIN_CHUNK_CHARS}자 미만 {dropped}개 제외) — "
              f"길이 최소 {lengths[0]} / 중앙값 {lengths[len(lengths) // 2]} / "
              f"평균 {sum(lengths) // len(lengths)} / 최대 {lengths[-1]}")
    else:
        print(f"산문 청크 0개 (전부 {config.MIN_CHUNK_CHARS}자 미만으로 제외) "
              "— 다이어그램만 인덱싱한다")

    # 다이어그램은 스플리터가 안 건드린 통짜 청크로 그대로 합류시킨다.
    # 800자 상한에 걸려 도중에 잘리던 것(예: 파이프라인이 중간 줄에서 끊김)을
    # 여기서 막는다 — 문장 단위가 없는 콘텐츠라 문자 기준 분할이 안 맞는다.
    if diagrams:
        dlen = sorted(len(d.page_content) for d in diagrams)
        print(f"+ 다이어그램 {len(diagrams)}개 (상한 미적용, 통짜 보존) — "
              f"길이 {dlen[0]}~{dlen[-1]}자")
    chunks += diagrams
    annotate_positions(chunks)
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

    # 인덱스와 chunks.jsonl은 **같은 인제스트에서 나온 한 쌍**이어야 한다.
    # 벡터 검색은 인덱스를, BM25는 chunks.jsonl을 각각 읽으므로, 둘이 어긋나면
    # 서로 다른 청킹 두 개를 RRF로 섞게 되고 아무도 눈치채지 못한다.
    # (sweep_chunk_size.py가 두 파일을 매 스텝 덮어쓰므로 중간에 죽으면
    #  실제로 이 상태가 된다.) 청크 지문을 남겨 로드 시 대조한다.
    manifest = {
        "chunks_md5": chunk_fingerprint(config.CHUNKS_PATH),
        "n_chunks": len(chunks),
        "vector_store": config.VECTOR_STORE,
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "min_chunk_chars": config.MIN_CHUNK_CHARS,
    }
    Path(config.INDEX_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"인덱스 매니페스트 저장: {config.INDEX_MANIFEST}")


if __name__ == "__main__":
    main()
