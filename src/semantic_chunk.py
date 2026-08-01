"""시맨틱 청킹 — 고정 글자 수 대신 의미가 급변하는 지점에서 자른다.

기본 청킹(RecursiveCharacterTextSplitter)은 800자 상한을 마크다운 제목
경계 우선으로 채운다. 제목이 없는 긴 산문 구간(TTS 딥다이브 문서의 서술형
섹션 등)은 여전히 글자 수로만 끊긴다 — 문장 중간이 아니라 문단 경계에서
끊긴다는 보장은 없다.

시맨틱 청킹은 인접 문장 임베딩의 코사인 유사도가 급격히 떨어지는 지점을
주제 전환으로 보고 그 자리에서 자른다. `langchain_experimental`의
SemanticChunker를 새 의존성으로 추가하는 대신, 이미 쓰는 bge-m3 임베딩과
표준 라이브러리 수준(numpy만)으로 직접 구현했다 — 이 프로젝트가
bm25_tokenize·RRF·calculate를 전부 이렇게 짜 온 것과 같은 방향이다.

핵심 로직(find_breakpoints·merge_into_chunks)은 임베딩을 인자로 받는
순수 함수라 실제 Ollama 호출 없이 유닛 테스트가 가능하다. I/O(문장 분리,
임베딩 호출)는 `chunk_semantically()`가 감싼다.
"""
import re

import numpy as np

# 한국어 종결어미(다/요/음/함/임)나 영문 문장부호 뒤의 공백, 또는 줄바꿈을
# 문장 경계로 본다. 완벽한 형태소 분석이 아니라 청킹 입력용 근사치다.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[다요임음함])\s*\n+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text)]
    return [s for s in parts if s]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def find_breakpoints(embeddings: list[list[float]], percentile: float = 95.0) -> set[int]:
    """반환: 이 인덱스의 문장 뒤에서 자른다.

    인접 문장 쌍의 (1 - 코사인 유사도) = "거리"를 전부 구하고, 상위
    percentile을 넘는 거리만 breakpoint로 삼는다 — 대부분의 인접 문장은
    비슷한데(같은 문단), 갑자기 튀는 지점만 주제 전환으로 본다는 가정.
    LangChain SemanticChunker의 "percentile" 방식과 같은 발상이다.
    """
    if len(embeddings) < 2:
        return set()
    vecs = [np.array(e, dtype=float) for e in embeddings]
    dists = [1 - _cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    # 거리가 전부(또는 거의) 같으면 "튀는 지점"이 없다는 뜻이다. 이 경우
    # percentile(dists, N)이 그 값 자체와 같아져 d >= threshold가 전부
    # 참이 되는 퇴화 케이스가 생긴다 — 분산이 0에 가까우면 자르지 않는다.
    if np.std(dists) < 1e-9:
        return set()
    threshold = np.percentile(dists, percentile)
    return {i for i, d in enumerate(dists) if d >= threshold}


def merge_into_chunks(sentences: list[str], breakpoints: set[int],
                      max_chars: int = 800, min_chars: int = 30) -> list[str]:
    """breakpoint에서 자르되, max_chars 상한도 강제한다(안전판) — 주제
    전환이 한참 없는 구간이 무한정 길어지는 것을 막는다. 너무 짧은
    조각(min_chars 미만)은 다음 조각에 이어 붙인다."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for i, sent in enumerate(sentences):
        current.append(sent)
        current_len += len(sent) + 1
        at_break = i in breakpoints or current_len >= max_chars
        if at_break and i < len(sentences) - 1:
            chunks.append(" ".join(current))
            current, current_len = [], 0
    if current:
        chunks.append(" ".join(current))

    # 너무 짧은 조각은 이웃과 합친다 (구두점만 남은 조각 등)
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_chars:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return merged


def chunk_semantically(text: str, embed_fn, percentile: float = 95.0,
                       max_chars: int = 800, min_chars: int = 30) -> list[str]:
    """embed_fn: list[str] -> list[list[float]] (예: OllamaEmbeddings.embed_documents)."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences
    embeddings = embed_fn(sentences)
    breakpoints = find_breakpoints(embeddings, percentile)
    return merge_into_chunks(sentences, breakpoints, max_chars, min_chars)
