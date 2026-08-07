"""KLUE-RE 2차 코퍼스 인덱스 구축 — 하이브리드 인덱스 + 정답 그래프 + 평가셋.

정답 그래프는 LLM 추출이 아니라 KLUE-RE의 전문가 라벨을 직접 쓴다(트리플
노이즈 원천 차단). 원 코퍼스 실험(README "Graph RAG 실험")에서 실패 원인의
상당수가 LLM 추출 노이즈(동사가 엔티티로 새는 등)였다 — 이 트랙에서는 그
변수를 없애고 "그래프 구조 자체가 이 규모·밀도에서 값을 하는가"만 순수하게
잰다.

사용: python src/ingest_klue.py
"""
import json

import networkx as nx
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

import config
import kg
import klue_re


def main():
    klue_re.KLUE_DIR.mkdir(parents=True, exist_ok=True)

    print("KLUE-RE 로드 중 (klue/klue, config=re)...")
    rows = klue_re.load_rows()
    chunks, triples_by_chunk = klue_re.build_chunks(rows)
    n_with_triples = sum(1 for t in triples_by_chunk.values() if t)
    print(f"고유 문장(청크) {len(chunks)}개 (트리플 있는 문장 {n_with_triples}개, "
          f"방해 요소 {len(chunks) - n_with_triples}개)")

    print(f"임베딩({config.EMBED_MODEL}) + FAISS 인덱스 구축 중...")
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(klue_re.KLUE_DB_DIR)

    with open(klue_re.KLUE_CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"page_content": c.page_content, "metadata": c.metadata},
                              ensure_ascii=False) + "\n")
    print(f"인덱스 저장: {klue_re.KLUE_DB_DIR}")

    print("정답 그래프 구축 중 (LLM 없음 — 전문가 라벨을 그대로 트리플로 사용)...")
    g = nx.MultiDiGraph()
    n_triples = 0
    for idx, triples in triples_by_chunk.items():
        n_triples += kg.add_triples(g, triples, chunk_index=idx, source="klue_re")
    kg.save(g, klue_re.KLUE_GRAPH_PATH)
    print(f"그래프: 노드 {g.number_of_nodes()}개, 엣지 {g.number_of_edges()}개 "
          f"(트리플 {n_triples}개)")

    print("평가 질문 생성 중...")
    questions = klue_re.build_questions(chunks, triples_by_chunk)
    klue_re.KLUE_RETRIEVAL_SET.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"질문 {len(questions)}개 저장: {klue_re.KLUE_RETRIEVAL_SET}")


if __name__ == "__main__":
    main()
