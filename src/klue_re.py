"""KLUE-RE(공개 관계추출 벤치마크) 기반 2차 코퍼스 — Graph RAG를 실제 규모에서
재검증한다.

동기: 이 프로젝트의 원 코퍼스(이력서 등 5개 문서, 48청크)는 하이브리드 검색이
이미 recall@1 90%로 천장이라 Graph RAG가 힘을 쓸 여지가 없었다(README
"Graph RAG 실험" 절). KLUE-RE는 위키피디아·뉴스 문장에 전문가가 (주어,관계,
목적어) 정답 라벨을 붙인 공개 데이터셋(3.2만 문장, cc-by-sa-4.0)이라:
  (1) LLM 추출 노이즈 없이 **정답 그래프**를 바로 만들 수 있고,
  (2) 같은 인물·조직이 여러 문장에 흩어져 등장해 그래프가 유리할 수 있는
      실제 구조(엔티티 중심의 다대다 관계)를 갖는다.

문장을 "문서"가 아니라 "검색 단위"로 쓴다 — 실제 문서 하나가 여러 청크로
쪼개지는 구조는 아니지만, 검색 벤치마크(recall@k / MRR)로는 유효하다.
"""
import hashlib
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

import config
from kg import Triple

DATASET_ID = "klue/klue"
DATASET_CONFIG = "re"

KLUE_DIR: Path = config.BASE_DIR / "data" / "klue"
KLUE_DB_DIR = str(KLUE_DIR / "faiss_index")
KLUE_CHUNKS_PATH = KLUE_DIR / "chunks.jsonl"
KLUE_GRAPH_PATH = KLUE_DIR / "knowledge_graph.json"
KLUE_RETRIEVAL_SET = KLUE_DIR / "retrieval_set.json"

N_CHUNKS = 1500          # 최종 고유 문장(청크) 수 목표
N_QUESTIONS = 120        # 생성할 평가 질문 수
MAX_PER_RELATION = 6     # 한 관계 유형이 평가셋을 독식하지 않도록 상한

# 관계 라벨 → 한국어 질문 템플릿. no_relation(라벨 0)은 질문을 만들 사실이
# 없으므로 제외한다 — 그래도 청크(문장)로는 남아 검색의 방해 요소(현실적
# 코퍼스에 흔한, 정답과 무관한 문장) 역할을 한다. KLUE-RE 라벨 29종 전부 매핑.
TEMPLATES: dict[str, str] = {
    "org:dissolved": "{s}{eun_neun} 언제 해체(폐업)되었나요?",
    "org:founded": "{s}{eun_neun} 언제 설립되었나요?",
    "org:place_of_headquarters": "{s}의 본사는 어디에 있나요?",
    "org:alternate_names": "{s}의 다른 이름(별칭)은 무엇인가요?",
    "org:member_of": "{s}{eun_neun} 어디에 소속되어 있나요?",
    "org:members": "{s}의 소속 구성원(회원)은 누구인가요?",
    "org:political/religious_affiliation": "{s}{eun_neun} 어떤 정치·종교 성향과 관련이 있나요?",
    "org:product": "{s}{i_ga} 만든 제품(생산물)은 무엇인가요?",
    "org:founded_by": "{s}{eul_reul} 설립한 사람(단체)은 누구인가요?",
    "org:top_members/employees": "{s}의 대표(임원)는 누구인가요?",
    "org:number_of_employees/members": "{s}의 구성원(직원) 수는 몇 명인가요?",
    "per:date_of_birth": "{s}{eun_neun} 언제 태어났나요?",
    "per:date_of_death": "{s}{eun_neun} 언제 사망했나요?",
    "per:place_of_birth": "{s}{eun_neun} 어디에서 태어났나요?",
    "per:place_of_death": "{s}{eun_neun} 어디에서 사망했나요?",
    "per:place_of_residence": "{s}{eun_neun} 어디에 거주하나요?",
    "per:origin": "{s}의 국적(출신)은 어디인가요?",
    "per:employee_of": "{s}{eun_neun} 어느 조직(회사)에 소속되어 있나요?",
    "per:schools_attended": "{s}{eun_neun} 어느 학교를 다녔나요?",
    "per:alternate_names": "{s}의 다른 이름(별칭)은 무엇인가요?",
    "per:parents": "{s}의 부모는 누구인가요?",
    "per:children": "{s}의 자녀는 누구인가요?",
    "per:siblings": "{s}의 형제자매는 누구인가요?",
    "per:spouse": "{s}의 배우자는 누구인가요?",
    "per:other_family": "{s}의 다른 가족 관계는 누구인가요?",
    "per:colleagues": "{s}의 동료는 누구인가요?",
    "per:product": "{s}{i_ga} 관여한 제품(작품)은 무엇인가요?",
    "per:religion": "{s}의 종교는 무엇인가요?",
    "per:title": "{s}의 직함(역할)은 무엇인가요?",
}


def _has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지 — 한글 완성형 유니코드 오프셋으로 판별
    (코드 - 0xAC00) % 28 == 0 이면 받침 없음. 한글이 아니면 받침 없음으로 본다."""
    ch = word.strip()[-1] if word.strip() else ""
    if not ("가" <= ch <= "힣"):
        return False
    return (ord(ch) - 0xAC00) % 28 != 0


def render_question(relation: str, subject: str) -> str:
    """관계·주어로 질문 문장을 만든다. 받침 유무로 조사(은/는·이/가·을/를)를
    맞춘다 — "는"만 기계적으로 붙이면 "이윤선는"처럼 비문이 된다."""
    b = _has_batchim(subject)
    return TEMPLATES[relation].format(
        s=subject, eun_neun="은" if b else "는",
        i_ga="이" if b else "가", eul_reul="을" if b else "를")


def load_rows(oversample: int = 3000, stride: int | None = None):
    """원본 row(HF Dataset) — 결정적 계통 표본. `datasets` 라이브러리는 이
    함수 안에서만 임포트해, 이 모듈을 쓰지 않는 다른 코드가 무거운 의존성을
    강제로 물지 않게 한다(visual-search·demand-forecast와 같은 계통 표본
    원칙 — 무작위 시드 대신 결정적 stride)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    n = len(ds)
    stride = stride or max(1, n // oversample)
    idx = list(range(0, n, stride))[:oversample]
    return ds.select(idx)


def build_chunks(rows, n_chunks: int = N_CHUNKS) -> tuple[list[Document], dict[int, list[Triple]]]:
    """행을 고유 문장 단위로 합쳐 청크를 만든다. (청크 목록, chunk_index→트리플
    목록) 반환. 같은 문장에 관계 라벨이 여러 개(다른 개체쌍) 달린 경우를 하나의
    청크로 합친다 — 순수 함수, LLM 호출 없음(단위 테스트 대상)."""
    label_names = rows.features["label"].names
    by_sentence: dict[str, list[Triple]] = defaultdict(list)
    order: list[str] = []
    for r in rows:
        sent = r["sentence"]
        if sent not in by_sentence:
            order.append(sent)
            by_sentence[sent] = []
        label = label_names[r["label"]]
        if label != "no_relation" and label in TEMPLATES:
            by_sentence[sent].append(Triple(
                subject=r["subject_entity"]["word"],
                relation=label,
                object=r["object_entity"]["word"],
            ))

    kept = order[:n_chunks]
    chunks = [Document(page_content=sent, metadata={"chunk_index": i, "source": "klue_re"})
              for i, sent in enumerate(kept)]
    triples_by_chunk = {i: by_sentence[sent] for i, sent in enumerate(kept)}
    return chunks, triples_by_chunk


def build_questions(chunks: list[Document], triples_by_chunk: dict[int, list[Triple]],
                    n_questions: int = N_QUESTIONS,
                    max_per_relation: int = MAX_PER_RELATION) -> list[dict]:
    """트리플에서 질문을 템플릿으로 생성하고, gold를 그 트리플의 출처 청크로
    단다. 관계 유형별 상한(max_per_relation)을 둬 흔한 유형(org:top_members
    /employees 등)이 평가셋을 독식하지 않게 한다. chunk_index 오름차순으로
    결정적으로 뽑는다 — 순수 함수, LLM 호출 없음."""
    per_relation_count: dict[str, int] = defaultdict(int)
    questions = []
    for chunk in chunks:
        idx = chunk.metadata["chunk_index"]
        for t in triples_by_chunk.get(idx, []):
            if len(questions) >= n_questions:
                return questions
            if per_relation_count[t.relation] >= max_per_relation:
                continue
            md5 = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
            questions.append({
                "question": render_question(t.relation, t.subject),
                "relation": t.relation,
                "gold": [{"chunk_index": idx, "source": "klue_re", "md5": md5,
                         "preview": chunk.page_content[:60]}],
            })
            per_relation_count[t.relation] += 1
    return questions


def hybrid_search(query: str, vectorstore, bm25, k: int) -> list[Document]:
    """graph.hybrid_search와 같은 RRF(K=60)를 독립 인덱스(vectorstore/bm25
    객체를 직접 받음)에 적용한다. graph.py의 전역 싱글턴(_vectorstore/_bm25)은
    원 코퍼스에 묶여 있어 재사용하지 않고, 여기서 같은 로직을 다시 구현한다 —
    두 코퍼스의 인덱스가 한 프로세스 안에서 섞일 위험을 원천 차단한다."""
    vec_docs = vectorstore.similarity_search(query, k=k)
    kw_docs = bm25.invoke(query)
    K = 60
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in (vec_docs, kw_docs):
        for rank, doc in enumerate(docs):
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [by_key[key] for key in ranked[:k]]
