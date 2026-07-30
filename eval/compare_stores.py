"""FAISS vs Qdrant — 무엇이 같고 무엇이 다른가.

같은 임베딩·같은 exact 검색이므로 **순수 의미 검색 결과는 같아야 한다.**
그걸 먼저 확인하고(같지 않으면 어딘가 틀린 것), 실제 차이인
**메타데이터 필터**를 비교한다.

    python eval/compare_stores.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import vectorstore as vs  # noqa: E402

QUERIES = [
    "TTFB를 얼마나 개선했나요?",
    "이윤선이 근무한 회사들",
    "화자 분할 모델",
]
FILTER_CASE = ("이윤선이 근무한 회사들", "resume.md")


def top_ids(docs) -> list[str]:
    """비교용 — 앞 40자로 청크를 식별."""
    return [d.page_content[:40].replace("\n", " ") for d in docs]


def main():
    results = {}
    for kind in ("faiss", "qdrant"):
        config.VECTOR_STORE = kind
        t0 = time.perf_counter()
        store = vs.load(kind)
        load_sec = time.perf_counter() - t0

        per_query, t0 = {}, time.perf_counter()
        for q in QUERIES:
            per_query[q] = top_ids(vs.search(store, q, config.TOP_K, kind=kind))
        search_sec = (time.perf_counter() - t0) / len(QUERIES)

        q, src = FILTER_CASE
        filtered = vs.search(store, q, config.TOP_K, source=src, kind=kind)
        results[kind] = {
            "load_sec": load_sec, "search_sec": search_sec,
            "per_query": per_query, "filtered": filtered,
        }
        print(f"[{kind}] 로드 {load_sec:.2f}s · 검색 평균 {search_sec:.2f}s")

    print("\n===== 1. 순수 의미 검색: 결과가 같은가 =====")
    same = True
    for q in QUERIES:
        a, b = results["faiss"]["per_query"][q], results["qdrant"]["per_query"][q]
        ok = a == b
        same &= ok
        print(f"  [{'동일' if ok else '다름'}] {q}")
        if not ok:
            for i, (x, y) in enumerate(zip(a, b), 1):
                if x != y:
                    print(f"      {i}위  faiss: {x}\n           qdrant: {y}")
    print(f"  → 전체 {'일치' if same else '불일치'}"
          " (같은 임베딩·exact 검색이라 같아야 정상)")

    print("\n===== 2. 메타데이터 필터: 여기가 진짜 차이 =====")
    q, src = FILTER_CASE
    print(f'  질의: "{q}"  /  조건: source == {src}  /  요청 k={config.TOP_K}\n')
    for kind in ("faiss", "qdrant"):
        docs = results[kind]["filtered"]
        wrong = [d for d in docs if d.metadata.get("source") != src]
        how = "사후 필터링(검색 후 파이썬)" if kind == "faiss" else "검색 단계에 필터 주입"
        print(f"  [{kind}] {how}")
        print(f"     받은 개수 {len(docs)}/{config.TOP_K}"
              f"{'  ← k를 못 채움' if len(docs) < config.TOP_K else ''}")
        print(f"     조건 위반 {len(wrong)}건")
        for d in docs[:3]:
            print(f"       · [{d.metadata.get('source')}] {d.page_content[:44].strip()}")
        print()

    print("===== 3. FAISS 사후 필터링은 '몇 배로 뽑을지'에 달려 있다 =====")
    print("  Qdrant는 필터를 검색에 넣으므로 배수라는 개념이 없다.")
    print("  FAISS는 넉넉히 뽑아 걸러내는데, 그 배수를 정할 근거가 없다.\n")

    config.VECTOR_STORE = "faiss"
    store = vs.load("faiss")
    lines = Path(config.CHUNKS_PATH).read_text(encoding="utf-8").splitlines()
    n_chunks = len(lines)
    k = config.TOP_K

    # 조건이 좁을수록(해당 문서의 청크가 적을수록) 어려워진다
    for src in ("resume.md", "publications.md", "patents.md"):
        owned = sum(1 for line in lines if f'"source": "{src}"' in line)
        print(f'  source == {src}  (전체 {n_chunks}청크 중 {owned}개)')
        for mult in (1, 2, 5, 10):
            docs = store.similarity_search(q, k=min(k * mult, n_chunks))
            hit = [d for d in docs if d.metadata.get("source") == src][:k]
            target = min(k, owned)
            mark = "OK" if len(hit) >= target else "부족"
            print(f"     {mult:>2}배({k * mult:>2}개) 뽑아 거름 → {len(hit)}/{target}  {mark}")
        print(f"     Qdrant(필터 주입)              → {min(k, owned)}/{min(k, owned)}  OK  "
              "(배수와 무관)")
        print()

    print("  → 이 규모(67청크)에선 넉넉히 뽑으면 대체로 채워진다.")
    print("     문제는 배수를 정할 근거가 없다는 것이고, 코퍼스가 커지고 조건이")
    print("     좁아질수록(예: 가격·재고 필터) 뽑은 창 안에 조건 통과 문서가")
    print("     아예 없는 상황이 생긴다. Qdrant는 그 지점에서 갈린다.")


if __name__ == "__main__":
    main()
