"""청크 크기를 바꿔가며 검색 성능을 잰다.

지금까지 800자는 **튜닝한 값이 아니라 합리적인 기본값**이었다. 실제로
스윕한 적이 없어서, 더 잘게/크게 자르면 어떻게 되는지 확인한다.

문제: gold 라벨이 청크 내용 md5라 크기를 바꾸면 전부 무효화된다.
해결: 크기와 무관한 **정답 앵커 문자열**로 gold를 정의한다. 어떤 크기로
      잘라도 "그 문자열을 담은 청크"가 정답이므로 라벨링이 필요 없다.
      앵커는 각각 현재 인덱스에서 1~4청크에만 등장하도록 좁게 골랐다.

원래는 아래 ANCHORS 10문항("하나라도 포함하면 gold")만 쟀다. 그런데
(1) 10문항은 이미 두 번 잘못된 결론을 만들었고, (2) "하나라도 회수"
정의는 집계·열거처럼 근거 전수가 필요한 질문의 실패를 숨긴다 —
eval_coverage.py가 그 이유로 존재한다. 그래서 같은 스윕에서
**eval_set의 gold_anchor_sets(정본 의미론 — 경로별 전수 충족, 65문항)**
도 함께 잰다. 10문항 표는 예전 결과와의 연속성을 위해 유지한다.

청킹이 앵커 문자열 자체를 반토막 내면 그 앵커는 어떤 검색으로도 충족
불가능해진다 — 이건 검색 실패가 아니라 **청킹의 근거 손실**이므로
따로 센다(evidence_loss). 이 구분이 없으면 작은 청크의 하락을 검색
탓으로 오독한다.

    python eval/sweep_chunk_size.py                    # 기본 스윕
    python eval/sweep_chunk_size.py --sizes 400 800    # 일부만
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from eval_coverage import anchor_chunks, score  # noqa: E402
from runmeta import run_metadata  # noqa: E402

# 질문 → 정답이 반드시 담겨 있어야 하는 문자열(하나라도 포함하면 gold)
ANCHORS = {
    "TTS 프로젝트에서 TTFB를 얼마나 개선했나요?": ["2292"],
    "스트리밍 오디오의 팝 노이즈 문제를 어떻게 해결했나요?": ["Residual Buffer", "팝 노이즈"],
    "이윤선의 제1저자 논문은 몇 편인가요?": ["제1저자"],
    "등록된 특허 번호를 알려주세요.": ["2538225"],
    "Kubernetes 인프라 구축에 어떤 CI/CD 도구를 사용했나요?": ["ArgoCD"],
    "우수 논문상을 받은 논문 제목은?": ["우수논문상", "우수 논문상"],
    "TTS 시스템의 동시 처리 채널은 몇 개로 확장했나요?": ["12 → 24"],
    "화자 분할에는 어떤 모델을 사용했나요?": ["yannote"],
    "웹소켓 세션 탈취 문제는 어떻게 방어했나요?": ["password_changed_at", "세션 탈취"],
    "이윤선이 근무한 회사들을 알려주세요.": ["Experience Overview"],
}
KS = [1, 3, 6]
KS_COV = [1, 3, 5, 6]     # 5 = GENERATE_TOP_N (생성이 실제로 받는 범위)

EVAL_SET = Path(__file__).parent / "eval_set.json"


def load_coverage_cases() -> list[dict]:
    """eval_set에서 gold가 정의된 문항(거부 제외)과 앵커 경로를 뽑는다."""
    cases = []
    for c in json.loads(EVAL_SET.read_text(encoding="utf-8")):
        sets = c.get("gold_anchor_sets") or (
            [[a] for a in c["gold_anchors"]] if c.get("gold_anchors") else None)
        if sets:
            cases.append({"question": c["question"],
                          "type": c.get("type", "fact"), "sets": sets})
    return cases


def coverage_eval(cov_cases: list[dict], chunk_texts: list[str],
                  graph_mod) -> dict:
    """이 청킹에서의 경로 기반 커버리지 (eval_coverage와 같은 판정).

    앵커→청크 매핑은 크기마다 다시 계산한다(앵커는 크기 무관, 매핑은
    크기 종속). 어떤 앵커가 0청크에 매핑되면 그 앵커를 요구하는 경로는
    충족 불가 — 그 문항을 evidence_loss로 센다(경로가 하나라도 살아
    있으면 손실이 아니다).
    """
    idx_of = {t: i for i, t in enumerate(chunk_texts)}
    sat = {k: 0 for k in KS_COV}
    cov = {k: 0.0 for k in KS_COV}
    evidence_loss = []
    for c in cov_cases:
        paths = [[anchor_chunks(a, chunk_texts) for a in path]
                 for path in c["sets"]]
        if not any(all(cand for cand in path) for path in paths):
            evidence_loss.append(c["question"])
        docs = graph_mod.hybrid_search(c["question"])
        got_all = [idx_of.get(d.page_content) for d in docs]
        for k in KS_COV:
            got = {i for i in got_all[:k] if i is not None}
            s, b = score(paths, got)
            sat[k] += s
            cov[k] += b
    n = len(cov_cases)
    return {
        **{f"satisfied@{k}": round(sat[k] / n * 100) for k in KS_COV},
        **{f"best_cov@{k}": round(cov[k] / n * 100) for k in KS_COV},
        "evidence_loss": evidence_loss,
    }


def build_and_eval(size: int, overlap: int, cov_cases: list[dict]) -> dict:
    """청크 크기를 바꿔 재인제스트하고 검색만 평가한다."""
    config.CHUNK_SIZE = size
    config.CHUNK_OVERLAP = overlap

    # config를 바꾼 뒤 매번 새로 임포트해야 반영된다
    for m in ("ingest", "graph", "vectorstore"):
        sys.modules.pop(m, None)
    import ingest  # noqa: PLC0415
    import graph   # noqa: PLC0415

    # 다이어그램은 크기를 바꿔도 항상 같은 통짜 청크라, 넣으면 median/mean이
    # "산문이 이 크기에서 어떤가"가 아니라 다이어그램 유무로 흔들린다.
    # recall 비교에는 넣되(검색 대상이어야 하므로) 길이 통계에서는 뺀다.
    #
    # 청킹은 ingest.build_chunks 하나만 쓴다. 예전엔 여기서 스플리터를 직접
    # 돌려, 통짜 청크 배치와 위치 메타데이터를 빠뜨린 **프로덕션과 다른
    # 코퍼스**를 재고 있었다.
    docs, diagrams = ingest.load_documents()
    chunks, _ = ingest.build_chunks(docs, diagrams, size, overlap)
    prose_chunks = [c for c in chunks if not c.metadata.get("kind")]

    import vectorstore as vs  # noqa: PLC0415
    vs.build(chunks)

    # BM25도 같은 청크로 다시 만들어야 하므로 chunks.jsonl 갱신
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"page_content": c.page_content,
                                "metadata": c.metadata}, ensure_ascii=False) + "\n")

    # 매니페스트도 같이 갱신 — check_index_consistency 가드 도입 이후 이
    # 스크립트는 첫 스텝의 hybrid_search에서 "인덱스와 chunks.jsonl이
    # 어긋났다"로 죽는 상태였다(가드가 지키려던 바로 그 스크립트가 가드에
    # 막힘). 스윕 산출물도 인덱스와 한 쌍이므로 지문을 남기는 게 맞다.
    Path(config.INDEX_MANIFEST).write_text(json.dumps({
        "chunks_md5": ingest.chunk_fingerprint(config.CHUNKS_PATH),
        "n_chunks": len(chunks),
        "vector_store": config.VECTOR_STORE,
        "embed_model": config.EMBED_MODEL,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "min_chunk_chars": config.MIN_CHUNK_CHARS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    graph._vectorstore = None      # 캐시 무효화
    graph._bm25 = None

    lengths = sorted(len(c.page_content) for c in prose_chunks)  # 산문만
    hits, ranks = {k: 0 for k in KS}, []
    for q, anchors in ANCHORS.items():
        docs_out = graph.hybrid_search(q)
        rank = next((i + 1 for i, d in enumerate(docs_out)
                     if any(a in d.page_content for a in anchors)), None)
        ranks.append(rank)
        for k in KS:
            hits[k] += rank is not None and rank <= k

    n = len(ANCHORS)

    # 무작위 기준선 — 청크가 커지면 개수가 줄어 우연히 맞을 확률이 올라간다.
    # 이걸 같이 보지 않으면 "큰 청크가 유리하다"는 착시에 빠진다.
    N = len(chunks)
    gold_counts = [sum(1 for c in chunks
                       if any(a in c.page_content for a in anc))
                   for anc in ANCHORS.values()]
    base = {}
    for k in KS:
        tot = 0.0
        for g in gold_counts:
            miss = 1.0
            for j in range(k):
                miss *= max(0.0, N - g - j) / (N - j)
            tot += 1 - miss
        base[f"recall@{k}"] = round(tot / n * 100)

    return {
        "size": size, "chunks": N,
        "median": lengths[len(lengths) // 2], "mean": sum(lengths) // len(lengths),
        **{f"recall@{k}": round(hits[k] / n * 100) for k in KS},
        "baseline": base,
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "misses": [q[:22] for q, r in zip(ANCHORS, ranks) if r is None or r > 3],
        "coverage": coverage_eval(
            cov_cases, [c.page_content for c in chunks], graph),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[300, 500, 800, 1200, 1600])
    args = ap.parse_args()

    # 이 스크립트는 인덱스와 data/chunks.jsonl(추적 중인 파일)을 매 스텝
    # 덮어쓴다. 끝까지 못 돌면 저장소가 마지막 스윕 크기 상태로 남으므로
    # 시작 시점에 알린다 — 복구 안내가 마지막 print에만 있으면 늦다.
    print(f"주의: data/chunks.jsonl과 {config.VECTOR_STORE} 인덱스를 덮어쓴다. "
          "중간에 멈추면 `python src/ingest.py`로 복구할 것.\n")

    cov_cases = load_coverage_cases()
    rows = []
    for size in args.sizes:
        overlap = max(20, size // 8)          # 원래 비율(800:100)을 유지
        print(f"청크 {size}자(overlap {overlap}) 구축·평가 중...", flush=True)
        rows.append(build_and_eval(size, overlap, cov_cases))

    print("\n===== 청크 크기 스윕 (검색 단독, 앵커 기반 gold) =====")
    print("  괄호 안은 무작위 기준선 — 청크가 커지면 개수가 줄어 기준선이 올라간다.")
    print("  크기  청크수  중앙값 |   recall@1      recall@3      recall@6    MRR")
    for r in rows:
        b = r["baseline"]
        print("  {:>4}  {:>5}  {:>5} | {:>3}% ({:>2}%)  {:>3}% ({:>2}%)  {:>3}% ({:>2}%)  {:.3f}".format(
            r["size"], r["chunks"], r["median"],
            r["recall@1"], b["recall@1"], r["recall@3"], b["recall@3"],
            r["recall@6"], b["recall@6"], r["mrr"]))

    print("\n  top-3 밖으로 밀린 질문 (generate가 못 받는 것):")
    for r in rows:
        print("   {:>4}자: {}".format(r["size"], ", ".join(r["misses"]) or "없음"))

    print(f"\n===== 경로 기반 커버리지 (정본 의미론, {len(cov_cases)}문항) =====")
    print("  '하나라도 회수'가 아니라 어느 한 근거 경로의 **전수** 충족 여부.")
    print("  크기 | sat@1  sat@3  sat@5  sat@6 | cov@5 | 근거손실")
    for r in rows:
        c = r["coverage"]
        print("  {:>4} | {:>4}%  {:>4}%  {:>4}%  {:>4}% | {:>4}% | {}건".format(
            r["size"], c["satisfied@1"], c["satisfied@3"], c["satisfied@5"],
            c["satisfied@6"], c["best_cov@5"], len(c["evidence_loss"])))

    for r in rows:
        if r["coverage"]["evidence_loss"]:
            print(f"\n  {r['size']}자에서 근거 손실(모든 경로에 충족 불가 앵커):")
            for q in r["coverage"]["evidence_loss"]:
                print(f"    - {q[:56]}")

    out = Path(__file__).parent / "results_chunk_sweep.json"
    # 이 스크립트는 인덱스를 매 스텝 덮어쓰므로 지문의 n_chunks·md5는
    # **마지막 스텝** 값이다. 비교에 쓸 건 embed_model·host 쪽이다.
    out.write_text(json.dumps({"env": run_metadata(), "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out}")
    print("주의: 이 스크립트는 인덱스와 chunks.jsonl을 덮어쓴다. "
          "끝나면 `python src/ingest.py`로 기본 설정(800자) 복구할 것.")


if __name__ == "__main__":
    main()
