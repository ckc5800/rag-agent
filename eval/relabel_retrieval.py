"""gold 라벨 재부착 — 청킹이 바뀌면 md5 라벨이 무효화되므로.

retrieval_set.json의 gold는 청크 내용 md5로 고정돼 있다. 정제 규칙이나
청킹 파라미터를 바꾸면 내용이 달라져 md5가 전부 어긋난다(의도한 설계다 —
옛 라벨로 새 인덱스를 채점하는 사고를 구조적으로 막는다).

이 스크립트는 git에 남아 있는 이전 chunks.jsonl에서 옛 gold 본문을 꺼내,
새 청크 중 가장 잘 겹치는 것을 찾아 라벨을 다시 붙인다. 다만 자동 매칭이
조용히 틀리면 평가가 통째로 거짓이 되므로, **매칭 점수를 전부 출력하고
임계 미만이면 실패**시킨다. 사람이 보고 승인하는 단계를 없애지 않는다.

    git show HEAD:data/chunks.jsonl > /tmp/old.jsonl
    python eval/relabel_retrieval.py --old /tmp/old.jsonl          # 미리보기
    python eval/relabel_retrieval.py --old /tmp/old.jsonl --write  # 실제 반영
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402

RETRIEVAL_SET = Path(__file__).parent / "retrieval_set.json"
MIN_SCORE = 0.35   # 이보다 낮으면 사람이 직접 봐야 한다

_NORM = re.compile(r"[^0-9A-Za-z가-힣]+")


def grams(text: str, n: int = 5) -> set[str]:
    """정규화한 문자 n-gram 집합. 공백·기호 차이에 둔감하게 비교하려고."""
    s = _NORM.sub("", text)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def containment(old: str, new: str) -> float:
    """옛 청크 내용이 새 청크에 얼마나 담겨 있는지 (0~1)."""
    a, b = grams(old), grams(new)
    return len(a & b) / len(a) if a else 0.0


def load_chunks(path: Path) -> list[dict]:
    # utf-8-sig: 셸 리다이렉트로 뽑은 파일에 BOM이 붙는 경우가 있다
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="gold 라벨 재부착")
    ap.add_argument("--old", required=True, help="이전 chunks.jsonl 경로")
    ap.add_argument("--write", action="store_true", help="retrieval_set.json에 실제 반영")
    args = ap.parse_args()

    old_chunks = load_chunks(Path(args.old))
    new_chunks = load_chunks(Path(config.CHUNKS_PATH))
    by_md5 = {hashlib.md5(c["page_content"].encode("utf-8")).hexdigest(): c
              for c in old_chunks}
    print(f"이전 {len(old_chunks)}청크 → 현재 {len(new_chunks)}청크\n")

    cases = json.loads(RETRIEVAL_SET.read_text(encoding="utf-8"))
    low = []

    for case in cases:
        print(f"[{case['question']}]")
        remapped = []
        for g in case["gold"]:
            old = by_md5.get(g["md5"])
            if old is None:
                print(f"   !! 옛 청크를 못 찾음 (md5 {g['md5'][:12]}) — 수동 확인 필요")
                low.append((case["question"], g["md5"], 0.0))
                continue

            scored = sorted(
                ((containment(old["page_content"], c["page_content"]), i, c)
                 for i, c in enumerate(new_chunks)),
                key=lambda t: t[0], reverse=True)
            score, idx, best = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0

            flag = "  " if score >= MIN_SCORE else "!!"
            print(f"   {flag} {score:.2f} (2위 {runner:.2f}) "
                  f"{old['metadata']['source']} → 청크 #{idx} "
                  f"{best['page_content'][:45].strip()!r}")
            if score < MIN_SCORE:
                low.append((case["question"], g["md5"], score))
                continue

            remapped.append({
                "chunk_index": idx,
                "source": best["metadata"]["source"],
                "md5": hashlib.md5(best["page_content"].encode("utf-8")).hexdigest(),
                "preview": best["page_content"][:70].replace("\n", " "),
            })

        # 서로 다른 옛 gold가 같은 새 청크로 합쳐질 수 있다 (청크 병합) → 중복 제거
        seen, uniq = set(), []
        for r in remapped:
            if r["md5"] not in seen:
                seen.add(r["md5"])
                uniq.append(r)
        if len(uniq) != len(remapped):
            print(f"      (옛 gold {len(remapped)}개가 새 청크 {len(uniq)}개로 병합됨)")
        case["gold"] = uniq
        print()

    if low:
        print(f"[FAIL] 신뢰할 수 없는 매칭 {len(low)}건 — 반영하지 않는다:")
        for q, md5, s in low:
            print(f"   {s:.2f} {md5[:12]} {q}")
        return 1

    if args.write:
        RETRIEVAL_SET.write_text(
            json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"반영 완료: {RETRIEVAL_SET}")
    else:
        print("미리보기만 했다. 실제 반영하려면 --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
