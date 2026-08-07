"""klue_re.py 결정적 로직 단위 테스트 — HF 다운로드 없이 CI에서 돈다.
build_chunks/build_questions는 순수 함수라 `datasets.Dataset` 대신 가벼운
가짜 rows로 검증한다(인터페이스: iterable of dict + .features['label'].names).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from klue_re import TEMPLATES, build_chunks, build_questions  # noqa: E402

LABELS = ["no_relation", "org:founded_by", "per:employee_of", "org:place_of_headquarters"]


def _row(sentence, subject, relation_name, obj):
    return {
        "sentence": sentence,
        "subject_entity": {"word": subject, "start_idx": 0, "end_idx": 0, "type": "ORG"},
        "object_entity": {"word": obj, "start_idx": 0, "end_idx": 0, "type": "PER"},
        "label": LABELS.index(relation_name),
    }


class FakeRows:
    """datasets.Dataset의 최소 인터페이스만 흉내낸다."""

    def __init__(self, rows):
        self._rows = rows
        self.features = {"label": SimpleNamespace(names=LABELS)}

    def __iter__(self):
        return iter(self._rows)


def test_all_klue_re_labels_have_templates():
    # 실제 KLUE-RE 라벨 29종(2026-08 확인) 전부가 매핑돼 있어야 한다 —
    # 하나라도 빠지면 그 관계의 트리플은 build_chunks에서 조용히 버려진다.
    real_labels = [
        "org:dissolved", "org:founded", "org:place_of_headquarters",
        "org:alternate_names", "org:member_of", "org:members",
        "org:political/religious_affiliation", "org:product", "org:founded_by",
        "org:top_members/employees", "org:number_of_employees/members",
        "per:date_of_birth", "per:date_of_death", "per:place_of_birth",
        "per:place_of_death", "per:place_of_residence", "per:origin",
        "per:employee_of", "per:schools_attended", "per:alternate_names",
        "per:parents", "per:children", "per:siblings", "per:spouse",
        "per:other_family", "per:colleagues", "per:product", "per:religion",
        "per:title",
    ]
    missing = [r for r in real_labels if r not in TEMPLATES]
    assert not missing, f"템플릿 누락: {missing}"


def test_build_chunks_dedupes_same_sentence_merges_triples():
    rows = FakeRows([
        _row("A는 B에서 일했다.", "A", "per:employee_of", "B"),
        _row("A는 B에서 일했다.", "A", "org:founded_by", "C"),  # 같은 문장, 다른 개체쌍
        _row("D는 E다.", "D", "no_relation", "E"),
    ])
    chunks, triples_by_chunk = build_chunks(rows)
    assert len(chunks) == 2  # 문장 고유 개수
    assert len(triples_by_chunk[0]) == 2  # 같은 문장의 트리플 두 개가 합쳐짐
    assert triples_by_chunk[1] == []  # no_relation은 트리플 없음, 청크는 유지


def test_build_chunks_respects_n_chunks_limit():
    rows = FakeRows([_row(f"문장{i}", "A", "per:employee_of", "B") for i in range(10)])
    chunks, _ = build_chunks(rows, n_chunks=3)
    assert len(chunks) == 3


def test_build_questions_generates_templated_gold():
    rows = FakeRows([_row("이윤선은 회사에서 일했다.", "이윤선", "per:employee_of", "회사")])
    chunks, triples_by_chunk = build_chunks(rows)
    qs = build_questions(chunks, triples_by_chunk)
    assert len(qs) == 1
    # "이윤선"은 받침(ㄴ)이 있어 "은"이 붙어야 한다("는"이면 비문)
    assert qs[0]["question"] == "이윤선은 어느 조직(회사)에 소속되어 있나요?"
    assert qs[0]["gold"][0]["chunk_index"] == 0


def test_render_question_josa_agreement():
    from klue_re import render_question

    assert render_question("per:employee_of", "이윤선").startswith("이윤선은")  # 받침 O
    assert render_question("per:employee_of", "김철수").startswith("김철수는")  # 받침 X
    assert render_question("org:product", "삼성전자").startswith("삼성전자가")  # 받침 X
    assert render_question("org:product", "카카오뱅크").startswith("카카오뱅크가")  # ㅋ 받침 X
    assert render_question("org:founded_by", "인피닉").startswith("인피닉을")  # 받침 O(ㄱ)


def test_build_questions_caps_per_relation():
    rows = FakeRows([
        _row(f"문장{i}", f"주체{i}", "per:employee_of", "회사") for i in range(10)
    ])
    chunks, triples_by_chunk = build_chunks(rows)
    qs = build_questions(chunks, triples_by_chunk, max_per_relation=3)
    assert len(qs) == 3


def test_build_questions_respects_total_limit():
    rows = FakeRows([
        _row(f"문장{i}", f"주체{i}", "per:employee_of", "회사") for i in range(5)
    ] + [
        _row(f"문장b{i}", f"주체b{i}", "org:founded_by", "회사") for i in range(5)
    ])
    chunks, triples_by_chunk = build_chunks(rows)
    qs = build_questions(chunks, triples_by_chunk, n_questions=4, max_per_relation=10)
    assert len(qs) == 4
