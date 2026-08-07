"""loaders.py 단위 테스트 — data/docs/의 실제 PDF·DOCX로 검증한다(모킹
없이). 이 프로젝트는 파서를 실물 문서로만 검증한다는 원칙(README "HWP는
실물이 없어 안 만듦")과 같은 이유로, 합성 PDF/DOCX 대신 실제 코퍼스 파일을
쓴다 — 실제 문서에서 안 깨지는 것 자체가 검증이다.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loaders import LOADERS, _normalize_whitespace, load_docx, load_hwpx, load_pdf  # noqa: E402

DOCS_DIR = ROOT / "data" / "docs"
PDF_PATH = DOCS_DIR / "segmentation-paper.pdf"
# DOCX는 코퍼스(data/docs/)가 아니라 테스트 전용 픽스처에 둔다 — resume.md와
# 거의 같은 내용이라 코퍼스에 넣으면 근중복 청크가 top-N 컨텍스트 슬롯을
# 잡아먹는 회귀가 실측됐다(README "비정형 문서 확장" 절). 파서 자체의 정확성은
# 여기서 별도로 검증한다.
DOCX_PATH = ROOT / "tests" / "fixtures" / "resume-original.docx"
# HWPX도 픽스처 전용 — 개인 소유 실물이 없어 국토교통부가 공개 배포한
# 보도자료로 검증한다(내용은 코퍼스와 무관해 애초에 넣을 이유가 없다).
HWPX_PATH = ROOT / "tests" / "fixtures" / "moltm-railway-day-notice.hwpx"


def test_normalize_whitespace_collapses_and_trims():
    assert _normalize_whitespace("a  \nb\n\n\n\nc  \n") == "a\nb\n\nc"


@pytest.mark.skipif(not PDF_PATH.exists(), reason="실물 PDF 없음")
def test_load_pdf_extracts_real_content():
    doc = load_pdf(PDF_PATH)
    assert doc.metadata["source"] == "segmentation-paper.pdf"
    assert doc.metadata["pages"] > 0
    assert len(doc.page_content) > 500
    # 논문 핵심 키워드가 실제로 뽑혔는지 — 빈 페이지 껍데기만 남는 회귀 방지
    assert "세그멘테이션" in doc.page_content


@pytest.mark.skipif(not DOCX_PATH.exists(), reason="실물 DOCX 없음")
def test_load_docx_extracts_real_content_and_agrees_with_resume_md():
    doc = load_docx(DOCX_PATH)
    assert doc.metadata["source"] == "resume-original.docx"
    assert len(doc.page_content) > 500
    # resume.md의 이미 검증된 핵심 수치와 교차 검증 — DOCX 파서가 같은
    # 사실을 뽑아내는지 확인한다(파서 신뢰도의 실질적인 증거).
    assert "2292" in doc.page_content or "0.33초" in doc.page_content
    assert "MiCo AI" in doc.page_content


@pytest.mark.skipif(not HWPX_PATH.exists(), reason="실물 HWPX 없음")
def test_load_hwpx_extracts_real_government_notice():
    doc = load_hwpx(HWPX_PATH)
    assert doc.metadata["source"] == "moltm-railway-day-notice.hwpx"
    assert len(doc.page_content) > 500
    # 국토교통부 실제 보도자료(2026 철도의 날) 핵심 내용이 뽑혔는지 확인
    assert "국토교통부" in doc.page_content
    assert "철도의 날" in doc.page_content


def test_loaders_registry_covers_pdf_docx_hwpx():
    assert set(LOADERS) == {".pdf", ".docx", ".hwpx"}
