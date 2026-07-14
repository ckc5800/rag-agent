# Portfolio RAG Agent

LangGraph 기반 **Corrective-RAG Q&A Agent** — 개인 포트폴리오/기술문서를 지식 베이스로,
검색 품질을 스스로 평가하고 질문을 재작성하는 self-corrective 루프를 갖춘 로컬 RAG 시스템입니다.

전부 로컬에서 동작합니다 (Ollama + ChromaDB, 외부 API 불필요).

## Architecture

```mermaid
graph LR
    Q[질문] --> R[retrieve<br/>ChromaDB Top-K]
    R --> G[grade<br/>LLM 검색 품질 평가]
    G -->|충분| GEN[generate<br/>근거 기반 답변 + 출처]
    G -->|부족| RW[rewrite<br/>질문 재작성]
    RW --> R
    GEN --> A[답변]
```

- **State Management**: LangGraph `StateGraph`로 질문/검색 질의/문서/재작성 횟수를 상태로 관리
- **Self-Correction**: 검색 결과가 부족하면 LLM이 질문을 키워드 중심으로 재작성해 재검색
- **Grounded Generation**: 문서에 없는 내용은 답변하지 않도록 프롬프트 설계 (hallucination 억제)
- **Evaluation Loop**: 키워드 기반 정답률 + 재작성률 + 지연시간을 측정하는 평가 셋 내장

## Tech Stack

| 구성 | 기술 |
|---|---|
| Agent Framework | LangGraph |
| LLM | Qwen2.5-7B (Ollama, 로컬) |
| Embedding | BGE-M3 (Ollama, 다국어) |
| Vector DB | ChromaDB |
| Serving | FastAPI |

## Quickstart

```bash
# 0. Ollama 설치 후 모델 준비
ollama pull qwen2.5:7b
ollama pull bge-m3

# 1. 의존성 설치
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. 문서 인제스트 (data/docs/*.md → ChromaDB)
python src/ingest.py

# 3-a. CLI로 질문
python src/cli.py

# 3-b. API 서버
uvicorn api:app --app-dir src --port 8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"TTS 프로젝트의 TTFB 개선 수치는?\"}"

# 4. 평가 실행
python eval/evaluate.py
```

## Evaluation

`eval/eval_set.json`의 10개 질문으로 성능을 측정합니다:

- **Answer Accuracy** — 기대 키워드가 답변에 포함되는 비율
- **Rewrite Rate** — self-correction 루프 발동 비율 (검색 품질 지표)
- **Latency** — 질문당 응답 시간

프롬프트·청킹 전략·모델을 바꿀 때마다 평가를 돌려 회귀 여부를 확인하는
피드백 루프로 사용합니다.
