# 이윤선 이력서

이윤선
6년차 AI 엔지니어
yoon7829@gmail.com | ckc5800.github.io | github.com/ckc5800 | taepseon.tistory.com
PROFESSIONAL SUMMARY
AI Engineer | MLOps Specialist
• AI·MLOps 전문 — 프로덕션 ML 시스템 구축 및 운영 6년차
• TTS 서비스: TTFB 85% 단축 (2292ms → 334ms, 6.85배 개선)
• MLOps 인프라: 회사 최초 Kubernetes 기반 AI 플랫폼 구축
• 학술 성과: 논문 7편(제1저자), 특허 2건(제1발명자), 우수논문상
에이아이세스 (AICESS)
Manager  |  Apr 2025 – Present
TTS 프로젝트 (2025.09 ~ 현재)
PM 및 풀스택 개발 (Backend/Frontend/Infrastructure)
▸ 배경: 고객 상담 센터 실시간 음성 합성 필요 — 기존 응답 시간(2.3초)·동시 채널(12개) 제약 극복 목표
▸ 시스템 아키텍처
FastAPI 기반 WebSocket/REST API 게이트웨이, vLLM-Omni(Qwen3-TTS-1.7B) 추론 엔진, Redis 분산락 & 캐싱 레이어, React 관리자 대시보드. 문장 단위 논리 청킹, 지속적 WebSocket 스트리밍, Pre-Normalization으로 초저지연 구현.
▸ 기능 확장
ICL(In-Context Learning) 기반 음성복제 기능 구현. OpenAI TTS API 호환 인터페이스 및 기존 WebSocket 프로토콜 호환 레이어 개발로 서비스 연동 지원.
▸ 성과 (KPI)
✓ TTFB: 2292ms → 334ms (85% 단축, 6.85배 개선)
✓ 동시 처리 채널: 12 → 24개 (2배 확대)
✓ Throughput: 0.68 → 1.41 rps (2배), 최적화 후 ~1.7 rps (2.5배 향상)
✓ 단건 지연시간 15% 감소 | RTF 0.1 이하 | CER(문자 오류율) 3~5%
✓ 32개 언어 지원 단일 플랫폼 | ITN 정규화 정확도 50.1% → 70.9%
▸ 핵심 엔지니어링 (5가지 문제 해결)
① 스트리밍 오디오 팝 노이즈 제거 – Residual Buffer 설계로 PCM 얼라인먼트 문제 해결, Micro-fade 적용
② 스토리지 누수 방지 – Deep GC 추적기 구현으로 MD5 해시 캐시 파일 완벽 정리
③ 웹소켓 세션 탈취 방어 – password_changed_at 검증으로 제로-트러스트 모델 구현
④ 정규식 토크나이저 고도화 – 스마트 ITN 엔진 개발로 URL/이메일 자연스럽게 처리
⑤ CPU 병목 제거 – ThreadPoolExecutor 오버헤드 제거, O(N²) → O(1) 성능 개선
▸ 기술 스택
Python 3.12, FastAPI, WebSocket, vLLM-Omni, Redis (Pub/Sub, Distributed Lock, TTL Cache), React 18, TypeScript, CUDA Optimization, FP8 Quantization
STT 배포 (2025.07 ~ 2025.09)
• 배경: AIG 상담센터 대량 상담 전화 실시간 처리 필요 — 기존 시스템 병목으로 동시 채널 부족
• WhisperX 기반 STT 엔진 성능 개선 및 운영 고도화 (배포/서빙 담당, 팀 협업)
• 기술: Triton Inference Server, Streaming ASR Pipeline, Docker
• 성과: AIG 요구사항 해결 4건, Triton 기반 추론 환경 최적화로 동시 처리 채널 확장 및 안정성 확보
Speaker Diarization (2025.04 ~ 2025.07)
• 배경: 상담 품질 분석 위한 발화자 식별 필요 — 상담사/고객 구분 자동화 목표
• Pyannote 사전 학습 모델을 한국어 상담 데이터로 파인튜닝, 화자 분할 → STT → 분석 파이프라인 구축
• 기술: Pyannote, PyTorch, Audio Processing
인피닉 (INFINIIC)
AI Engineer/MLOps Engineer  Aug 2022 – Apr 2025
Kubernetes 기반 AI 인프라 (2024.03 ~ 2025.04)
• 배경: 20개 이상 AI 서비스 수동 배포로 배포 시간 수 시간 소요 — 자동화 및 확장성 확보 목표
• 회사 최초 Kubernetes 도입: VM/수동 배포에서 컨테이너 기반 자동화 배포 체계로 전환
• 기술: Kubernetes, Docker, Helm, GitLab CI/CD, Jenkins, ArgoCD, Prometheus/Grafana/Loki/ELK
• 성과: 20개 이상 ML 모델 운영 환경 구축, GitLab → Jenkins → ArgoCD 배포 자동화
NLP-based Text Summarization System (2024.01 ~ 2024.07)
• 배경: 사내 회의록·업무 문서 다수 생성 — 핵심 내용 자동 요약 및 업무 효율화 필요
• 사내 메신저 및 업무 문서 데이터 기반 요약 모델(BART/T5) 학습 및 성능 개선
• 기술: BART, T5, Hugging Face, PyTorch
• 성과: 사내 자체 개발 메신저에 기능 추가하여 회의록, 문서 요약에 적용
Generative Model Research (2023.01 ~ 2023.12)
• 배경: 국방 도메인 학습 데이터 부족 — 생성 모델 기반 데이터 확보 연구
• Latent Diffusion 기반 합성 데이터 생성 및 실제 모델 학습 적용 검증
• 기술: Latent Diffusion, GAN, PyTorch
• 성과: 논문 2편 게재(제1저자) | 우수 논문상 수상(국방기술학회 2023.11)
3D Semantic Segmentation Research (2022.08 ~ 2022.12)
• 배경: 자율주행 환경 인식에 3D 거리/높이 정보 필수 — 2D 세그멘테이션 한계 극복 위한 센서 퓨전 연구
• 성과: 논문 2편 게재(제1저자, 한국자동차공학회) | 특허 2건 등록(제1발명자)
• 기술: Trans-Unet, LiDAR + Camera Sensor Fusion, PyTorch
이든티앤에스 (EDEN T&S)
AI Engineer  Apr 2022 – Aug 2022
OCR Dataset Auto-Generation Tool
• 배경: OCR 학습 데이터 대량 생성 필요 — 수동 주석 작업 병목 해결 목표
• Annotation Tool 설계 및 단독 개발 (이미지 로딩, 라벨링, 데이터 저장 기능)
• 기술: PyQt, Tesseract OCR, ViT-based Table Detection, RNN Text Extraction
• 성과: OCR 학습 데이터 생성 자동화, 문서 인식 성능 개선, 프로젝트 성공 성과금 수령
큐헷지 (QUHEDGE)
Intern  |  Sep 2020 – Sep 2021
금융 데이터 수집 및 전처리 파이프라인
• 배경: 금융 데이터 분석 위한 고품질 데이터 확보 필요 — 다중 소스 수집·정제 자동화 목표
• 금융 데이터 크롤링 및 Pandas/NumPy 기반 정제·검증·저장 자동화 파이프라인 구축
• 기술: Python, Pandas, NumPy, SQL
KISTI
알바  |  Oct 2017 – Jan 2018
한국어 고어 데이터 전처리
• 한국어 고어(古語) 데이터 정제 및 전처리 작업 수행
TECHNICAL SKILLS
Languages: Python (Expert), TypeScript, Bash, SQL
AI/ML: PyTorch, Hugging Face, vLLM, ONNX Runtime, Whisper
Model Serving: vLLM, Triton Inference Server, ONNX Runtime
Backend: FastAPI, gRPC, WebSocket, REST APIs, Asyncio
DevOps: Kubernetes, Docker, GitLab CI, Jenkins, ArgoCD, Harbor, Nginx
Data & Storage: Redis, PostgreSQL, MongoDB, S3, NFS
Monitoring: Prometheus, Grafana, Loki, ELK Stack, AlertManager
EDUCATION & CREDENTIALS
M.S. Computer Science  |  Inha University (2021.08)
B.S. Computer Science  |  Hannam University (2018.02)
Certifications: Linux Master (2016), Data Analytics Semi-Professional (2021)
Language: English (Intermediate, OPIc IM1)
PUBLICATIONS & PATENTS
Publications (First Author): 
• 국방 데이터 확보를 위한 생성모델 Latent Diffusion 실험  |  국방기술학회 (2023) 우수논문상
• GAN을 활용한 데이터 생성 연구 동향  |  한국항공우주학회 (2023)
• 센서 퓨전 기반의 Trans-Unet을 활용한 2차원 시맨틱 세그멘테이션의 3차원적 해석  |  한국자동차공학회 (2023)
• 자율 주행 도메인의 3차원 시맨틱 세그멘테이션을 위한 센서 퓨전 기반의 Trans-Unet  |  한국자동차공학회 (2022)
• 비정형, 정형 데이터의 이미지 학습을 활용한 시장예측  |  스마트미디어학회 (2021)
• Trend of Malware Detection Using Deep Learning  |  ACM International Conference (2018)
• Deep Learning을 활용한 악성코드 탐지 방법 동향 분석  |  한국정보기술학회 (2018)
Patents (First Inventor): 
• Sensor Fusion-based Semantic Segmentation Method  |  Reg. No. 1025382250000
• 3D Interpretation Method for Semantic Segmentation  |  Reg. No. 1025382310000
Awards: Outstanding Paper Award – Korea Defense Technology Society (Nov 2023)