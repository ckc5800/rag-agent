"""기동 전 사전 점검 — 실패를 스택트레이스가 아니라 문장으로 알린다.

지금까지 `src/` 전체에 예외 처리가 두 곳뿐이었다. Ollama가 안 떠 있거나,
인덱스를 아직 안 만들었거나, 모델을 안 받았으면 raw 스택트레이스가 나온다.
셋 다 흔한 상황이고 해결책이 한 줄인데 메시지가 그걸 안 알려준다.

    from preflight import check_all
    check_all()          # 문제가 있으면 SystemExit(사람이 읽을 수 있는 안내)

Ollama 확인은 표준 라이브러리(urllib)만 쓴다 — 점검 때문에 의존성을 늘리지
않는다. 모델 목록까지 받아 오므로 "서버는 떴는데 모델을 안 받은" 흔한
상태도 구분해서 알려준다.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import config

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"       # 'host:port' 형식도 받는다


def check_index() -> list[str]:
    """인덱스·청크 파일 존재와 상호 일관성. 문제 메시지 목록을 돌려준다."""
    import graph

    problems = []
    store_dir = Path(config.DB_DIR if config.VECTOR_STORE == "faiss"
                     else config.QDRANT_PATH)
    if not store_dir.exists():
        problems.append(
            f"{config.VECTOR_STORE} 인덱스가 없습니다: {store_dir}\n"
            "    → python src/ingest.py")
        return problems
    try:
        graph.check_index_consistency()
    except graph.IndexError_ as e:
        problems.append(str(e))
    return problems


def check_ollama() -> list[str]:
    """서버 응답과 필요한 모델 보유 여부."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            tags = json.load(r)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        return [f"Ollama에 연결할 수 없습니다 ({OLLAMA_HOST}): "
                f"{type(e).__name__}\n"
                "    → Ollama를 실행하세요. 호스트가 다르면 OLLAMA_HOST 설정."]

    have = {m["name"] for m in tags.get("models", [])}
    # 태그 없이 적힌 모델명은 :latest 로 저장돼 있다
    have |= {n.split(":")[0] for n in have}

    problems = []
    for model in (config.LLM_MODEL, config.EMBED_MODEL):
        if model not in have and model.split(":")[0] not in have:
            problems.append(f"모델이 없습니다: {model}\n"
                            f"    → ollama pull {model}")
    return problems


def check_all(strict: bool = True) -> list[str]:
    """전체 점검. strict면 문제가 있을 때 SystemExit."""
    problems = check_index() + check_ollama()
    if problems and strict:
        raise SystemExit("사전 점검 실패:\n\n" +
                         "\n\n".join(f"  - {p}" for p in problems))
    return problems


if __name__ == "__main__":
    found = check_all(strict=False)
    if found:
        print("사전 점검 실패:\n")
        for p in found:
            print(f"  - {p}\n")
        raise SystemExit(1)
    print(f"사전 점검 통과 — 인덱스 OK, Ollama OK "
          f"({config.LLM_MODEL} / {config.EMBED_MODEL})")
