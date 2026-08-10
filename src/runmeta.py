"""평가 실행 환경 지문 — 결과 파일에 붙여 비교 가능성을 보장한다.

이 프로젝트에서 실제로 겪은 문제: README에 90%로 기록된 수치와 다른 머신에서
잰 74%가 비교 가능한지 판단할 근거가 결과 파일에 없었다. 모델·Ollama 버전·
인덱스·하드웨어·주요 파라미터가 전부 결과 밖에 있었기 때문이다.

results*.json 에 이 딕셔너리를 함께 저장하면, 나중에 두 결과를 놓고
"이건 비교할 수 있는 수치인가"를 파일만 보고 판단할 수 있다.
"""
import json
import os
import platform
import urllib.error
import urllib.request
from pathlib import Path

import config


def _ollama_version() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"
    try:
        with urllib.request.urlopen(f"{host}/api/version", timeout=5) as r:
            return json.load(r).get("version", "?")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return "unreachable"


def _gpu() -> str:
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "cpu/unknown"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "?"
    except Exception:                                        # noqa: BLE001
        return "?"


def run_metadata() -> dict:
    """결과 파일에 함께 저장할 환경·파라미터 지문."""
    manifest = {}
    mp = Path(config.INDEX_MANIFEST)
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    return {
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "ollama_version": _ollama_version(),
        "vector_store": config.VECTOR_STORE,
        "index_chunks_md5": manifest.get("chunks_md5"),
        "n_chunks": manifest.get("n_chunks"),
        # 손잡이 전부. 예전엔 5개만 적어 둬서 결과 파일만 보고는
        # HyDE 실행인지 아닌지 구분할 수 없었다 — 지문의 존재 이유가
        # 비교 가능성인데 그게 반만 됐다.
        "params": {**config.knobs(),
                   "CHUNK_SIZE": config.CHUNK_SIZE,
                   "CHUNK_OVERLAP": config.CHUNK_OVERLAP},
        "host": {"os": platform.platform(), "python": platform.python_version(),
                 "gpu": _gpu()},
    }
