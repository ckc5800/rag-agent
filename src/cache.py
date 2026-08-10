"""응답 캐시 — 같은 질문·같은 설정 조합에 다시 LLM을 부르지 않는다.

CPU 추론 기준 질문 하나에 60~150초가 걸린다(README 실측). 데모·면접
자리에서 같은 질문이 반복되는 경우가 실제로 많아, 정답을 다시 계산하지
않고 즉시 돌려주는 게 체감 차이가 크다.

**graph.ask()가 아니라 api.py의 서빙 경계에만 건다.** graph.py 안에
넣으면 ab_rewrite.py·ab_type_routing.py 같은 평가 스크립트가 같은 질문을
설정만 바꿔(MAX_REWRITES, TYPE_ROUTING 등) 반복 호출할 때 캐시가 예전
설정의 답을 그대로 돌려줘 A/B 결과가 조용히 오염된다 — 이 프로젝트가
가장 경계해온 실패 유형(지표가 조용히 거짓이 되는 것)이다. 그래서 캐시
키에 "답에 영향을 주는 설정 전부"의 지문을 넣어, 설정이 바뀌면 자동으로
다른 키가 되게 한다(수동 무효화 불필요).
"""
import hashlib
import json
import threading
from pathlib import Path

import config

CACHE_PATH = Path(config.BASE_DIR) / "data" / "answer_cache.json"

_lock = threading.Lock()
_cache: dict[str, dict] | None = None


def _config_fingerprint() -> str:
    # config.knobs()가 환경변수 손잡이 전부를 돌려준다. 예전엔 여기에 손으로
    # 적은 목록이 있었고 17개 중 10개만 들어 있어, HYDE·VERIFY_GROUNDING·
    # TEAM_ROUTING 등을 바꿔도 캐시 키가 그대로였다.
    return "|".join(f"{k}={v}" for k, v in sorted(config.knobs().items()))


def _cache_key(question: str) -> str:
    payload = f"{_config_fingerprint()}::{question.strip()}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _load() -> dict[str, dict]:
    global _cache
    if _cache is None:
        if CACHE_PATH.exists():
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _cache = {}
    return _cache


def get(question: str) -> dict | None:
    with _lock:
        return _load().get(_cache_key(question))


def put(question: str, result: dict) -> None:
    with _lock:
        cache = _load()
        cache[_cache_key(question)] = result
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def clear() -> None:
    """테스트·디버깅용. 파일도 함께 지운다."""
    global _cache
    with _lock:
        _cache = {}
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
