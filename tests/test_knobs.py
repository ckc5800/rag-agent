"""손잡이 지문이 다시 드리프트하지 않는지 고정한다.

cache.py와 runmeta.py가 각자 손으로 적은 목록을 들고 있다가 둘 다 새 손잡이를
놓쳤다(17개 중 10개·5개). 그래서 HYDE=1로 띄우면 HYDE=0 시절 캐시 답이 그대로
나왔다 — 캐시 모듈이 막겠다고 선언한 사고다. config._env 통로 하나로 모았으니,
그 통로를 우회하는 순간 실패하게 만든다.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cache  # noqa: E402
import config  # noqa: E402


def test_every_env_knob_goes_through_the_helper():
    src = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
    # _env 정의 본문 한 줄만 예외
    strays = [ln.strip() for ln in src.splitlines()
              if "os.environ.get" in ln and "def _env" not in ln
              and "cast(os.environ.get" not in ln]
    assert not strays, f"_env를 우회한 손잡이: {strays}"


def test_knobs_covers_all_uppercase_env_settings():
    src = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
    declared = set(re.findall(r'_env\("(\w+)"', src))
    assert declared == set(config.knobs()), (
        f"선언 {declared - set(config.knobs())} / "
        f"등록 {set(config.knobs()) - declared}")
    assert len(declared) >= 17


def test_cache_key_changes_when_any_knob_changes(monkeypatch):
    """어떤 손잡이를 바꿔도 캐시 키가 달라져야 한다 — 이게 깨진 게 원래 버그."""
    base = cache._cache_key("질문")
    for name, value in config.knobs().items():
        flipped = (not value) if isinstance(value, bool) else \
                  (value + 1) if isinstance(value, int) else value + "_x"
        monkeypatch.setitem(config._KNOBS, name, flipped)
        assert cache._cache_key("질문") != base, f"{name}이 캐시 키에 없다"
        monkeypatch.undo()


def test_runmeta_params_include_every_knob():
    import runmeta

    params = runmeta.run_metadata()["params"]
    missing = set(config.knobs()) - set(params)
    assert not missing, f"실행 지문에 빠진 손잡이: {missing}"
