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
        # 모듈 속성을 바꾼다 — A/B 스크립트가 실제로 하는 방식이다.
        # _KNOBS를 직접 건드리면 런타임 반영 여부를 못 잡는다.
        monkeypatch.setattr(config, name, flipped)
        assert cache._cache_key("질문") != base, f"{name}이 캐시 키에 없다"
        monkeypatch.undo()


def test_runmeta_params_include_every_knob():
    import runmeta

    params = runmeta.run_metadata()["params"]
    missing = set(config.knobs()) - set(params)
    assert not missing, f"실행 지문에 빠진 손잡이: {missing}"


def test_bm25_k_follows_runtime_top_k(monkeypatch):
    """TOP_K를 런타임에 바꾸면 BM25도 같은 k로 뽑아야 한다.

    예전엔 인덱스 빌드 시점에 bm25.k를 한 번만 넣어서, TOP_K를 바꾸면
    벡터는 새 k로 BM25는 옛 k로 뽑아 RRF가 비대칭이 됐다. sweep_top_k.py가
    직접 bm25.k를 갱신하는 우회 코드를 갖고 있던 이유다 — 증상 없이 순위만
    틀어지는 종류라 우회를 잊으면 조용히 틀린 수치가 나온다.
    """
    import graph
    from langchain_core.documents import Document

    class FakeStore:
        def similarity_search(self, q, k):
            return [Document(page_content=f"v{i}", metadata={}) for i in range(k)]

    class FakeBM25:
        k = 99

        def invoke(self, q):
            return [Document(page_content=f"b{i}", metadata={})
                    for i in range(self.k)]

    fake = FakeBM25()
    monkeypatch.setattr(graph, "_vectorstore", FakeStore())
    monkeypatch.setattr(graph, "_bm25", fake)
    for k in (6, 15):
        monkeypatch.setattr(config, "TOP_K", k)
        graph.hybrid_search("질의")
        assert fake.k == k, f"TOP_K={k}인데 bm25.k={fake.k}"


def test_team_workers_run_concurrently_and_keep_order(monkeypatch):
    """멀티에이전트 worker가 동시에 돌고, 순서는 planner 순서를 지키는가.

    순차 실행이면 sub-질문 3개에 지연이 3배다. 답이 바뀌지 않는 변경이라
    품질 A/B 없이 넣었지만, "정말 동시에 도는가"와 "순서가 섞이지 않는가"는
    코드로 고정해야 한다 — 둘 중 하나만 깨져도 조용히 손해다.
    """
    import time

    import team

    class SlowGraph:
        def invoke(self, state):
            time.sleep(0.15)
            return {"answer": f"A:{state['question']}", "sources": ["s.md"]}

    monkeypatch.setattr(team, "build_graph", lambda: SlowGraph())
    monkeypatch.setattr(team, "get_llm", lambda *a, **k: None)

    built = team.build_team()
    node = built.nodes["workers"]
    subs = ["q1", "q2", "q3"]

    t0 = time.perf_counter()
    out = node.invoke({"question": "원 질문", "sub_questions": subs})
    elapsed = time.perf_counter() - t0

    assert [r["question"] for r in out["sub_answers"]] == subs   # 순서 보존
    assert [r["answer"] for r in out["sub_answers"]] == [f"A:{q}" for q in subs]
    assert elapsed < 0.3, f"순차 실행으로 보인다 ({elapsed:.2f}s, 순차면 0.45s)"


def test_node_timings_accumulate_on_revisit():
    """노드별 시간이 State에 쌓이고, 재방문 노드는 누적되는가.

    리듀서가 없으면 LangGraph가 같은 키를 덮어써서 **마지막 노드 기록만**
    남는다. 그리고 retrieve는 재작성 후 다시 도므로 누적돼야 "재작성이
    발동하면 검색에 시간을 두 배 쓴다"가 보인다.
    """
    from graph import _merge_timings, timed

    assert _merge_timings({}, {"retrieve": 1.0}) == {"retrieve": 1.0}
    assert _merge_timings({"retrieve": 1.0}, {"grade": 0.5}) == {
        "retrieve": 1.0, "grade": 0.5}                     # 덮어쓰지 않는다
    assert _merge_timings({"retrieve": 1.0}, {"retrieve": 2.0}) == {
        "retrieve": 3.0}                                   # 재방문은 누적
    assert _merge_timings(None, None) == {}


def test_timed_records_node_name_and_keeps_output():
    """감싼 노드가 원래 반환값을 그대로 돌려주고 timings만 얹는가."""
    import time

    from graph import timed

    def slow_node(state):
        time.sleep(0.05)
        return {"answer": "A"}

    out = timed(slow_node)({"question": "q"})
    assert out["answer"] == "A"                            # 원래 출력 보존
    assert set(out["timings"]) == {"slow_node"}            # 함수 이름으로 기록
    assert out["timings"]["slow_node"] >= 0.05


def test_graph_nodes_are_instrumented():
    """배선이 실제로 감싸져 있는가 — 노드 함수 자체는 안 건드린다."""
    import graph

    g = graph.build_graph()
    assert {"route_strategy", "retrieve", "grade", "rewrite", "generate",
            "verify"} <= set(g.nodes)
    # 노드 함수를 직접 부르면 timings가 안 붙는다(eval 스크립트·단위 테스트가
    # 직접 호출하므로 그래야 한다)
    assert "timings" not in graph.route_strategy({"question": "q"})
