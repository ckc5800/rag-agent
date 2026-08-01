"""실사용 질의 트레이스 — JSONL로 이어 쓴다.

eval_set.json의 51문항은 전부 본인이 직접 만든 것이다. 실제로 누가 이
봇에게 뭘 물어보는지 아는 방법이 없었다. api.py(/ask)·cli.py 같은
**실사용 경계**에서만 기록한다 — eval 스크립트가 넣는 합성 트래픽까지
같이 쌓이면 신호가 잡음에 묻힌다(캐시를 graph.ask()가 아니라 api.py에만
건 것과 같은 이유, cache.py 참고).

로그 하나가 앞으로 eval_set.json 후보 하나다: 실패했거나(거부·낮은 신뢰)
반복되는 질문을 여기서 골라 새 유형·앵커를 붙이면 된다.
"""
import json
import time
from pathlib import Path

import config

TRACE_PATH = Path(config.BASE_DIR) / "data" / "query_trace.jsonl"


def log(question: str, result: dict, elapsed_sec: float, **extra) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": question,
        "answer": result.get("answer", "")[:500],
        "sources": result.get("sources", []),
        "rewrites": result.get("rewrites"),
        "elapsed_sec": round(elapsed_sec, 1),
        **extra,
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
