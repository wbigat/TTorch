"""witness 逐步计时的最小契约。"""
import json
from types import SimpleNamespace

import torch

from ttorch.witness.step_timer import train_step_timer


def test_step_timer_writes_successful_iteration(monkeypatch, tmp_path):
    destination = tmp_path / "steps.jsonl"
    monkeypatch.setenv("TT_BIG_LEAF_STEP_TRACE", str(destination))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    trainer = SimpleNamespace(state=SimpleNamespace(iteration=4))
    with train_step_timer(trainer):
        pass
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["iteration"] == 5
    assert rows[0]["success"] is True
    assert rows[0]["elapsed_ms"] >= 0
