"""可选的真实训练步计时；不侵入默认热路径，不产生日志时零同步开销。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


@contextmanager
def train_step_timer(trainer):
    path = os.environ.get("TT_BIG_LEAF_STEP_TRACE")
    if not path:
        yield
        return

    import torch
    torch.cuda.synchronize()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    iteration = int(getattr(trainer.state, "iteration", -1)) + 1
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        if rank == 0:
            _append(path, {"iteration": iteration, "success": False})
        raise
    else:
        torch.cuda.synchronize()
        if rank == 0:
            _append(path, {
                "iteration": iteration,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "success": True,
            })


def _append(path, record):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as writer:
        writer.write(json.dumps(record) + "\n")
