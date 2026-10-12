"""Provenance：Provider 侧一等能力。

记录三类事实，供功能/性能分析与事后审计：
- 选择期（select/gate）：每个 (semantic, slot) 选了谁、谁被拒绝及原因、
  各候选的门禁耗时（收益证据的采集位）；
- 运行期（call）：每次调用的实际 impl、参数签名（shape/flags，
  不记数据本体）、耗时、是否回退及回退到谁（fallback_actual）；
- 管理面（switch）：启用状态变更（谁、何时、改成什么）。

内存中保留全量 RECORDS，可导出 JSONL。
"""
import json
import time

RECORDS = []
_seq = 0


def record(event, **fields):
    global _seq
    _seq += 1
    RECORDS.append({"seq": _seq, "ts": round(time.time(), 6),
                    "event": event, **fields})


def export_jsonl(path):
    with open(path, "w") as f:
        for r in RECORDS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
