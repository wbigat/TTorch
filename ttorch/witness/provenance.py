"""统一 provenance：bridge 绑定与 provider 生效共用一份事件流。

动机（2026-10-14）：此前 bridge 侧是 `cuda_compat` 的绑定快照、provider 侧是
provider 的门禁/选择/调用事件，两份 schema 各自为政，排障要对账。统一为一个
信封 + 一个 append-only 存储，使两层能合并成一份"这次训练用了哪些替身和加速"
的报告。

统一信封（每层记录都长这样）：

  seq      全局单调序号——跨层排序，还原"先绑（bridge）后选/调（provider）"的时序
  ts       时间戳
  layer    "bridge" | "provider"——哪一层产生
  event    事件类型：bridge=bind；provider=gate/select/call/switch
  subject  作用对象：bridge=符号点分名（torch.cuda.synchronize）；
           provider="semantic/slot"（select/call）或 provider 名（gate/switch）
  outcome  结果：bridge=remap/fallback/exception/default-deny/fallback-exec；
           provider=pass/fail（gate）、selected（select）、called/fallback（call）、
           enabled/disabled（switch）
  evidence 层特有字段（bridge: target/intent/swept_modules；provider:
           probe_us/ref_us/speedup/max_abs_err/fallback_actual/rejected/...）
  pid/rank 产生该事件的**进程来源**（2026-10-21，D11）。多进程下 N 份记录格式
           相同、无法区分来源——跨 rank 对账与 fork 继承分析都因此无从下手。
           rank 读 RANK/LOCAL_RANK（缺省 None = 非分布式环境，与 rank 0 区分）。

关于"统一 vs 分离"的取舍：信封字段（seq/ts/layer/event/subject/outcome）归一，
层差异收进 evidence——两层确有真实差异（bridge 无运行期 per-call、provider 无
swept 概念），强压完全一致反而失真，故信封归一 + evidence 分层。这既满足
"合并成一份报告"，又不制造伪统一。

存储是模块级私有 `_RECORDS`，外部只能经 `record()` 写、`records()`/`report()`/
`export_jsonl()` 读——不暴露可变 list，杜绝外部 append/clear 破坏不变式。
"""
import json
import os
import time

_RECORDS = []
_seq = 0

LAYER_BRIDGE = "bridge"
LAYER_PROVIDER = "provider"
LAYER_PLATFORM = "platform"   # conformance 等 platform 层的观测（如 D7 口径画像）


def _current_rank():
    """当前进程的分布式 rank（读框架注入的 RANK/LOCAL_RANK），非分布式返回 None。

    刻意区分 None 与 0："不在分布式环境"和"是 rank 0"不是一回事，
    聚合对账时需要能分辨。
    """
    for var in ("RANK", "LOCAL_RANK"):
        v = os.getenv(var)
        if v is not None and v != "":
            try:
                return int(v)
            except ValueError:
                continue
    return None


def record(layer, event, subject, outcome, **evidence):
    """追加一条统一信封事件。layer 取 LAYER_BRIDGE / LAYER_PROVIDER。

    信封自带 pid/rank（D11）：多进程训练（torchrun 多 rank、DataLoader worker、
    fork 派生）产出 N 份格式相同的记录，若无来源字段则聚合时对不上是谁产生的。
    """
    global _seq
    _seq += 1
    _RECORDS.append({"seq": _seq, "ts": round(time.time(), 6),
                     "layer": layer, "event": event, "subject": subject,
                     "outcome": outcome, "pid": os.getpid(),
                     "rank": _current_rank(), "evidence": evidence})


def records(layer=None):
    """只读快照（tuple）；layer 给定时只取该层（保持各自相对时序）。"""
    if layer is None:
        return tuple(_RECORDS)
    return tuple(r for r in _RECORDS if r["layer"] == layer)


def default_export_path(base="provenance.jsonl"):
    """默认导出文件名：带 pid/rank 后缀，避免多 rank 进程互相覆盖（D11）。"""
    stem, _, ext = base.rpartition(".")
    rank = _current_rank()
    suffix = f".r{rank}" if rank is not None else ""
    return f"{stem}{suffix}.{os.getpid()}.{ext or 'jsonl'}"


def export_jsonl(path=None, layer=None):
    """导出 JSONL（默认全量，可按层过滤）。生成物，供 diff/审计。

    path 缺省时用 `default_export_path()`——多进程下固定文件名会被最后一个
    写完的进程覆盖，跨 rank 对账拿到的只有一份。
    """
    path = path or default_export_path()
    with open(path, "w") as f:
        for r in records(layer):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def fork_reset(parent_pid):
    """fork 派生进程调用：丢弃继承来的父进程记录，本进程从干净状态开始。

    为什么必须显式重置（D11）：fork 会连 `_RECORDS` 和 `_seq` 一起继承——
    父子记录混杂在同一份事件流里，且 `seq` 从继承值继续递增，多个 worker
    聚合时 **seq 重复、无法排序**。派生进程（DataLoader worker 等）自己的
    事件流里本来也不该出现父进程的绑定记录（那不是它产生的）。

    重置后留一条 `fork/inherited` 事件指明来源（parent_pid），保证
    "一份 JSONL = 一个进程的事件流"，且来源仍可追溯。
    """
    global _seq
    parent = parent_pid
    _RECORDS.clear()
    _seq = 0
    record(LAYER_BRIDGE, "fork", "process", "inherited",
           parent_pid=parent,
           detail="fork 派生进程：已丢弃继承的父进程记录，本进程事件流从 "
                  "seq=1 重新计数（父进程事件见其自身的 JSONL）")


def session():
    """最近一次 provider 启动快照（可复现的关键）：平台/能力/版本/注册了哪些 provider。

    §4"可复现"硬要求的落点——另一次运行凭这份快照（capability 快照 + 版本）能
    重放同一套选择。无则返回 None。
    """
    for r in reversed(_RECORDS):
        if r["layer"] == LAYER_PROVIDER and r["event"] == "session":
            return r["evidence"]
    return None


def summary():
    """按模块聚合 provider provenance，直接回答"用了谁/收益多少/拒了谁/退到哪"。

    返回 {semantic/slot: {selected, speedup, rejected, selects, fallbacks,
    fallback_actual}}。这是排障/灰度/回归的**摘要视图**；详细时序见 report()。
    """
    modules = {}
    for r in _RECORDS:
        if r["layer"] != LAYER_PROVIDER:
            continue
        ev = r["evidence"]
        if r["event"] == "select":
            d = modules.setdefault(r["subject"], {"rejected": [], "fallbacks": 0,
                                                  "selects": 0})
            d["selects"] += 1
            d["selected"] = ev.get("selected")
            d["speedup"] = ev.get("selected_speedup")
            d["rejected"] = ev.get("rejected", [])
        elif r["event"] == "call" and r["outcome"] == "fallback":
            d = modules.setdefault(r["subject"], {"rejected": [], "fallbacks": 0,
                                                  "selects": 0})
            d["fallbacks"] += 1
            d["fallback_actual"] = ev.get("fallback_actual")
    return modules


def report():
    """合并两层的可读报告：顶部摘要（每模块用了谁/收益/回退）+ 详细事件流。"""
    lines = []
    # 顶部：可复现快照 + 每模块聚合摘要
    sess = session()
    if sess:
        lines.append(f"[session] platform={sess.get('platform')} "
                     f"vendor={sess.get('vendor')} torch={sess.get('torch_version')} "
                     f"registered={sess.get('registered')}")
    summ = summary()
    if summ:
        lines.append("== 摘要（每模块：用了谁/收益/被拒/回退）==")
        for sem, d in summ.items():
            sp = d.get("speedup")
            sp_str = f"{sp}x" if sp is not None else "native"
            lines.append(f"  {sem}: selected={d.get('selected')} speedup={sp_str} "
                         f"rejected={len(d.get('rejected', []))} fallbacks={d.get('fallbacks', 0)}"
                         + (f" fallback_to={d.get('fallback_actual')}" if d.get('fallbacks') else ""))
        lines.append("== 详细事件流 ==")
    for r in _RECORDS:
        layer, ev = r["layer"], r["evidence"]
        if r["event"] == "session":
            continue  # 已在顶部摘要
        if layer == LAYER_BRIDGE:
            lines.append(f"[bridge] {r['subject']:<38} {r['outcome']:<14}"
                         f" -> {ev.get('target')}")
        elif layer == LAYER_PLATFORM:
            # D7 口径画像（caliber 事件）：让口径差异在报告中可见——文档 D7 要求
            # "即使不强制 FAIL，也让口径差异可见"，此前只进 JSONL 不进报告
            detail = " ".join(f"{k}={v}" for k, v in ev.items() if v is not None)
            lines.append(f"[caliber] {r['subject']}: {r['outcome']}  ({detail})")
        elif r["event"] == "gate":
            mark = "PASS" if r["outcome"] == "pass" else "FAIL"
            extra = " ".join(f"{k}={ev[k]}" for k in
                             ("probe_us", "ref_us", "speedup", "max_abs_err") if k in ev)
            lines.append(f"[gate]   {r['subject']}: {mark} {ev.get('reason', '')}"
                         + (f"  ({extra})" if extra else ""))
        elif r["event"] == "select":
            sp = ev.get('selected_speedup')
            sp_str = f"speedup={sp}x" if sp is not None else "native 底线"
            lines.append(f"[select] {r['subject']} -> {ev.get('selected')} "
                         f"({sp_str}, platform={ev.get('platform')})")
            for name, reason in ev.get("rejected", []):
                lines.append(f"    rejected: {name}  ({reason})")
        elif r["event"] == "wrap":
            lines.append(f"[wrap]   {r['subject']}: patch {ev.get('count')} 个模块 "
                         f"-> {sorted({m.split('->')[1] for m in ev.get('modules', [])})}")
        elif r["event"] == "switch":
            lines.append(f"[switch] {r['subject']} -> {r['outcome']}")
        elif r["event"] == "call" and r["outcome"] == "fallback":
            lines.append(f"[call]   {r['subject']}: {ev.get('impl')} -> "
                         f"{ev.get('fallback_actual')} ({ev.get('reason') or ev.get('error')})")
    return "\n".join(lines) if lines else "(无 provenance 记录)"
