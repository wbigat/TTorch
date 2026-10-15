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

关于"统一 vs 分离"的取舍：信封字段（seq/ts/layer/event/subject/outcome）归一，
层差异收进 evidence——两层确有真实差异（bridge 无运行期 per-call、provider 无
swept 概念），强压完全一致反而失真，故信封归一 + evidence 分层。这既满足
"合并成一份报告"，又不制造伪统一。

存储是模块级私有 `_RECORDS`，外部只能经 `record()` 写、`records()`/`report()`/
`export_jsonl()` 读——不暴露可变 list，杜绝外部 append/clear 破坏不变式。
"""
import json
import time

_RECORDS = []
_seq = 0

LAYER_BRIDGE = "bridge"
LAYER_PROVIDER = "provider"


def record(layer, event, subject, outcome, **evidence):
    """追加一条统一信封事件。layer 取 LAYER_BRIDGE / LAYER_PROVIDER。"""
    global _seq
    _seq += 1
    _RECORDS.append({"seq": _seq, "ts": round(time.time(), 6),
                     "layer": layer, "event": event, "subject": subject,
                     "outcome": outcome, "evidence": evidence})


def records(layer=None):
    """只读快照（tuple）；layer 给定时只取该层（保持各自相对时序）。"""
    if layer is None:
        return tuple(_RECORDS)
    return tuple(r for r in _RECORDS if r["layer"] == layer)


def export_jsonl(path, layer=None):
    """导出 JSONL（默认全量，可按层过滤）。生成物，供 diff/审计。"""
    with open(path, "w") as f:
        for r in records(layer):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
