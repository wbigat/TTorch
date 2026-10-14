"""诊断与观测（diagnostics）：绑定状态确认 + 运行期中转证据。

归位理由（TT_Bridge_Provider_Architecture.md §9）：可观测性是 TT 的核心
工程资产，不是入口函数的附属品。这里收编两类检查，都回答"绑定是否真实
生效"，但层次不同：

- check_bindings()      静态/激活期：每个符号是否绑到预期门面
  （逐条跑 admission 测试，含 G14 的 vendor_internal_construction）
- start_call_trace()    运行期：训练过程中**实际**经 TT 中转的调用与调用方
  /call_trace_report()    （`torch.cuda.X is interface.X` 只证明符号指向，
                            不证明真的调用到了——中转是运行期事实）

与 Bridge 内层的关系：本模块只**读** platform 的门面与激活态
（经 ttorch.active() 拿 entries），不改绑定。绑定的写入只发生在
ttorch.activate()，观测与操作分离。

这两个函数都是调试/验收工具，不进热路径（探针有 traceback 开销，
仅在需要取证时 start_call_trace()，默认不开启）。
"""
import collections
import sys
import traceback

_calls = None


def check_bindings(entries):
    """绑定正确性自检：逐条跑 admission 测试。

    entries 由调用方传入（ttorch.active().entries），本模块不反向依赖
    入口层——保持观测模块独立，可被 Bridge 内层复用而不成环。
    返回 (n_fail, rows)，rows 为 (ok, test_name, detail)。
    """
    import torch
    ctx = {"vendor_cuda": torch.cuda}
    rows = []
    n_fail = 0
    for e in entries:
        ok, detail = e.test(ctx)
        rows.append((ok, e.test.__name__, detail))
        n_fail += 0 if ok else 1
    return n_fail, rows


def start_call_trace(entries):
    """给门面装计数探针，统计运行期实际经 TT 中转的调用及其调用方。

    探针插在门面与框架之间（替换 torch.cuda 上绑定后的符号），框架侧
    看到的仍是绑定结果，行为不变，仅多一层计数。

    返回 collections.Counter（符号 -> 次数）；调用方用 call_trace_report()
    读取。幂等：重复调用返回同一份计数器。
    """
    from ttorch.platform import interface as pm

    global _calls
    if _calls is not None:
        return _calls
    _calls = collections.Counter()
    callers = collections.defaultdict(set)
    _calls.callers = callers  # 附在同一对象上，调用方不必管两份状态

    import torch
    for e in entries:
        name = e.symbol.__name__
        interface = getattr(pm, name, None)
        if interface is None or not callable(interface):
            continue

        def probe(*a, _n=name, _f=interface, **k):
            _calls[_n] += 1
            # 向上找第一个非 ttorch 栈帧 = 真实调用方（跳过门面与本模块）
            for fr in traceback.extract_stack()[:-1][::-1]:
                if "/ttorch/" not in fr.filename:
                    callers[_n].add(f"{fr.filename.split('/')[-1]}:{fr.name}")
                    break
            return _f(*a, **k)

        # 只替换"框架侧看到的就是门面"的符号；厂商内部构造路径
        # （G14 的 Stream(stream_id=...)）不挂探针，避免干扰分流
        if getattr(torch.cuda, name, None) is interface:
            setattr(torch.cuda, name, probe)
    return _calls


def call_trace_report(min_calls=1, exclude_prefixes=()):
    """中转证据报告。

    返回 (ok, rows)。每行 (符号, 次数, 全部调用方, 外部调用方)。
    exclude_prefixes：调用方文件名前缀黑名单——把调用方自己的验收代码
    排除掉，只留"框架侧"的真实调用。ok = 存在至少 min_calls 次框架侧调用，
    这是"TT 在链路上而非形同虚设"的最低证据。
    """
    if _calls is None:
        return False, []
    rows = []
    for name, n in _calls.most_common():
        callers = sorted(_calls.callers.get(name, ()))
        external = [c for c in callers
                    if not c.startswith(exclude_prefixes)] if exclude_prefixes else list(callers)
        rows.append((name, n, callers, external))
    ok = sum(n for _, n, _, ext in rows if ext) >= min_calls
    return ok, rows
