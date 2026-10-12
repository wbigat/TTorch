"""真实环境 preflight：摘掉 xswift、启用 TT Bridge 跑真实模型前的环境检查。

用法：
    python preflight_env.py

判定：退出码 0 = 可以进入 L1 验证（swift native SFT）；
     任何 FAIL 先解决再继续；SKIP/WARN 需人工确认是否符合预期。

检查项分组：
  A. 厂商栈与 xswift 状态（手术是否做干净）
  B. TT Bridge 激活链路（平台/conformance/cuda-compat 绑定）
  C. 训练依赖探针（transformers/deepspeed/accelerate/datasets 等版本）
  D. 分布式后端探针（单 rank init/destroy，默认 nccl——masquerade 栈会
     映射到 XCCL/BKCL；失败不阻塞但必须在跑多卡前搞清楚）
"""
import argparse
import importlib.util
import os
import sys

RESULTS = []


def report(level, item, detail=""):
    RESULTS.append((level, item, detail))
    print(f"  [{level}] {item}" + (f"：{detail}" if detail else ""))


def check(label, fn):
    print(f"\n== {label} ==")
    try:
        fn()
    except Exception as e:
        report("FAIL", label, f"{type(e).__name__}: {e}")


def _has_pkg(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def group_a():
    # 厂商栈必须在（masquerade 探测的 native 通道）
    if _has_pkg("torch_xmlir"):
        report("PASS", "torch_xmlir 已安装（KL3 厂商栈 native 通道）")
    else:
        report("WARN", "torch_xmlir 不存在", "非 KL3 栈则忽略；KL3 上缺失说明环境异常")

    # xswift 手术验证：两刀任选，关键是训练进程里它不生效
    xswift_present = _has_pkg("xswift_ext")
    env_gate = os.getenv("XSWIFT_PLUGIN_ENABLE", "")
    if not xswift_present:
        report("PASS", "xswift_ext 不可 import（已卸载/改名）")
    elif env_gate.upper() not in {"1", "TRUE", "YES", "Y", "ON"}:
        report("WARN", "xswift_ext 仍在但 XSWIFT_PLUGIN_ENABLE 未开",
               "补丁不会生效；若目标是硬摘除，建议改名目录双保险")
    else:
        report("FAIL", "xswift_ext 在场且 XSWIFT_PLUGIN_ENABLE 已开启",
               "补丁会生效，与 TT Bridge 叠加产生双层 patch，先摘干净")

    # 版本参照（xswift 验证过的适配窗口，偏离则上游行为未经验证）
    from importlib import metadata
    for dist, series in (("ms-swift", "4.3.x"), ("mcore-bridge", "1.5.x")):
        try:
            v = metadata.version(dist)
            ok = v.startswith(series.rstrip("x").rstrip("."))
            report("PASS" if ok else "WARN", f"{dist}={v}",
                   "" if ok else f"偏离 xswift 验证窗口 {series}，裸上游行为未经验证")
        except metadata.PackageNotFoundError:
            report("WARN", f"{dist} 未安装", "megatron 路径必需；native 路径可忽略 mcore-bridge")


def group_b(platform=None):
    import torch
    report("PASS", f"torch={torch.__version__}")
    from tt_bridge import entry
    name = entry.activate(platform=platform)
    report("PASS", f"TT Bridge 激活 platform={name}",
           f"vendor={__import__('tt_bridge.platform_mgr', fromlist=['x']).vendor_name()}")

    # 绑定后真实行为抽查：张量经门面语义落到设备
    from tt_bridge import platform_mgr
    t = torch.tensor([1.0]).to(platform_mgr.device_spec())
    torch.cuda.synchronize()  # 经绑定的门面
    report("PASS", f"张量上卡 + 门面 synchronize 正常（device={t.device}）")


def group_c():
    for pkg in ("transformers", "deepspeed", "accelerate", "datasets", "trl", "peft"):
        try:
            m = __import__(pkg)
            report("PASS", f"{pkg}={getattr(m, '__version__', '?')}")
        except ImportError:
            report("WARN" if pkg in ("trl", "peft") else "FAIL", f"{pkg} 缺失")


def group_d():
    import torch
    import torch.distributed as dist
    backend = os.getenv("TT_PREFLIGHT_DIST_BACKEND", "nccl")
    if not dist.is_available():
        report("FAIL", "torch.distributed 不可用")
        return
    try:
        port = 29511
        dist.init_process_group(backend=backend, init_method=f"tcp://127.0.0.1:{port}",
                                world_size=1, rank=0)
        dist.destroy_process_group()
        report("PASS", f"dist backend={backend} 单 rank init/destroy 正常",
               "masquerade 栈应映射到 XCCL/BKCL；多卡前再验真实集合通信")
    except Exception as e:
        report("WARN", f"dist backend={backend} 初始化失败", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=None,
                    help="调试 override（如 cpu 做本地验证）；默认自动检测")
    args = ap.parse_args()
    print("== TT Bridge preflight（摘掉 xswift 后的真实环境检查）==")
    check("A. 厂商栈与 xswift 状态", group_a)
    check("B. TT Bridge 激活链路", lambda: group_b(args.platform))
    check("C. 训练依赖探针", group_c)
    check("D. 分布式后端探针", group_d)

    fails = sum(1 for lv, _, _ in RESULTS if lv == "FAIL")
    warns = sum(1 for lv, _, _ in RESULTS if lv == "WARN")
    print(f"\n汇总: FAIL={fails} WARN={warns} PASS={sum(1 for lv, *_ in RESULTS if lv == 'PASS')}")
    print("判定: " + ("可进入 L1 验证" if fails == 0 else "先解决 FAIL 项"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
