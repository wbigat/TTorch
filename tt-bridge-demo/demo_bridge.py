"""Bridge 垂直切片 demo（v10：P0 接口族 + 基础 SFT 流程实战）。

v10 新增（对照 TT_Swift_Cuda_API_Inventory.md 的 P0 清单）：
- is_available（capability/probed：分支选择器）
- stream 对象族：Stream/current_stream/stream ctx/wait_stream
  （XPU 侧 streamWait 等全是非标名，adapter 填 gap）
- RNG state 存取（不透明 str 契约，checkpoint resume 正确性）
- empty_cache / mem_get_info（XPU 侧 MiB dict -> 字节 tuple 换算）
- conformance 新增 5 项强制测试：rng 回环、stream fencing 依赖边、
  stream 上下文恢复、mem_info 口径一致、设备标识可回喂
- 第 11 节：基础 SFT 流程实战（swift 主链路形态），在 xpu-only
  机器上不改框架源码跑通

v9 新增（FlagOS 真坑沉淀）：
- 双身份：vendor_name（厂商身份，provider 分发/provenance 用）与
  device_name（torch 认得的设备类型，框架 device.type 比较的存活线），
  作为 capability 的 identity 档由 manager 注入，adapter 不许自报
- masquerade 降级：GCU 类栈冒充 CUDA（torch.cuda.is_available() 报 True），
  native 探测命中时把被冒充的 compat claim 降级为证据而非冲突；
  无 native adapter 时冒充必然得逞（对照演示双身份必须成对建设）

v8 仅做组织调整：adapter 契约（platform_adapter.py）与各实现
（adapters.py）独立成文件，platform_mgr.py 只剩管理面 + 分发门面；
另补 capabilities() 全量快照查询。

场景：xpu-only 机器（没有 CUDA 驱动），框架源码一行不改。
"""
import os
from collections import Counter

import framework
import mock_hw
import vendor_cuda
from tt_bridge import bindings  # noqa: F401  # import 即注册（binding_register 调用在模块级）
from tt_bridge import admission, cuda_compat, platform_mgr, registry  # noqa: F401
from tt_bridge.adapters import CudaPlatform, GcuPlatform, NaiveXpuPlatform, XpuPlatform

print("== 0. 未激活 Bridge：框架在 xpu-only 机器上直接跑 ==")
try:
    framework.init_device(0)
    print("  意外成功（不应发生）")
except RuntimeError as e:
    print(f"  崩溃（符合预期）: {e}")

print("\n== 1. Platform Manager：注册全部类型 + 自动检测 + 激活门禁 ==")
platform_mgr.register(XpuPlatform)
platform_mgr.register(CudaPlatform)      # 注册了但本机检测为 absent
platform_mgr.register(NaiveXpuPlatform)  # auto_detect=False，不参与自动检测
platform_mgr.auto_activate()             # 内部：版本协商 -> init -> schema -> conformance

# red 1：多种硬件同时在线时拒绝猜测
mock_hw.PRESENT_HARDWARE.add("cuda")
try:
    platform_mgr.auto_activate()
    print("  意外成功（不应发生）")
except RuntimeError as e:
    print(f"  多平台冲突 fail fast（符合预期）: {e}")
finally:
    mock_hw.PRESENT_HARDWARE.discard("cuda")

# red 2：不填 gap 的 adapter 被 conformance 门禁拦在激活期
try:
    platform_mgr.select("xpu-naive")
    print("  意外成功（不应发生）")
except platform_mgr.TTPlatformError as e:
    print(f"  naive adapter 被门禁拦下（符合预期）: {e}")

# red 3：SDK 版本越界被版本协商拦下
_orig_ver = mock_hw.XpuHw.sdk_version
mock_hw.XpuHw.sdk_version = "9.9"
try:
    platform_mgr.select("xpu")
    print("  意外成功（不应发生）")
except platform_mgr.TTPlatformError as e:
    print(f"  SDK 版本越界被拦（符合预期）: {e}")
finally:
    mock_hw.XpuHw.sdk_version = _orig_ver

# red 4：masquerade —— GCU 栈冒充 CUDA，native 探测命中后 cuda claim 降级
platform_mgr.register(GcuPlatform)
_saved_hw = set(mock_hw.PRESENT_HARDWARE)
mock_hw.PRESENT_HARDWARE.clear()
mock_hw.PRESENT_HARDWARE.add("gcu")
platform_mgr.auto_activate()
assert platform_mgr.vendor_name() == "gcu", platform_mgr.vendor_name()
assert platform_mgr.device_name() == "cuda", platform_mgr.device_name()
print(f"  GCU 机器：激活 vendor={platform_mgr.vendor_name()}，"
      f"torch 视角 device_name={platform_mgr.device_name()} "
      f"（未误判成 NVIDIA，框架 device.type 比较继续成立）")

# red 4 对照：native adapter 缺失时冒充必然得逞 —— 双身份必须成对建设
platform_mgr._REGISTRY.pop("gcu")
platform_mgr.auto_activate()
_trapped = platform_mgr.vendor_name()
assert _trapped == "nvidia-cuda", _trapped
print(f"  对照（无 GCU native adapter）：误激活 vendor={_trapped}，masquerade 得逞")
platform_mgr.register(GcuPlatform)
mock_hw.PRESENT_HARDWARE.clear()
mock_hw.PRESENT_HARDWARE.update(_saved_hw)
platform_mgr.auto_activate()  # 恢复 xpu 激活态，继续主流程

print("\n== 2. policy 注册表（code-first，import 期已完成全部校验）==")
entries = registry.entries()
cats = Counter(e.category.value for e in entries)
print(f"  注册 {len(entries)} 条: {dict(cats)}")

print("\n== 3. cuda-compat 激活（绑定 + from-import 扫尾 + 默认拒绝）==")
cuda_compat.activate(entries)
ok = framework.set_device is platform_mgr.set_device
print(f"  from-import 扫尾生效: framework.set_device is platform_mgr.set_device -> {ok}")
assert ok
print(f"  原始实现留痕: hasattr(vendor_cuda, '_origin_set_device') -> "
      f"{hasattr(vendor_cuda, '_origin_set_device')}")

print("\n== 4. admission 准入测试（只验绑定正确性；契约履行已在激活期由 conformance 验过）==")
ctx = {"framework": framework, "vendor_cuda": vendor_cuda}
ran = set()
for e in entries:
    if e.test and e.test not in ran:
        ran.add(e.test)
        ok, detail = e.test(ctx)
        print(f"  [{'PASS' if ok else 'FAIL'}] {e.test.__name__:<24} "
              f"@{e.category.value:<12}: {detail}")

print("\n== 5. 默认拒绝：未注册的符号 ==")
try:
    vendor_cuda.set_sync_debug_mode(1)
    print("  意外成功（不应发生）")
except cuda_compat.TTCompatError as e:
    print(f"  fail fast（符合预期）: {e}")

print("\n== 6. 框架正常运行：三类接口 + 三种决策的实际去向 ==")
framework.init_device(1)
framework.all_reduce_hook()
print(f"  behavior   : pick_comm_path -> {framework.pick_comm_path()}"
      f"（capability 选择期消费：XPU 无 P2P，走 staging）")
framework.annotated_step(lambda: None)
print(f"  behavior   : annotated_step -> 硬件 marks={platform_mgr.active().hw.marks}")
print(f"  state_query: memory_report -> {framework.memory_report()}B（实时读）")
r = framework.capture_step(lambda: "train step executed (eager)")
print(f"  fallback   : graph_capture -> {r}")
try:
    framework.profile_step()
except cuda_compat.TTCompatError as e:
    print(f"  exception  : {e}")

print("\n== 7. 统一错误映射：厂商原生错误 -> TTPlatformError ==")
try:
    framework.init_device(99)  # 越界设备号
    print("  意外成功（不应发生）")
except platform_mgr.TTPlatformError as e:
    print(f"  非法设备 -> 统一类别 [{e.category}]（XpuHwError 已被 adapter 翻译）: {e}")

print("\n== 8. 导出 YAML 快照（生成物，CI diff 评审用，永不手改）==")
snap = os.path.join(os.path.dirname(__file__), "policy_snapshot.yaml")
registry.export_yaml(snap)
with open(snap) as f:
    print("\n".join("  " + line for line in f.read().splitlines()[:8]))
print(f"  ...（完整快照见 {os.path.basename(snap)}，提交时 diff 必须一致）")

print("\n== 9. 覆盖对账 + trace（Bridge 侧 provenance，可导出）==")
for dotted, status in cuda_compat.coverage_report(entries, ["vendor_cuda"]):
    print(f"  {dotted:<38}{status}")
print()
print(f"  {'symbol':<36}{'category':<14}{'decision':<14}{'target':<18}swept")
for t in cuda_compat.TRACE:
    print(f"  {t['symbol']:<36}{t['category']:<14}{t['decision']:<14}"
          f"{str(t['target']):<18}{t['swept_modules']}")

print("\n== 11. 基础 SFT 流程实战（P0 接口族，swift 主链路形态）==")
dev = framework.pick_device_or_cpu()
assert dev == "accelerator"
print(f"  is_available 分支: {dev}（rollout.py 形态）")
free_gb, total_gb = framework.device_budget_gb()
assert total_gb == 80.0 and 0 < free_gb <= total_gb
print(f"  设备预算: free={free_gb}GB / total={total_gb}GB（patcher max_memory 形态）")
t, dev_id = framework.tensor_on_current_device(4)
assert len(t) == 4 and dev_id == platform_mgr.current_device()
print(f"  current_device 当设备标识: tensor(len=4) on device={dev_id}"
      f"（megatron utils.py:214 形态）")
ckpt = framework.sft_step()
hw = platform_mgr.active().hw
assert len(hw.wait_edges) >= 2, hw.wait_edges  # 双向 fencing 依赖边
print(f"  sft_step 完成: RNG 已存={ckpt['rng']!r}，fencing 依赖边={hw.wait_edges}")
framework.resume_from(ckpt)
assert platform_mgr.get_rng_state() == ckpt["rng"]
print(f"  resume: RNG 回环一致（megatron_lm_utils.py:107/599 形态）")
assert "purge" in hw.marks  # empty_cache 落到 XPU 非标 purge
print(f"  empty_cache -> XPU 非标 purge 留痕: {hw.marks}")

print("\n== 10. 生命周期收尾 ==")
platform_mgr.finalize()