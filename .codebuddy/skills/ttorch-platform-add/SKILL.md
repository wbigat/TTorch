---
name: ttorch-platform-add
description: TTorch 新增 platform 平台类型（接入新硬件/新软件栈）：catalog detect + SPEC 登记、adapter 实现（PlatformBase 子类）、conformance 门禁用例。当接入一块新卡/新厂商栈、修改平台探测/激活流水线、写平台适配层时使用。关键词：platform、adapter、catalog、detect、PlatformBase、PlatformSpec、conformance、device_spec、masquerade、双身份、新硬件接入。
---

# TTorch 新增 Platform 类型

接入一块新卡 = 两处改动：`catalog.py` 加一段（detect + SPEC）+ `adapters/<平台>_adapter.py`
写实现类。激活流水线（装配 → SDK 版本协商 → initialize → conformance 门禁）自动复用。

## 1. catalog 登记（ttorch/platform/adapters/catalog.py）

```python
NEW = PlatformSpec(
    name="...", vendor_name="...", device_name="...",
    detect=_detect_new, impl="ttorch.platform.adapters.new_adapter:TorchNewPlatform")
```

detect 硬规则：
- **只看不初始化**：`find_spec` 查包、`torch.cuda/xpu` API 探测；**禁 import 任何厂商包**
  （厂商包 import 有副作用）。
- **torch 惰性 import**：无 torch 的机器要能 import catalog（mock demo 的设计验证环境）；
  detect 拿不到 torch 就诚实报缺席。
- **masquerade 防御**：冒充 CUDA 的栈会让 `is_available()` 报 True——detect "cuda" 要
  用厂商包/卡名排除伪装者；masquerade 平台自身设 `masquerades_cuda=True`。

## 2. adapter 实现（PlatformBase 子类）

- **契约即 `torch.cuda.*` 真实符号**（TORCH_API 17 个）：方法名/签名逐一对齐且为超集。

### 2.1 契约必须三处同步（最易漏，2026-10-21 实测踩过）

往 `TORCH_API` 加一个符号 = **同时改三处**，缺任一处都是契约漂移：

| # | 位置 | 作用 |
|---|------|------|
| 1 | `platform_base.TORCH_API` 元组 | 门面转发白名单（`interface.__getattr__` 据此生成） |
| 2 | **`PlatformBase` 基类的默认实现**（抛 unsupported） | 未实现的平台得到**统一**的 `TTPlatformError("unsupported")` 而非 `AttributeError` |
| 3 | 各 adapter 子类实现 | 真实平台行为 |

为什么第 2 处最容易漏：子类一实现，当前平台立刻能跑通——问题只在"别的平台没
实现"时才暴露，症状是框架拿到 `AttributeError` 而不是约定的降级路径。**这是
"不支持即不实现"原则的前提**：基类必须先给出 unsupported 兜底，"不实现"才
等价于"明确不支持"。

反之同样要查：子类新增的 public 方法若在基类没有对应（且不是 `EXTRA_FORWARD`
这类可选扩展），说明契约登记漏了。

自检（加完符号必跑，另见 §5）：

```bash
python3 -c "from ttorch.platform.platform_base import TORCH_API, PlatformBase; \
print([n for n in TORCH_API if not hasattr(PlatformBase, n)] or '契约对齐 ✅')"
```
- 走 torch 设备模块范式（torch.cuda/xpu/npu 同名 API）就继承 `TorchDeviceModuleBase`：
  只需指定 `mod_path` + 实现 `sdk_version()`；所有 torch 调用经 `_orig(name)` 拿绑定前
  原实现（防"门面→adapter→门面"无限递归）。
- 范式外平台（如 CPU）直接继承 `PlatformBase`。
- **不支持 = 不实现**：保持默认抛 `TTPlatformError("unsupported")`，框架走降级；
  不写假实现。能力由 `handle.supports(name)` 从"子类有没有覆盖"派生（无自报表）。
- **G13**：可选 device 参数 None=省略实参。**G15**：RNG state 类型=平台原生对象。
- **stream 返回原生对象**，门面不包装（G14）；`stream()` ctx 交给厂商实现，不自造。
- `sdk_range`/`sdk_version()` 对齐厂商栈版本格式（可能带 `+后缀`，`_ver_tuple` 已处理）。

## 3. 双身份（写 SPEC 时想清楚）

- `vendor_name`：厂商身份（provider 分发/provenance 用）。
- `device_name`：torch 认得的设备类型——框架 `device.type == "cuda"` 比较的存活线。
  masquerade 栈必须 `"cuda"`；native 栈用真实名（如 `"xpu"`）。
- 框架构造 torch.device 的唯一正途：`handle.device_spec(idx)`。

## 4. conformance 门禁（新平台必须通过）

激活期强制。写新平台时对照既有用例自验：sync 必须 device-wide（含运行期新建 stream）、
stream fencing 是 tensor 级语义（跨流等待落成依赖边）、RNG 回环相等 + 非法值统一拒绝、
mem 查询实时口径、current_device 返回值能回喂 set_device/张量构造。

## 5. 验证

```bash
python3 -c "import ttorch; ttorch.activate()"   # detect 命中 + conformance 全 PASS
python3 -c "import ttorch; ttorch.activate(); print(ttorch.check_bindings()[0])"  # 绑定验收
cd examples/bridge-demo && python3 demo_bridge.py     # mock 链路不被打坏

# 契约三处同步自检（加/改 TORCH_API 后必跑，应输出"契约对齐 ✅"）
python3 -c "from ttorch.platform.platform_base import TORCH_API, PlatformBase; \
print([n for n in TORCH_API if not hasattr(PlatformBase, n)] or '契约对齐 ✅')"
```
