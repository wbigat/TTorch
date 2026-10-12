# 摘掉 xswift、启用 TT Bridge 跑真实模型 SFT — 迁移 Runbook

> 目标：在 KL3 真实环境上，临时摘除 xswift_ext，用 TT Bridge 接管设备层，
> 跑通基础 LLM SFT。本文档回答三个问题：xswift 到底挂载在哪、摘掉它会丢什么、
> TT 侧要补什么、按什么顺序验证。

## 1. xswift_ext 的挂载机制（手术对象清单）

xswift_ext **没有 setup.py/pyproject**——不是标准 pip 包，靠三件套生效：

| 层 | 机制 | 文件证据 |
|---|---|---|
| 触发 | 解释器启动时装 lazy import 钩子（通常经 site-packages 里的 `.pth` 文件 `import xswift_ext.auto_init`） | `auto_init.py`（模块末行 `init_if_enabled()`） |
| 门禁 | 环境变量 `XSWIFT_PLUGIN_ENABLE=1` 才真正初始化 | `auto_init.py::_is_enabled` |
| 执行 | 首次 `import swift` / `import mcore_bridge` 前触发 `xswift_plugin_init()`：版本检查（ms-swift 4.3.x / mcore-bridge 1.5.x）→ `envs.init_envs()` → 三组符号重写注册 → `KlxSymbolRewriter.run()` | `entry.py`、`plugin.py` |

**注意**：xswift 只 patch，不提供训练本体——`swift`（ms-swift）和 `mcore_bridge`
是独立安装的 dist，摘掉 xswift 后它们以**裸上游形态**运行。

## 2. 摘掉 xswift 会丢什么（按训练路径分组）

补丁全清单（`plugin.py` 三个注册函数），按"第一刀验证路径"标注处置：

| 补丁 | 路径 | 性质 | 摘掉后果 | 处置 |
|---|---|---|---|---|
| `swift.loss.embedding.gather_object` → CPU gather | 共有（embedding loss 特性） | 正确性 | 用到该特性时 CUDA 反序列化崩溃 | 第一轮不启用该特性即可规避 |
| `PrintCallback` wrapper | megatron | 监控增强 | 少几行日志 | 不需要 |
| `BaseMegatronTrainer.__init__` wrapper | megatron | profiling 注入 | 无 profiler | 不需要（TT provenance 部分覆盖） |
| torch DCP `_validate_global_plan` wrapper | megatron ckpt | 版本兼容（正确性） | save/load 在新 torch 上可能炸 | 第一轮不存 ckpt 可规避；之后移植为 TT WRAP 决策 |
| `wrap_model` 替换（DDP 侧流 fencing） | megatron 多卡 | **正确性** | DDP 广播与计算流的竞态（hang/脏数据风险） | **megatron 多卡前必须移植**（WRAP 决策候选） |
| evaluate 禁用 fast-FC | megatron eval | XPU 规避 | eval 可能出错 | 第一轮跳过 eval |
| `Conv3d.forward` wrapper | VLM | 正确性 | 纯文本 LLM 无影响 | 文本 SFT 不需要 |
| BSHD/THD rope wrapper（fused rope） | megatron | **性能**（自带 unfused 回退） | rope 变慢但正确 | 正好是 TT Provider 的接管位（demo 已验证模式） |
| XTE large-weight / LoRA merge wrapper | megatron（env-gated） | 权重布局 | env 不开就不触发 | 保持 env 关闭 |
| max-seqlen pad warmup 三处 | 共有（env-gated，默认关） | 性能 | 默认关 | 不需要 |

**关键结论**：
- **swift native SFT（HF trainer + DeepSpeed）路径**：xswift 默认配置下几乎没有
  有效补丁（megatron 专属的不加载，env-gated 的默认关）——**第一轮验证选它**，
  丢失风险最小。
- **megatron 路径**：`wrap_model` fencing 是正确性补丁（多卡必踩），DCP wrapper
  影响 ckpt——**这两个移植进 TT 之前，megatron 多卡不要摘掉 xswift**。

## 3. TT 侧要补的东西（本次已落地）

| 组件 | 文件 | 作用 |
|---|---|---|
| 激活入口 | `tt_bridge/entry.py` | `activate()` 一键完成 平台注册→自动检测→conformance 门禁→cuda-compat 绑定；幂等；rank0-only 摘要；`TT_PLATFORM` env 调试 override |
| 懒加载钩子 | `entry.install_lazy_hook()` | 与 xswift auto_init 同形态：首次 import swift/mcore_bridge/megatron 前自动激活（sitecustomize 场景用） |
| launcher | `tt_bridge/launch.py` | `python -m tt_bridge.launch <script>` 先激活再 runpy 目标脚本——**激活先于框架 import，from-import 快照天然拿到门面符号，扫尾几乎不需要** |
| preflight | `preflight_env.py` | 上机先跑：A 厂商栈/xswift 状态、B Bridge 激活链路、C 训练依赖、D dist 后端探针；退出码判定能否进 L1 |

**真实环境运行方式**（tt-bridge-demo 目录上 PYTHONPATH）：

```bash
export PYTHONPATH=/path/to/tt-bridge-demo:$PYTHONPATH
python preflight_env.py                     # 手术与环境检查，FAIL=0 再继续
python -m tt_bridge.launch $(which swift) sft --model <model> --dataset <ds> ...  # 原命令照抄
```

## 4. 手术方案（三档，渐进式）

| 档 | 操作 | 可逆性 | 适用 |
|---|---|---|---|
| S1 软摘除 | `unset XSWIFT_PLUGIN_ENABLE`（或设 0） | 秒级恢复 | 第一次试跑，xswift 包仍在但不生效 |
| S2 断触发 | 改名 site-packages 里的 `xswift_ext.pth`（→ `.pth.bak`） | 改回即恢复 | 确认 S1 稳定后，排除钩子层干扰 |
| S3 硬摘除 | 改名 `xswift_ext/` 目录 | 改回即恢复 | 验证"包不存在"形态；注意 launch 脚本里若有显式 `import xswift_ext` 会先炸 |

建议顺序 S1→S2→S3，每档都跑一遍 L1 验证。**不要 pip uninstall**——非标准包
卸载不干净且不可快速恢复。

## 5. 验证阶梯

| 级 | 内容 | 通过标准 |
|---|---|---|
| L0 | `python preflight_env.py` | FAIL=0（torch_xmlir 在、xswift 不生效、Bridge 激活、dist 后端可 init） |
| L1 | `python real_sft_smoke.py`（仓库自带，不依赖 swift） | SMOKE PASS（已修复 G13 后应全绿） |
| L2 | swift native SFT 小配置：小模型（如 Qwen3-0.6B）+ 小数据集 + 单卡 + `MAX_STEPS` 截断 | loss 正常下降、无 CUDA 字样报错、无 hang |
| L3 | 同配置跑基线对照：恢复 xswift（S1 反向）跑一次 | 两侧 loss 曲线前几步量级一致（不要求逐位，KL3 栈自身有数值噪声） |
| L4 | 多卡（DeepSpeed） | dist 正常、各 rank 显存均衡、吞吐不低于带 xswift 基线的 90%（裸上游少了 fused rope，megatron 路径才会有差距） |

## 6. 已知风险与开放问题

1. **dist 后端字符串**：masquerade 栈上 `init_process_group(backend="nccl")` 是否
   被 shim 映射到 BKCL——preflight D 组只能验单 rank init；多卡集合通信的
   真实验证在 L4，若 hang 先查 `XCCL_*/BKCL_*` 环境变量是否需要替代 `NCCL_*`。
2. **flash-attn**：环境日志显示 xpu_fla 把 triton patch 成 CPU 回退——swift 若
   默认选 flash-attn 实现可能踩坑，L2 配置显式指定 attention 实现（如 sdpa/eager）。
3. **数值口径**：L3 对照时不要追求逐位一致；跨硬件精度口径 abs/rel<5e-3，
   同硬件 patch 有无的对照只看 loss 趋势与量级。
4. **megatron 路径的两个正确性补丁**（wrap_model fencing、DCP wrapper）未移植前，
   megatron 多卡仍依赖 xswift——它们属于框架内部 WRAP，是 TT cuda-compat
   WRAP 决策落地后的第一批真实用户。
