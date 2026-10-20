# TTorch 未提交代码评审（2026-10-19）

> 评审对象：HEAD（1159c40）之后的全部未提交改动（25 文件，+751/-136 + 2 新增模块 +
> 3 新语义包）。评审依据：`docs/TT_Bridge_Provider_Architecture.md`（分层职责/边界）、
> `docs/TT_Provider_Function_Map.md`（四条不变式/六职责）、实测复现（每条 P0/P1 附证据）。
> 范围：cuda 伪装类硬件（xpu-masquerade）功能面 + ttorch 框架整体设计。

## 结论先行

本轮改动方向正确且主干坚实：review 十二项修复全部对症，函数级接入（bind_points +
dispatcher）把性能符号纳入了 provider 统一体系，门禁 fp32 精度轨道与 bwd 对账是真升级。
但有 **2 个实证确认的 P1 缺陷**（dispatcher 闭包不可 pickle、自检异常可炸穿
wrap_model）必须在合入前修复；另有若干一致性与文档漂移问题。修完两个 P1 即可提交。

---

## 一、P1：实证确认的缺陷（合入前必修）

### 1. dispatcher 闭包不可 pickle —— `torch.save(整模型)` 会炸

**实证**：`pickle.PicklingError: Can't pickle <function tt_dispatch_swiglu>`。

链路：`make_dispatcher` 返回的是闭包；megatron `MLP.__init__` 里
`self.activation_func = self.config.activation_func`（mlp.py:234）把绑定后的
dispatcher 存进模块实例属性 → 整模型 pickle 时炸。与上一轮 `wrap_module` 闭包
（已修为模块级 `ProviderModuleForward`）是**同一个坑的函数级版本**——修了一个
形态漏了另一个。

影响面：主训练路径的 checkpoint 走 state_dict（不受影响，已实测跑通）；EMA /
整模型序列化场景中招。

**修法**：dispatcher 改模块级可调用类（持有 `semantic` + `orig`——orig 是 megatron
模块级函数，可 pickle）：

```python
class FunctionDispatcher:
    """函数级接入的 dispatcher（模块级类：可 pickle/deepcopy——
    megatron MLP 实例的 activation_func 会持有它，闭包会让 torch.save 炸）。"""
    def __init__(self, semantic, orig):
        self._semantic = semantic
        self._orig = orig
    def __call__(self, *args, **kw):
        from ttorch.provider import manager as _pm
        m = _pm.instance()
        if m._bypass:
            return self._orig(*args, **kw)
        return m.call(self._semantic, *args, **kw)
```

### 2. `_self_check` 无异常保护 —— fp32 自检期模块异常炸穿 `wrap_model`

**实证**：构造 fp32 输入下抛 RuntimeError 的模块，`provider_wrap` 直接炸穿
（`RuntimeError` 穿透——`wrap_model` 只 catch `TypeError`）。

违反不变式 1（provider 缺失只降性能不降功能）：自检是 provider 内部机制，它的
失败必须收敛为"不接管"，不能影响模型可用性。fp32 轨道上 orig_forward 对输入
dtype/设备有新假设（厂商局部类、特殊模块未必支持 fp32），炸的概率不低。

**修法**：`_self_check` 里 `module.forward(x)` 与 native 调用段包 try——异常 =
自检失败，记 provenance（`rejected/self_check_error`）返回不接管。

---

## 二、P2：设计一致性问题（建议本轮处理）

### 3. provider → bridge.framework_binding 的依赖是 §154 条文的扩展

原文只约束"Provider 不依赖 cuda-compat、cuda-compat 不依赖 Provider"。本轮引入
`provider/manager.py → bridge.framework_binding`（注册 API）。方向上单向无环、
语义上可辩护（provider 消费 bridge 的绑定服务，类似 provider→witness 记录），
但**这是对成文边界的第一次扩展**——必须在 `TT_Provider_Function_Map.md` §6.1
补记一笔（含"为什么是注册式而非 bridge 反向 import provider"），否则下一个
读文档的人会误判为破坏。

### 4. `verify_impl` 的 kwargs 张量共享

精度段 kw 里浮点张量转 fp32 后，native 与 impl **共享同一对象**（未像 args 那样
clone 两份）。impl 若原地写 kw 张量会污染基准。当前三个语义（rmsnorm/swiglu/
rope_bshd）的 kw 均无张量，未触发——作为测量原语应堵上（clone 或注释声明约束）。

### 5. rope 门禁的 `max_grad_err=nan` 值得追查留证

厂商 `rotary_pos_emb_backward` 在 fp32 轨道产出 **nan**（此前 bf16 轨道实测错 2.43）
——比预期更糟。门禁行为正确（拒），但 provenance 里的 nan 证据是向厂商报 bug 的
关键素材，建议在 `vendor-stack.md` 缺陷条目里补上 fp32 轨道这一笔。

### 6. `weighted_bias_swiglu_impl` 在体系化后丢失盘点留痕

hack 版里有 verify=None 的 skipped 留痕；重构后该符号没有 provider 语义、不再注册
——megatron 若走到它则完全不经 TT（静默原生）。功能不破，但"盘点面"少了这条。
建议：建 `weighted_bias_swiglu` 语义（native 一行公式）或在该符号注释留档。

---

## 三、P3：文档漂移与工程细节

7. **`TT_Provider_Function_Map.md` §6 已落后于代码**：bind_points/函数级 dispatcher、
   fp32 精度轨道 + 生产 dtype 性能轨道、bwd 对账、负载签名（特征维归并）、自检
   两判据——§6.1/§6.3 都需要更新，否则 as-built 文档失去意义。
8. **热路径签名计算 vs §3.2-4"不在逐算子热路径做硬件识别和分发"**：签名计算是
   元数据读取（非硬件探测），实测开销在噪声内（v5 26s ≈ v3 直绑 26s）；且 §509
   "shape bucket 变化允许重选"正是签名分键的条文依据。设计自洽，但建议在 §6.3
   写明这个权衡（特征维归并的理由：变长训练每步重门禁不可接受 vs shape-sensitive
   防护）。
9. `verify.py:103` 错别字："反向前账" → "反向对账"。
10. `_bypass` 私有属性被 `ProviderModuleForward` / dispatcher 直接读——同包约定可
    接受，建议加 read-only 公开属性 `is_bypassed()` 收口。

---

## 四、做得对的（评审不是只挑错）

| 决策 | 依据 |
|---|---|
| 函数级接入统一进 provider（bind_points + dispatcher），而非散装 setattr | 分叉轴保持"语义 × 硬件"，无模型名/框架名渗入（§3.2-5）；门禁/回退/开关/provenance 全共享 |
| 门禁精度升 fp32 轨道、性能留生产 dtype | 与 §2 gate"数值等价 + 收益证据"拆判据的精神一致；实测 swiglu bf16 err 3e-2（误判区）vs fp32 err 2.4e-7（ulp 级）证明轨道选择正确 |
| bwd 对账进精度门禁 | 直接拦住厂商 rope kernel（fwd 逐位一致、bwd 错）——若无此门禁就是又一个静默数值事故 |
| 负载签名按特征维归并 | §509"shape bucket 变化允许重选"的落地；实测 D=256→512 重门禁拒绝、序列维变化直通 |
| 自检两判据分离（识别无阈值 + 准入 1e-3） | 4% 偏差模块被拒、整网 61 个 zcg 模块仍全接管——两个真实 case 都过 |
| 通道 B 的机制/语义分离（重构后） | bridge 只剩通用钩子机制，实现与判据归 provider，符合 §8 边界 |

## 五、修复建议顺序

1. P1-1 dispatcher 改模块级类（`FunctionDispatcher`）+ P1-2 `_self_check` 异常收敛
   ——半小时级，修完跑 demo.py + demo_replace_torchxmlir.py 回归；
2. P2-3 文档补记（Function_Map §6）+ P2-4 kwargs clone；
3. P3 项随下一轮顺手清。

---

## 附：评审方法与证据位置

- 设计条文引用：`docs/TT_Bridge_Provider_Architecture.md` §3/§4/§7/§8、
  `docs/TT_Provider_Function_Map.md` §0/§2/§6。
- 实证脚本均已当面试跑（dispatcher pickle、自检炸穿）；
- 性能证据：四次训练采集（44s 无优化 / 25s 标杆 / 26s ttorch 终态）；
  门禁证据：provenance 事件流（swiglu pass 1.99x、rope bwd 拒）。
