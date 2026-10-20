---
name: ttorch-bridge-extend
description: TTorch bridge 层兼容接口扩展：新增/修改 torch.cuda 符号绑定（policy Entry）、扫尾白名单、FALLBACK/EXCEPTION 决策、checkpoint bridge。当需要让某个 torch API 经门面收束到平台实现、排查绑定/扫尾/篡改问题、或扩展 bridge 兼容面时使用。关键词：bridge、cuda_compat、policy、Entry、绑定、门面、扫尾、sweep、REMAP、FALLBACK、admission、check_bindings。
---

# TTorch Bridge 兼容接口扩展

bridge 层的职责：把框架的 `torch.cuda.*` 调用收束到平台门面（功能接入 + checkpoint）。
依赖方向：bridge → platform / witness，不碰 provider。

## 全景：四个注册点 + 四种生效方式（排障先定位在哪一环）

```
① .pth 阶段（框架未 import）：平台 SPECS 注册（catalog）+ torch.cuda policy
   注册（module_binding）；TT_PLUGIN_ENABLE=1 时 activate()：platform 激活 →
   cuda_compat 绑定（通道 A）→ provider 注册 + bind_points 注册 → install 钩子
② 框架 import：通道 B 钩子触发 → 函数符号 = FunctionDispatcher
③ 模型构建后：provider_wrap(model) → 模块级 wrap（fp32 自检，不过不接管）
④ 首次前向：惰性 select()（三级门禁）→ 绑定缓存（按负载签名），之后直通
```

| 环节 | 注册内容 | 生效时机 | 失败行为 |
|---|---|---|---|
| 平台 | PlatformSpec（import 即注册） | auto_activate（conformance 门禁） | fail fast |
| torch.cuda policy | Entry（import 期静态构造） | cuda_compat.activate + verify_sweep | 扫尾残留 fail fast |
| provider | Provider（activate eager 注册） | 首次 call 惰性 select | 门禁全拒→native |
| 函数级 | bind_points → provider/fn_binding | 目标模块 import 完成瞬间 | 符号不存在→skipped |

不变量：注册 eager 无副作用，生效必须有证据；时序靠证据不靠猜（verify_sweep /
assert_bindings_intact）；一切失败退到功能底线。

## 新增一条绑定 = 注册一条 Entry（ttorch/bridge/module_binding.py）

```python
register_binding(Entry(symbol=torch.cuda.X, decision=R,
               impl=interface.X,            # 或 impl_factory=...（二选一）
               test=admission.xxx,          # 准入测试，不许免检
               intent="为什么这样处置",      # 必填：没有 intent 半年后没人懂
               bind_module="torch.cuda"))   # 显式给框架可见的 re-export 路径
```

## 硬规则（违反即事故，均来自实测）

1. **类型位符号绑类对象**：`torch.cuda.Stream` 等参与注解求值（`Stream | None`，
   transformers 5.x import 期即求值）、isinstance、type() 的符号，用
   `impl_factory(entry, original) -> object` 在 bind 期解析平台原生类——绑转发函数
   是实测过的现网阻断（P0）。
2. **G12**：门面方法名/签名 = torch 真实符号的超集（改名/收窄 → 带参调用点 TypeError）。
3. **G13**：可选 device 参数 None=**省略实参**，不位置透传（厂商栈不一定接受 None）。
4. **G14**：门面不包装原生 stream 对象；torch 内部 `Stream(stream_id=...)` 构造也落
   门面——原生类绑定后自然正确。
5. **G15**：RNG state 类型必须与平台 torch 栈原生对象一致（会流出 TT 边界被
   `Generator.set_state()` 直接消费）。
6. **FALLBACK 收窄语义**：只适配"首参 callable 的加速壳"（`graph_capture(body)` →
   eager 执行 body）；不要当"转发到原生实现"用——原生壳往往就是要避开的平台专属路径。
7. **default_deny**：白名单管框架调用面，不接管 torch 内部；扫尾白名单
   `FRAMEWORK_PACKAGES` 是显式清单，新接框架加一项。
8. **激活时序证据制**：不猜 import 先后；激活末尾 `verify_sweep()` 按对象身份验收，
   残留即 fail fast。

## 通道 B：框架性能符号的后置绑定（provider/fn_binding.py）

torch.cuda policy（本层，cuda_compat）管功能域收束；**框架内部的性能符号**
（megatron 的 swiglu/rope 等）走 **`ttorch/provider/fn_binding.py`**——注意归属：
性能替换的执行（含机制）归 provider（§8：Bridge 不负责模块性能替换），bridge
只管 torch.cuda.*。机制：activate 时目标模块还不存在，装 import 钩子在目标模块
`exec_module` 完成瞬间绑定义点（后续 from-import 自然拿新实现；已在场直接绑 +
扫尾兜底）。**绑定目标是 FunctionDispatcher 而非厂商实现**——透明分发到
manager.call，数值判责全在 provider 门禁（fp32 轨道 fwd+bwd）。已落地：
swiglu/bias_swiglu_impl（门禁通过）；rope 被 bwd 门禁拒（厂商 kernel 不等价，
见 provider skill 的 vendor-stack.md）。

## 验证（改完必跑）

```bash
cd tt-bridge-demo && python3 demo_bridge.py        # mock 全链路（无卡可跑）
# 真实环境：
python3 -c "import ttorch; ttorch.activate(); print(ttorch.check_bindings())"
```

- `ttorch.check_bindings()`：激活期逐条 admission（绑定指向/语义保持）。
- `ttorch.assert_bindings_intact()`：运行期篡改检测（import 触发的后置替换会覆盖门面，
  靠它检出；篡改记 provenance `tampered`）。
- 端到端链路证据：`tests/e2e/real_qwen_smoke.py` 的中转报告（call_trace；调用次数是
  下界，from-import 调用点不经探针）。
