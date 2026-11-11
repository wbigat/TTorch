# big_leaf：连续存储 + 自持有大叶子的 MoE GroupedLinear 加速（生产级实现）
#
# 原理（实证链见 RESULTS.md §6-§8）：
#   LW=1 的收益不在 kernel，而在 host（每步 weight{i} 收集/noop_cat 探测）、
#   backward 梯度拆分、optimizer 参数数三处。big_leaf 把 G 个专家权重 repack
#   到一块连续 buffer，自造 plain-tensor 大叶子直喂 kernel：
#   - 非 Parameter → 不进 kernel pin 路径（避免原地覆写污染共享存储）；
#   - autograd leaf → dw 一次 accumulate 进 big_leaf.grad，零拆分；
#   - optimizer 只见 big_leaf → 参数数开销归零；
#   - weight{i} 注册/state_dict 键/TP 属性完全不变（.data 重绑为 buf 切片）。
#
# 用法：
#   plan = apply_big_leaf(model)                    # 建模后、optimizer 构建前
#   params = plan.optimizer_params(model)           # 替代 model.parameters()
#   opt = torch.optim.AdamW(params, lr=...)         # 任意逐元素 optimizer
#   ...
#   plan.restore()                                  # 回滚（A/B 对比/退场）
#
# 注意（生产集成边界，RESULTS.md §7.7）：
#   - Megatron dist-opt 必须使用 .megatron.BigLeafMegatronPlan 的
#     canonical layout + main_grad flush，不能把本模块单独 apply 后直接交给 DDP；
#   - optimizer ckpt 转换见 .checkpoint.py；dist-opt 推荐 fully-reshardable；
#   - `load_state_dict(..., assign=True)` 会替换 weight{i}.data 的 storage 对象，
#     破坏 repack——禁止 assign=True（默认 copy_ 原地路径已验证安全，见全链路验证）。
import contextlib
import copy
import dataclasses
import functools
import hashlib
import inspect
import os
import types
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformer_engine.pytorch.env_config import xte_env
from transformer_engine.pytorch.module.grouped_linear import (
    GroupedLinear,
    MoeFcFusionImpl,
    _Linear,
    cast_if_needed,
)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


@dataclasses.dataclass
class _LeafBinding:
    module: GroupedLinear
    buf: torch.Tensor                        # 权重连续存储本体
    bbuf: Optional[torch.Tensor]             # bias 连续存储（无 bias 为 None）
    big: torch.Tensor                        # 权重 big_leaf（plain leaf）
    big_b: Optional[torch.Tensor]            # bias big_leaf
    forward_mode: str = "switch"            # switch=厂商分支；patch=独立 forward
    megatron_wrapper: bool = False          # 保留 Megatron TEGroupedLinear 返回包装
    original_forward: Optional[object] = None
    guard_hook: Optional[object] = None
    plan: Optional[object] = None
    # 注意：不持有 repack 前的旧 storage（否则权重显存常驻 2×，D1）——
    # restore 时按需新分配。


@dataclasses.dataclass
class BigLeafPlan:
    bindings: List[_LeafBinding]
    model: Optional[torch.nn.Module] = None
    forward_mode: str = "switch"
    _lora_originals: Dict[torch.nn.Module, Tuple[object, object]] = dataclasses.field(
        default_factory=dict, init=False)
    _load_state_dict_originals: Dict[torch.nn.Module, object] = dataclasses.field(
        default_factory=dict, init=False)
    _active: bool = dataclasses.field(default=True, init=False)
    _broken: bool = dataclasses.field(default=False, init=False)

    @property
    def leafs(self) -> List[torch.Tensor]:
        """optimizer 应更新的实体（每模块 1 个，含 bias 时 2 个）。"""
        out = []
        for b in self.bindings:
            out.append(b.big)
            if b.big_b is not None:
                out.append(b.big_b)
        return out

    def grad_tensors(self) -> List[torch.Tensor]:
        """grad clipping / norm 统计用（语义等价于合并范数，套件 C6 已证）。"""
        return self.leafs

    def optimizer_params(self, model: torch.nn.Module) -> List[torch.Tensor]:
        """构造 optimizer 参数列表：非 GroupedLinear 参数原样 + big_leaf 们。

        weight{i}/bias{i} 被排除（它们不进 autograd 图、无 grad、由 big_leaf
        共享存储原地更新）。
        """
        gl_param_ids = set()
        for b in self.bindings:
            m = b.module
            for i in range(m.num_gemms):
                gl_param_ids.add(id(getattr(m, f"weight{i}")))
                if m.use_bias:
                    gl_param_ids.add(id(getattr(m, f"bias{i}")))
        others = [p for p in model.parameters() if id(p) not in gl_param_ids]
        return others + self.leafs

    def named_leafs(self, model: torch.nn.Module) -> List[tuple]:
        """(名字, leaf) 列表——对接按名匹配的 optimizer 分组规则（muon/no-wd 等）。

        名字取模块的限定名 + `.large_weight`/`.large_bias`。
        """
        mod_names = {id(m): n for n, m in model.named_modules()}
        out = []
        for b in self.bindings:
            base = mod_names.get(id(b.module), "?")
            out.append((f"{base}.large_weight" if base else "large_weight", b.big))
            if b.big_b is not None:
                out.append((f"{base}.large_bias" if base else "large_bias", b.big_b))
        return out

    def named_optimizer_params(self, model: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
        """返回 optimizer 实体的稳定名称，供 checkpoint 与按名分组共同使用。"""
        gl_param_ids = {id(p) for p, _ in self._expert_params()}
        others = [(n, p) for n, p in model.named_parameters() if id(p) not in gl_param_ids]
        return others + self.named_leafs(model)

    def _expert_params(self) -> Iterable[Tuple[torch.nn.Parameter, Tuple[int, str]]]:
        for bi, b in enumerate(self.bindings):
            for kind in ("weight", "bias"):
                if kind == "bias" and b.bbuf is None:
                    continue
                for i in range(b.module.num_gemms):
                    yield getattr(b.module, f"{kind}{i}"), (bi, kind)

    def optimizer_param_groups(self, model: torch.nn.Module,
                               param_groups: List[dict]) -> List[dict]:
        """把 stock 参数组映射为 big_leaf 参数组，保留每组超参数。

        同一权重/ bias 的 G 个原参数必须同组，否则合并后会改变 lr/wd/betas 等
        语义，直接拒绝。weight 与 bias 可以分别位于不同组（例如 no-wd bias）。
        """
        source_to_target = {}
        target_sources = {}
        for p, key in self._expert_params():
            bi, kind = key
            target = self.bindings[bi].big if kind == "weight" else self.bindings[bi].big_b
            source_to_target[id(p)] = (key, target)
            target_sources.setdefault(key, []).append(p)

        param_group_index = {}
        for gi, group in enumerate(param_groups):
            for p in group["params"]:
                if id(p) in param_group_index:
                    raise ValueError(f"参数重复出现在 optimizer groups: {param_group_index[id(p)]}, {gi}")
                param_group_index[id(p)] = gi

        skipped_frozen = set()
        for key, sources in target_sources.items():
            groups = {param_group_index.get(id(p)) for p in sources}
            if groups == {None} and not any(p.requires_grad for p in sources):
                skipped_frozen.add(key)
                continue
            if None in groups or len(groups) != 1:
                bi, kind = key
                raise ValueError(
                    f"{self.bindings[bi].module} 的 {kind} 专家未全部位于同一 optimizer group；"
                    "合并会改变分组超参数语义，请保持同组或放弃 big_leaf")

        out = []
        inserted = set()
        for group in param_groups:
            params = []
            for p in group["params"]:
                mapped = source_to_target.get(id(p))
                if mapped is None:
                    params.append(p)
                    continue
                key, target = mapped
                if key not in inserted:
                    params.append(target)
                    inserted.add(key)
            new_group = copy.copy(group)
            new_group["params"] = params
            out.append(new_group)

        expected = set(target_sources) - skipped_frozen
        if inserted != expected:
            missing = expected - inserted
            raise ValueError(f"optimizer groups 缺少 big_leaf 所需原参数: {sorted(missing)}")
        return out

    @contextlib.contextmanager
    def external_weight_update(self):
        """保护 safe-merge 等 ``param.data = new_tensor`` 形式的外部更新。

        退出时先把外部新值写回连续 buf，再恢复视图绑定；异常路径同样执行，
        避免 LoRA 标记为 merged 但 big_leaf 仍读旧值。
        """
        try:
            yield
        finally:
            try:
                for b in self.bindings:
                    self._repair_external_rebind(b, "weight")
                    if b.bbuf is not None:
                        self._repair_external_rebind(b, "bias")
                self.assert_healthy()
            except Exception:
                self._broken = True
                raise

    @staticmethod
    def _repair_external_rebind(b: _LeafBinding, kind: str) -> None:
        buf = b.buf if kind == "weight" else b.bbuf
        for i in range(b.module.num_gemms):
            p = getattr(b.module, f"{kind}{i}")
            if p.data.data_ptr() == buf[i].data_ptr():
                continue
            if (p.shape != buf[i].shape or p.dtype != buf[i].dtype
                    or p.device != buf[i].device):
                raise RuntimeError(
                    f"{b.module}: {kind}{i} 外部重绑改变 shape/dtype/device，"
                    "无法安全写回 big_leaf；请重建 plan")
            with torch.no_grad():
                buf[i].copy_(p.data)
                p.data = buf[i]

    def _patch_lora_merges(self) -> None:
        if self.model is None:
            return
        binding_modules = {id(b.module) for b in self.bindings}
        for module in self.model.modules():
            get_base = getattr(module, "get_base_layer", None)
            if get_base is None:
                continue
            try:
                base = get_base()
            except Exception:
                continue
            if id(base) not in binding_modules:
                continue
            originals = []
            for name in ("merge", "unmerge"):
                original = getattr(module, name, None)
                if original is None:
                    originals.append(None)
                    continue

                @functools.wraps(original)
                def guarded(*args, __original=original, **kwargs):
                    with self.external_weight_update():
                        return __original(*args, **kwargs)

                setattr(module, name, guarded)
                originals.append(original)
            self._lora_originals[module] = tuple(originals)

    def _restore_lora_merges(self) -> None:
        for module, originals in self._lora_originals.items():
            for name, original in zip(("merge", "unmerge"), originals):
                if original is not None:
                    setattr(module, name, original)
                elif name in module.__dict__:
                    delattr(module, name)
        self._lora_originals.clear()

    def _patch_load_state_dict_for(self, target: torch.nn.Module) -> None:
        if target in self._load_state_dict_originals:
            return
        original = target.load_state_dict
        self._load_state_dict_originals[target] = original

        @functools.wraps(original)
        def checked_load(*args, **kwargs):
            bound = inspect.signature(original).bind_partial(*args, **kwargs)
            if bound.arguments.get("assign", False):
                raise ValueError("big_leaf 禁止 load_state_dict(assign=True)；请使用默认 copy_ 路径")
            result = original(*args, **kwargs)
            self.assert_healthy()
            return result

        target.load_state_dict = checked_load

    def _patch_load_state_dict(self) -> None:
        targets = ([self.model] if self.model is not None else []) + [b.module for b in self.bindings]
        ddp = getattr(self, "ddp", None)
        if ddp is not None:
            targets.append(ddp)
        for target in targets:
            self._patch_load_state_dict_for(target)

    def _restore_load_state_dict(self) -> None:
        for target, original in self._load_state_dict_originals.items():
            target.load_state_dict = original
        self._load_state_dict_originals.clear()

    def assert_healthy(self) -> None:
        """启动期/周期性健康断言（fail-fast，无静默回退）。"""
        if not self._active:
            return
        if self._broken:
            raise RuntimeError("big_leaf plan 在 restore 过程中失败，禁止继续训练；请重建模型")
        for b in self.bindings:
            _check_binding_runtime(b)
            m = b.module
            G = m.num_gemms
            for i in range(G):
                w = getattr(m, f"weight{i}")
                _require(w.data.data_ptr() == b.buf[i].data_ptr(),
                         f"{m}: weight{i} 已脱离连续 buffer（repack 被破坏）")
                if b.bbuf is not None:
                    _require(getattr(m, f"bias{i}").data.data_ptr() == b.bbuf[i].data_ptr(),
                             f"{m}: bias{i} 已脱离连续 buffer")
            if b.forward_mode == "switch":
                _require(m._big_leaf_guard_binding is b and b.guard_hook is not None,
                         f"{m}: 运行期状态检查 hook 被移除")
                _require(m.large_weight is b.big, f"{m}: large_weight 被改写")
                _require(m.large_weight_switch, f"{m}: large_weight_switch 被关")
                if b.big_b is not None:
                    _require(m.large_bias is b.big_b, f"{m}: large_bias 被改写")
                else:
                    _require(m.large_bias is None, f"{m}: 无 bias 形态 large_bias 必须为 None")
            else:
                _require(not m.large_weight_switch, f"{m}: patch 模式不得打开私有 switch")
                _require(m._big_leaf_binding is b, f"{m}: patch binding 被改写")
                expected_forward = _megatron_big_leaf_forward if b.megatron_wrapper \
                    else _big_leaf_forward
                _require(getattr(m.forward, "__func__", None) is expected_forward,
                         f"{m}: 独立 forward 被替换")
            _require(b.big.is_leaf and b.big.data_ptr() == b.buf.data_ptr(),
                     f"{m}: big_leaf 非叶子或已脱离 buf")
            if b.big_b is not None:
                _require(b.big_b.is_leaf and b.big_b.data_ptr() == b.bbuf.data_ptr(),
                         f"{m}: bias big_leaf 非叶子或已脱离 bbuf")

    def restore(self) -> None:
        """幂等回滚到 stock；先完成全部分配，再提交重绑，保留训练更新。

        restore 后旧 optimizer 持有的 big_leaf 引用失效，必须重建 optimizer。
        已 restore 后再次调用不会触碰之后的 stock 训练结果。
        """
        if not self._active:
            return
        if self._broken:
            raise RuntimeError("big_leaf plan 此前 restore 失败，禁止重复回滚；请重建模型")

        prepared = []
        for b in self.bindings:
            replacements = []
            for i in range(b.module.num_gemms):
                replacements.append((getattr(b.module, f"weight{i}"), b.buf[i].clone()))
                if b.bbuf is not None:
                    replacements.append((getattr(b.module, f"bias{i}"), b.bbuf[i].clone()))
            prepared.append((b, replacements))

        try:
            with torch.no_grad():
                for b, replacements in prepared:
                    for param, value in replacements:
                        param.data = value
                    _uninstall_forward(b)
        except Exception:
            self._broken = True
            raise

        self._restore_lora_merges()
        self._restore_load_state_dict()
        if self.model is not None and "_big_leaf_plan" in self.model.__dict__:
            delattr(self.model, "_big_leaf_plan")
        self._active = False


def _check_binding_runtime(binding: _LeafBinding) -> None:
    """检测 plan 失败、动态冻结与新增原参数 hook，避免静默绕过。"""
    if binding.plan is not None and binding.plan._broken:
        raise RuntimeError("big_leaf plan 已处于 broken 状态，禁止继续 forward")
    module = binding.module
    for kind in (("weight", "bias") if module.use_bias else ("weight",)):
        params = [getattr(module, f"{kind}{i}") for i in range(module.num_gemms)]
        expected = params[0].requires_grad
        leaf = binding.big if kind == "weight" else binding.big_b
        for p in params:
            if p.requires_grad != expected:
                raise RuntimeError(f"{module}: 不允许 apply 后形成混合冻结；请重建 plan")
            if (getattr(p, "_backward_hooks", None)
                    or getattr(p, "_post_accumulate_grad_hooks", None)):
                raise RuntimeError(
                    f"{module}: apply 后新增原参数 hook 不会执行；请先 restore 再迁移 hook")
        if leaf.requires_grad != expected:
            raise RuntimeError(
                f"{module}: 检测到 apply 后动态冻结/解冻；当前 plan 不支持，请重建 plan")


def _validate_repack_module(mod: GroupedLinear, allow_module_hooks: bool = False,
                            allow_main_grad: bool = False) -> None:
    if not allow_module_hooks:
        for hook_attr in ("_forward_pre_hooks", "_forward_hooks",
                          "_backward_pre_hooks", "_backward_hooks"):
            if getattr(mod, hook_attr, None):
                raise ValueError(f"{mod}: 已有 {hook_attr}，请先迁移或移除后再 apply")
    if mod.wgrad_store.delay_wgrad_compute():
        raise ValueError(f"{mod}: big_leaf 暂不支持 delayed-wgrad，请先关闭该配置")
    if not mod.use_bias and getattr(mod, "return_bias", False):
        raise ValueError(f"{mod}: return_bias=True 且无 bias 的形态暂不支持")
    if getattr(mod, "primary_weights_in_fp8", False) or getattr(mod, "fp8", False):
        raise ValueError(f"{mod}: big_leaf 当前仅支持未量化 Parameter，FP8 需独立适配")
    for kind in (("weight", "bias") if mod.use_bias else ("weight",)):
        params = [getattr(mod, f"{kind}{i}") for i in range(mod.num_gemms)]
        first = params[0]
        for p in params:
            if p.grad is not None or (getattr(p, "main_grad", None) is not None
                                      and not allow_main_grad):
                raise ValueError(
                    f"{mod}: apply 前已有 grad/main_grad（可能已 backward 或接入 DDP）；"
                    "请在 DDP 前且无累积梯度时 apply")
            if not p.is_floating_point():
                raise ValueError(f"{mod}: big_leaf 不支持非浮点/量化参数 dtype={p.dtype}")
            if p.shape != first.shape or p.dtype != first.dtype or p.device != first.device:
                raise ValueError(f"{mod}: {kind} 专家参数 shape/dtype/device 不一致")
            if p.requires_grad != first.requires_grad:
                raise ValueError(f"{mod}: 不支持同一 {kind} 内混合冻结/可训专家")
            if (getattr(p, "_backward_hooks", None)
                    or getattr(p, "_post_accumulate_grad_hooks", None)):
                raise ValueError(f"{mod}: 原 weight/bias 上已有 per-param hook，请先迁移到 big_leaf")


def _repack_one(mod: GroupedLinear) -> _LeafBinding:
    _validate_repack_module(mod)
    G = mod.num_gemms
    ws = [getattr(mod, f"weight{i}") for i in range(G)]
    bs = [getattr(mod, f"bias{i}") for i in range(G)] if mod.use_bias else []
    # 先完成本模块全部分配与复制；OOM/复制失败时原 Parameter 尚未改变。
    buf = torch.empty(G, *ws[0].shape, device=ws[0].device, dtype=ws[0].dtype)
    bbuf = torch.empty(G, *bs[0].shape, device=bs[0].device, dtype=bs[0].dtype) if bs else None
    with torch.no_grad():
        for i, w in enumerate(ws):
            buf[i].copy_(w)
        if bs:
            for i, bias in enumerate(bs):
                bbuf[i].copy_(bias)

    big = buf.detach().requires_grad_(ws[0].requires_grad)
    big_b = bbuf.detach().requires_grad_(bs[0].requires_grad) if bs else None
    originals = [(p, p.data) for p in ws + bs]
    try:
        with torch.no_grad():
            for i, w in enumerate(ws):
                w.data = buf[i]
            for i, bias in enumerate(bs):
                bias.data = bbuf[i]
    except Exception:
        with torch.no_grad():
            for p, original in originals:
                p.data = original
        raise
    return _LeafBinding(mod, buf, bbuf, big, big_b)


_SUPPORTED_FORWARD_SHA256 = "ccca37704ab411a516a7226dde22d718c13d3195aa14206ce115408353f46243"


def _assert_supported_vendor_forward() -> None:
    source = inspect.getsource(GroupedLinear.forward)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != _SUPPORTED_FORWARD_SHA256:
        raise RuntimeError(
            f"GroupedLinear.forward 源码已变化（{digest}），patch 模式需重新对拍；"
            "请使用 forward_mode='switch' 或更新独立 forward")


_SUPPORTED_MEGATRON_FORWARD_SHA256 = "8c81986769a44b610b48ef957ffd71031d1af7535f9bc302c904919b0b2593a6"
_MEGATRON_QUANT_CONTEXT = None


def _validate_patch_module(module: GroupedLinear) -> bool:
    """返回是否需要 Megatron 子类包装；未知子类 fail-fast。"""
    global _MEGATRON_QUANT_CONTEXT
    if type(module) is GroupedLinear:
        return False
    try:
        from megatron.core.extensions.transformer_engine import (
            TEGroupedLinear,
            _get_fp8_autocast_for_quant_params,
        )
    except ImportError as e:
        raise RuntimeError(f"未知 GroupedLinear 子类 {type(module)}，拒绝 patch") from e
    if not isinstance(module, TEGroupedLinear) or type(module).forward is not TEGroupedLinear.forward:
        raise RuntimeError(
            f"GroupedLinear 子类 {type(module)} 覆盖了 forward 或类型未知，拒绝 patch")
    source = inspect.getsource(TEGroupedLinear.forward)
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != _SUPPORTED_MEGATRON_FORWARD_SHA256:
        raise RuntimeError(f"Megatron TEGroupedLinear.forward 源码已变化（{digest}），拒绝 patch")
    _MEGATRON_QUANT_CONTEXT = _get_fp8_autocast_for_quant_params
    return True


def _validate_switch_module(module: GroupedLinear) -> None:
    if type(module) is GroupedLinear:
        return
    try:
        from megatron.core.extensions.transformer_engine import TEGroupedLinear
    except ImportError as e:
        raise RuntimeError(f"未知 GroupedLinear 子类 {type(module)}，拒绝 switch") from e
    if not isinstance(module, TEGroupedLinear) or type(module).forward is not TEGroupedLinear.forward:
        raise RuntimeError(
            f"GroupedLinear 子类 {type(module)} 覆盖了 forward 或类型未知，拒绝 switch")


def _big_leaf_forward(module: GroupedLinear, inp: torch.Tensor, m_splits: List[int],
                      is_first_microbatch: Optional[bool] = None):
    """不依赖 large_weight_switch 的独立 forward，只覆盖明确支持的 BF16 路径。"""
    if isinstance(inp, tuple) or os.environ.get("USE_MOE_INT8_QUANT", "0") == "1" \
            or os.environ.get("USE_MOE_SMOOTHQUANT", "0") == "1":
        raise RuntimeError("big_leaf patch forward 不支持 int8/smoothquant 输入")
    if len(m_splits) != module.num_gemms:
        raise ValueError("Number of splits should match number of GEMMs.")
    binding = module._big_leaf_binding
    _check_binding_runtime(binding)
    weight_tensor = binding.big
    bias_tensor = binding.big_b
    bias_tensor_list = ([bias_tensor[i] for i in range(module.num_gemms)]
                        if bias_tensor is not None else [None] * module.num_gemms)
    inp = module.prepare_forward(inp, num_gemms=module.num_gemms)
    try:
        if xte_env.get_bool("XTE_USE_GROUPED_FC_FUSION"):
            raise RuntimeError("big_leaf patch forward 不支持 XTE_USE_GROUPED_FC_FUSION=1")
        use_accel_bf16_fc = xte_env.get_bool("XMLIR_ENABLE_FAST_FC")
        use_fc_fusion = xte_env.get_bool("USE_CAST_FC_FUSION")
        is_accel_bf16 = (use_fc_fusion or use_accel_bf16_fc) and inp.dtype == torch.bfloat16
        if is_accel_bf16:
            m_splits_device = torch.tensor(m_splits).to(device=inp.device, non_blocking=True)
            has_bias = module.apply_bias and not module.gemm_bias_unfused_add
            bias = bias_tensor if has_bias else None

            def _call_grouped(x):
                if torch.is_grad_enabled():
                    return MoeFcFusionImpl.apply(
                        x, weight_tensor, m_splits_device, module.wgrad_store, bias, True)
                return MoeFcFusionImpl.forward(
                    None, x, weight_tensor, m_splits_device, module.wgrad_store, bias, True)

            if inp.ndim == 2:
                out = _call_grouped(inp)
            else:
                if inp.ndim < 2:
                    raise ValueError("input tensor must have at least two dimensions")
                non_feature_dims, feature_dim = inp.shape[:-1], inp.shape[-1]
                out = _call_grouped(inp.view(-1, feature_dim)).view(*non_feature_dims, -1)
        else:
            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                ctx_args = []
            else:
                linear_fn = _Linear.forward
                ctx_args = [None]
            non_tensor_args = (
                module.apply_bias and not module.gemm_bias_unfused_add,
                is_first_microbatch,
                False, False, module.wgrad_store,
                None, None, None, None, None, None, False, False, None, 1, False, False,
                module.activation_dtype, module.parallel_mode, torch.is_grad_enabled(),
                False, False, False, False, False, False, module.ub_name, False, None,
                module, None, None, False, False,
            )
            non_feature_dims, feature_dim = inp.shape[:-1], inp.shape[-1]
            weight_tensor_list = [weight_tensor[i] for i in range(module.num_gemms)]
            out = torch.cat([
                linear_fn(*(ctx_args + [w, i, b]), non_tensor_args)
                for i, w, b in zip(
                    torch.split(inp.view(-1, feature_dim), m_splits),
                    weight_tensor_list,
                    bias_tensor_list,
                )
            ])
            out = out.unflatten(0, non_feature_dims)
    finally:
        module.end_forward()

    if module.gemm_bias_unfused_add and out.numel() > 0:
        out_shape = out.shape
        out = torch.cat([
            o + cast_if_needed(b, module.activation_dtype)
            for o, b in zip(
                torch.split(out.view(-1, module.out_features), m_splits), bias_tensor_list)
        ]).view(out_shape)
    if module.return_bias:
        return out, [cast_if_needed(b, module.activation_dtype) for b in bias_tensor_list]
    return out


def _megatron_big_leaf_forward(module: GroupedLinear, x, m_splits):
    """保留 Megatron TEGroupedLinear 的 context 与固定二元返回约定。"""
    is_first = None if module.disable_parameter_transpose_cache else module.is_first_microbatch
    with _MEGATRON_QUANT_CONTEXT(module.te_quant_params, module.training):
        out = _big_leaf_forward(module, x, m_splits, is_first_microbatch=is_first)
    module.is_first_microbatch = False
    if module.te_return_bias:
        return out
    return out, None


def _switch_forward_guard(module, args) -> None:
    _check_binding_runtime(module._big_leaf_guard_binding)


def _install_forward(binding: _LeafBinding, mode: str) -> None:
    if mode not in ("switch", "patch"):
        raise ValueError(f"未知 forward_mode={mode}")
    module = binding.module
    binding.forward_mode = mode
    if mode == "switch":
        if xte_env.get_bool("XTE_USE_GROUPED_FC_FUSION"):
            raise ValueError("big_leaf switch 模式不支持 XTE_USE_GROUPED_FC_FUSION=1；请关闭或使用 patch")
        module._big_leaf_guard_binding = binding
        binding.guard_hook = module.register_forward_pre_hook(_switch_forward_guard)
        module.large_weight = binding.big
        module.large_bias = binding.big_b
        module.large_weight_switch = True
        return
    _assert_supported_vendor_forward()
    binding.megatron_wrapper = _validate_patch_module(module)
    binding.original_forward = module.forward
    module._big_leaf_binding = binding
    module.large_weight_switch = False
    for attr in ("large_weight", "large_bias"):
        if attr in module.__dict__:
            delattr(module, attr)
    forward_fn = _megatron_big_leaf_forward if binding.megatron_wrapper else _big_leaf_forward
    module.forward = types.MethodType(forward_fn, module)


def _uninstall_forward(binding: _LeafBinding) -> None:
    module = binding.module
    if binding.guard_hook is not None:
        binding.guard_hook.remove()
        binding.guard_hook = None
    if "_big_leaf_guard_binding" in module.__dict__:
        delattr(module, "_big_leaf_guard_binding")
    if binding.forward_mode == "patch":
        if binding.original_forward is not None:
            module.forward = binding.original_forward
            binding.original_forward = None
        if "_big_leaf_binding" in module.__dict__:
            delattr(module, "_big_leaf_binding")
    for attr in ("large_weight", "large_bias"):
        if attr in module.__dict__:
            delattr(module, attr)
    module.large_weight_switch = False


def apply_big_leaf(model: torch.nn.Module, *,
                   probe_input: Optional[torch.Tensor] = None,
                   forward_mode: str = "switch") -> BigLeafPlan:
    """对 model 内所有 GroupedLinear 应用 big_leaf 加速。

    时序约束（违反会被 assert_healthy 抓住，但请在正确时机调用）：
    必须在「model.to(device)/dtype 转换 + load 权重完成之后、optimizer/DDP
    构建之前」调用——`model.to()` 会替换 weight{i}.data 的 storage，
    使 big_leaf 与参数脱节（ptr 断言 fail-fast）。

    probe_input: 可选 (tokens, fin) 探测输入——提供时做一次前向烟测
    （验证 switch 直取通道在当前厂商栈版本存在且可达），失败即 raise。
    """
    if forward_mode not in ("switch", "patch"):
        raise ValueError(f"未知 forward_mode={forward_mode}")
    if forward_mode == "patch":
        _assert_supported_vendor_forward()
    gls = [m for m in model.modules() if isinstance(m, GroupedLinear)]
    if not gls:
        raise ValueError("model 内未发现 GroupedLinear 模块")
    if (getattr(model, "_big_leaf_plan", None) is not None
            or any(getattr(m, "large_weight_switch", False) for m in gls)):
        raise ValueError("模型已被 apply_big_leaf 处理过或处于 LW=1（不支持重复 apply；"
                         "请先 restore 或使用 LW=0 模型）")
    if forward_mode == "switch" and xte_env.get_bool("XTE_USE_GROUPED_FC_FUSION"):
        raise ValueError("big_leaf switch 模式不支持 XTE_USE_GROUPED_FC_FUSION=1")

    # 所有确定性校验先于任何参数修改；从首次 repack 起即进入可回滚区。
    for module in gls:
        _validate_repack_module(module)
        if forward_mode == "patch":
            _validate_patch_module(module)
        else:
            _validate_switch_module(module)
    plan = BigLeafPlan([], model=model, forward_mode=forward_mode)
    try:
        for module in gls:
            binding = _repack_one(module)
            binding.plan = plan
            plan.bindings.append(binding)
        for binding in plan.bindings:
            _install_forward(binding, forward_mode)
        model._big_leaf_plan = plan
        plan.assert_healthy()
        plan._patch_lora_merges()
        plan._patch_load_state_dict()
        if probe_input is not None:
            b0 = plan.bindings[0]
            G = b0.module.num_gemms
            quotient, remainder = divmod(probe_input.shape[0], G)
            ms = [quotient + (i < remainder) for i in range(G)]
            snap = b0.buf[0].clone()
            with torch.no_grad():
                out = b0.module(probe_input, ms)
            if isinstance(out, tuple):
                out = out[0]
            _require(out.shape[0] == probe_input.shape[0], "probe 前向 shape 异常")
            _require(torch.equal(b0.buf[0], snap),
                     "probe 后权重被原地改写（厂商栈行为漂移，禁用本方案）")
    except Exception as original_error:
        try:
            plan.restore()
        except Exception as rollback_error:
            raise RuntimeError(f"apply 失败且回滚失败: {original_error}") from rollback_error
        raise
    return plan
