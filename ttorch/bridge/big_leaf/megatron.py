"""big_leaf 的 Megatron DDP / DistributedOptimizer 适配。

两阶段使用：
1. DDP 前 prepare + compute_full_param_layout，并把 layout 传给 DDP；
2. DDP 后 attach_ddp，建立 big_leaf 视图和 main_grad flush hook。

dist-opt 保留原专家 Parameter 作为 optimizer/checkpoint 逻辑实体；大叶子只负责
forward/backward 粒度，不替换 DistributedOptimizer 的 shard 参数列表。
"""
import dataclasses
from typing import Dict, List, Optional, Tuple

import torch
from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers
from megatron.core.optimizer.param_layout import (
    FullParamLayout,
    PerBufferParamLayout,
    pad_bucket_end,
    pad_param_start,
)
from transformer_engine.pytorch import GroupedLinear

from .core import (
    BigLeafPlan,
    _LeafBinding,
    _assert_supported_vendor_forward,
    _install_forward,
    _require,
    _uninstall_forward,
    _validate_patch_module,
    _validate_repack_module,
    _validate_switch_module,
)
from transformer_engine.pytorch.env_config import xte_env


def _validate_megatron_module(module: GroupedLinear, forward_mode: str,
                              allow_module_hooks: bool = False,
                              allow_main_grad: bool = False) -> None:
    _validate_repack_module(module, allow_module_hooks=allow_module_hooks,
                            allow_main_grad=allow_main_grad)
    config = getattr(module, "config", None)
    if (getattr(module, "delay_wgrad_compute", False)
            or getattr(config, "delay_wgrad_compute", False)
            or getattr(config, "overlap_dispatch_backward_with_experts_wgrad", False)):
        raise ValueError(f"{module}: big_leaf 暂不支持 delayed-wgrad/dispatch-wgrad overlap")
    if forward_mode == "patch":
        _validate_patch_module(module)
    else:
        _validate_switch_module(module)


@dataclasses.dataclass
class BigLeafMegatronPlan(BigLeafPlan):
    ddp: Optional[object] = None
    modules_to_bind: List[GroupedLinear] = dataclasses.field(default_factory=list)
    grad_accs: List[object] = dataclasses.field(default_factory=list)
    grad_hook_handles: List[object] = dataclasses.field(default_factory=list)
    main_grad_views: Dict[Tuple[int, str], Optional[torch.Tensor]] = dataclasses.field(
        default_factory=dict)
    flush_counts: Dict[Tuple[int, str], int] = dataclasses.field(default_factory=dict)

    @classmethod
    def prepare(cls, model: torch.nn.Module,
                forward_mode: str = "switch") -> "BigLeafMegatronPlan":
        if forward_mode not in ("switch", "patch"):
            raise ValueError(f"未知 forward_mode={forward_mode}")
        if getattr(model, "_big_leaf_plan", None) is not None:
            raise ValueError("模型已经存在 big_leaf plan；DDP 模式必须在 repack/DDP 前 prepare")
        if forward_mode == "patch":
            _assert_supported_vendor_forward()
        elif xte_env.get_bool("XTE_USE_GROUPED_FC_FUSION"):
            raise ValueError("big_leaf switch 模式不支持 XTE_USE_GROUPED_FC_FUSION=1")
        modules = []
        found_grouped = False
        for module in model.modules():
            if not isinstance(module, GroupedLinear):
                continue
            found_grouped = True
            if getattr(module, "large_weight_switch", False):
                raise ValueError("big_leaf DDP prepare 需要原生 LW=0 模型，不能接 LW=1")
            _validate_megatron_module(module, forward_mode)
            params = [p for p in module.parameters(recurse=False)]
            if not any(p.requires_grad for p in params):
                continue
            if not all(p.requires_grad for p in params):
                raise ValueError(
                    f"{module}: dist-opt 首版只支持整模块可训或整模块冻结，不支持部分参数冻结")
            modules.append(module)
        if not found_grouped:
            raise ValueError("model 内未发现 GroupedLinear 模块")
        return cls(bindings=[], model=model, forward_mode=forward_mode,
                   modules_to_bind=modules)

    def _owner_map(self):
        owners = {}
        for module in self.modules_to_bind:
            for kind in (("weight", "bias") if module.use_bias else ("weight",)):
                for i in range(module.num_gemms):
                    p = getattr(module, f"{kind}{i}")
                    owners[id(p)] = (module, kind, i)
        return owners

    def compute_full_param_layout(
        self,
        params: List[torch.nn.Parameter],
        bucket_size: Optional[int],
        data_parallel_world_size: int,
        ddp_config,
        expert_data_parallel_world_size: Optional[int] = None,
    ) -> FullParamLayout:
        if set(params) != set(self.model.parameters()):
            raise ValueError("compute_full_param_layout 的 params 必须恰好是 model.parameters()")
        if (getattr(ddp_config, "delay_wgrad_compute", False)
                or getattr(ddp_config, "overlap_dispatch_backward_with_experts_wgrad", False)):
            raise ValueError("big_leaf 不支持 delayed-wgrad/dispatch-wgrad overlap")
        if getattr(ddp_config, "external_cuda_graph", False):
            raise ValueError("big_leaf 暂不支持 external CUDA graph")
        if getattr(ddp_config, "num_distributed_optimizer_instances", 1) != 1:
            raise ValueError("big_leaf 暂不支持多 dist-opt instance")
        if getattr(ddp_config, "reduce_scatter_with_fp32_accumulation", False):
            raise ValueError("big_leaf 暂不支持 reduce_scatter_with_fp32_accumulation")

        owners = self._owner_map()
        trainable_params = [p for p in params if p.requires_grad]
        buffer_groups = group_params_for_buffers(
            trainable_params, ddp_config.grad_reduce_in_fp32)
        layouts = {}
        for buffer_key, (group_params, param_indices) in buffer_groups.items():
            if getattr(buffer_key, "is_managed_by_layer_wise_optimizer", False):
                raise ValueError("big_leaf 暂不支持 LayerWiseDistributedOptimizer layout")
            world_size = (expert_data_parallel_world_size or data_parallel_world_size) \
                if buffer_key.is_expert_parallel else data_parallel_world_size
            present = {id(p) for p in group_params}
            desired_physical = []
            emitted_modules = set()
            for p in reversed(group_params):
                owner = owners.get(id(p))
                if owner is None:
                    desired_physical.append(p)
                    continue
                module = owner[0]
                if id(module) in emitted_modules:
                    continue
                members = [q for q in group_params
                           if id(q) in owners and owners[id(q)][0] is module]
                members.sort(key=lambda q: (0 if owners[id(q)][1] == "weight" else 1,
                                            owners[id(q)][2]))
                _require(all(id(q) in present for q in members),
                         f"{module}: big_leaf 成员未全部进入 canonical layout")
                desired_physical.extend(members)
                emitted_modules.add(id(module))
            layouts[buffer_key] = self._atomic_group_layout(
                group_params, desired_physical, param_indices, bucket_size, world_size,
                ddp_config, owners)
        return FullParamLayout(layouts=layouts)

    @staticmethod
    def _atomic_group_layout(group_params, desired_physical, param_indices, bucket_size,
                             world_size, ddp_config, owners) -> PerBufferParamLayout:
        param_ranges = {}
        bucket_indices = []
        per_bucket_numel_unpadded = []
        bucket_params = set()
        bucket_start = 0
        bucket_id = 0
        current = 0

        def flush_bucket(end):
            nonlocal bucket_start, bucket_id, bucket_params
            per_bucket_numel_unpadded.append(end - bucket_start)
            padded_end = pad_bucket_end(
                end, world_size, ddp_config.pad_buckets_for_high_nccl_busbw)
            bucket_indices.append((bucket_start, padded_end))
            bucket_start = padded_end
            bucket_id += 1
            bucket_params = set()

        for index, p in enumerate(desired_physical):
            owner = owners.get(id(p))
            previous_same_group = False
            at_group_end = True
            if owner is not None:
                if index > 0:
                    previous = owners.get(id(desired_physical[index - 1]))
                    previous_same_group = previous is not None and previous[0] is owner[0]
                if index + 1 < len(desired_physical):
                    following = owners.get(id(desired_physical[index + 1]))
                    at_group_end = following is None or following[0] is not owner[0]

            is_shared_embedding = getattr(p, "shared_embedding", False)
            start = current if previous_same_group else pad_param_start(current)
            if is_shared_embedding and bucket_params:
                flush_bucket(start)
                start = bucket_start
            end = start + p.numel()
            param_ranges[p] = (start, end, bucket_id)
            bucket_params.add(p)
            current = end
            bucket_exceeds = bucket_size is not None and (current - bucket_start) >= bucket_size
            if at_group_end and (is_shared_embedding or bucket_exceeds):
                flush_bucket(current)
                current = bucket_start

        if bucket_params:
            flush_bucket(current)
        param_index_map = {p: param_ranges[p] for p in group_params}
        return PerBufferParamLayout(
            param_index_map=param_index_map,
            bucket_indices=bucket_indices,
            per_bucket_numel_unpadded=per_bucket_numel_unpadded,
            param_indices=list(param_indices),
        )

    def _find_buffer(self, param):
        matches = [b for b in self.ddp.buffers + self.ddp.expert_parallel_buffers
                   if param in b.param_index_map]
        if len(matches) != 1:
            raise RuntimeError(f"参数必须恰好属于一个 DDP buffer，实际 {len(matches)}")
        return matches[0]

    def _contiguous_buf(self, buffer, module, kind) -> torch.Tensor:
        params = [getattr(module, f"{kind}{i}") for i in range(module.num_gemms)]
        ranges = [buffer.param_index_map[p] for p in params]
        for i, p in enumerate(params):
            if i and ranges[i][0] != ranges[i - 1][1]:
                raise RuntimeError(f"{module}: {kind} 在 DDP layout 中不连续；请传入自定义 layout")
            start, end, _ = ranges[i]
            if end - start != p.numel():
                raise RuntimeError(f"{module}: {kind}{i} range 与参数 numel 不一致")
        flat = buffer.param_data
        start, end = ranges[0][0], ranges[-1][1]
        return flat[start:end].view(module.num_gemms, *params[0].shape)

    def _main_grad_view(self, params, big):
        first = params[0].main_grad
        offset = 0
        contiguous = True
        for p in params:
            expected = first.data_ptr() + offset * first.element_size()
            if p.main_grad.data_ptr() != expected:
                contiguous = False
                break
            offset += p.numel()
        if contiguous:
            try:
                return first.as_strided(big.shape, big.stride())
            except RuntimeError:
                pass
        return None

    def _register_flush(self, binding_index: int, kind: str, big, params, buffer):
        key = (binding_index, kind)
        main_view = self._main_grad_view(params, big)
        self.main_grad_views[key] = main_view
        self.flush_counts[key] = 0

        def flush(*unused):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("big_leaf 暂不支持 CUDA graph capture 中的 main_grad flush")
            if big.grad is None:
                raise RuntimeError(f"big_leaf {key} flush 时 grad=None")
            with torch.no_grad():
                if main_view is not None:
                    main_view.add_(big.grad)
                else:
                    for i, p in enumerate(params):
                        p.main_grad.add_(big.grad[i])
                big.grad = None
                for p in params:
                    p.grad = None
                    if self.ddp.ddp_config.overlap_grad_reduce:
                        self.ddp.param_to_bucket_group[p].register_grad_ready(
                            p, self.ddp.force_all_reduce)
            self.flush_counts[key] += 1

        grad_acc = big.expand_as(big).grad_fn.next_functions[0][0]
        handle = grad_acc.register_hook(flush)
        self.grad_accs.append(grad_acc)
        self.grad_hook_handles.append(handle)

    def attach_ddp(self, ddp) -> "BigLeafMegatronPlan":
        if not self._active:
            raise ValueError("plan 已 restore，不能重复 attach；请重新 prepare")
        if self.bindings:
            raise ValueError("重复 attach_ddp")
        if ddp.module is not self.model:
            raise ValueError("ddp.module 与 prepare 的模型不一致")
        if not ddp.ddp_config.use_distributed_optimizer:
            raise ValueError("本适配器要求 use_distributed_optimizer=True")
        self.ddp = ddp
        try:
            for module in self.modules_to_bind:
                _validate_megatron_module(
                    module, self.forward_mode,
                    allow_module_hooks=True, allow_main_grad=True)
                weight0 = module.weight0
                buffer = self._find_buffer(weight0)
                params = [p for p in module.parameters(recurse=False)]
                if any(self._find_buffer(p) is not buffer for p in params):
                    raise RuntimeError(f"{module}: weight/bias 跨 buffer，无法构造 big_leaf")
                buf = self._contiguous_buf(buffer, module, "weight")
                bbuf = self._contiguous_buf(buffer, module, "bias") if module.use_bias else None
                big = buf.detach().requires_grad_(True)
                big_b = bbuf.detach().requires_grad_(True) if bbuf is not None else None
                binding = _LeafBinding(module=module, buf=buf, bbuf=bbuf, big=big, big_b=big_b)
                binding.plan = self
                self.bindings.append(binding)
                _install_forward(binding, self.forward_mode)
                bi = len(self.bindings) - 1
                weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
                self._register_flush(bi, "weight", big, weights, buffer)
                if big_b is not None:
                    biases = [getattr(module, f"bias{i}") for i in range(module.num_gemms)]
                    self._register_flush(bi, "bias", big_b, biases, buffer)
            self.model._big_leaf_plan = self
            self.assert_healthy()
            self._patch_lora_merges()
            self._patch_load_state_dict()
        except Exception:
            self.restore()
            raise
        return self

    def optimizer_params(self, model):
        raise RuntimeError(
            "dist-opt 模式必须保留 DDP 原参数并调用 get_megatron_optimizer；"
            "不能使用非 dist-opt 的 plan.optimizer_params()")

    def assert_healthy(self):
        super().assert_healthy()
        if self.ddp is not None:
            for key, view in self.main_grad_views.items():
                if view is None:
                    continue
                binding = self.bindings[key[0]]
                big = binding.big if key[1] == "weight" else binding.big_b
                _require(view.shape == big.shape, f"{key}: main_grad 大视图 shape 漂移")

    def restore(self):
        """幂等摘除 big_leaf，不破坏 DDP 已建立的参数视图和 optimizer。"""
        if not self._active:
            return
        if self._broken:
            raise RuntimeError("big_leaf Megatron plan 此前 restore 失败，请重建模型")
        try:
            for handle in self.grad_hook_handles:
                if hasattr(handle, "remove"):
                    handle.remove()
            for binding in self.bindings:
                _uninstall_forward(binding)
        except Exception:
            self._broken = True
            raise
        self.grad_hook_handles.clear()
        self.grad_accs.clear()
        self._restore_lora_merges()
        self._restore_load_state_dict()
        self.bindings.clear()
        self.main_grad_views.clear()
        if self.model is not None and "_big_leaf_plan" in self.model.__dict__:
            delattr(self.model, "_big_leaf_plan")
        self._active = False
