"""big_leaf 的 Megatron DDP / DistributedOptimizer 适配（双线实体版）。

两阶段使用：
1. DDP 前 prepare（构造内部线实体）+ compute_full_param_layout，并把 layout 传给 DDP；
2. DDP 后 attach_ddp，把外部线 weight{i}/bias{i} 重绑为内部线实体的存储切片视图。

**双线设计（2026-11-19，§13 绩效回收）**：

- 外部线 = 模块注册的 `weight{i}`/`bias{i}`（`state_dict` 键、查询、HF/模型 ckpt
  导出全部无感知，读到的永远是内部线存储的最新值）；
- 内部线 = 每个 GroupedLinear 一个 `big`/`big_b` 实体（独立 `nn.Parameter`，
  **不注册进任何 module**）。它同时是：
  DDP buffer 的参数实体（梯度整体 accumulate、reduce-scatter 一次完成）、
  DistributedOptimizer 的参数实体（Adam 状态按大实体建，shard 数比
  per-weight{i} 少 G 倍）——optimizer 单实体与 DDP 元数据缩减两项收益由此回收。
- 两线共享同一 param_data 存储：DDP 构造期间经 `entity_swap()` 临时让
  GroupedLinear 的 `parameters()`/`named_parameters()` 产出内部线实体（mcore
  的 DDP 一致性断言与 `get_megatron_optimizer` 都从 `module.named_parameters()`
  收集参数，2026-11-19 核实源码），边界调用结束后恢复——外部世界零感知。
- 实体必须是 `nn.Parameter` 而非 plain tensor：mcore buffer 用
  `param.data = 视图` 重绑存储（`param_and_grad_buffer.py`），只有 Parameter 的
  `.data` setter 才真正换 storage。它不注册进任何 module，因此不会进入 TE 的
  kernel pin 路径（pin 遍历模块 `_parameters`），原 plain-tensor 的安全理由保持。

与旧版（per-weight{i} 实体 + main_grad flush）相比：
- 删除 flush hook 全套（big.grad 走标准 DDP 路径，overlap_grad_reduce 原样兼容）；
- `restore()` 语义收紧：只摘 forward 与守卫，`weight{i}` 保持 param_data 视图
  （值持续最新）；**restore 后不能继续训练**（DDP 梯度实体是内部线，重建需
  新进程/新 DDP——与生产回滚策略"优先新进程"一致）。
- optimizer checkpoint 状态改按内部线实体保存；旧 per-weight{i} 格式的迁移见
  `.checkpoint.convert_distopt_state_layout()`。

**冻结模块（full/LoRA base）**：`prepare()` 把整模块冻结的 GroupedLinear 归入
`frozen_modules`，`attach_ddp` 用独立 buf（`_repack_one`）绑定，只拿 forward
收益（forward 直取 + 免 cat），不进 DDP buffer / optimizer（无梯度、无 Adam
状态）。LoRA base 即此形态；restore 时冻结权重脱离独立 buf 视图（克隆回独立
存储，避免悬空 storage）。
"""
import contextlib
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
    _repack_one,
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
    # 整模块冻结的 GroupedLinear：不进 DDP buffer / optimizer（无梯度），只用
    # 独立 buf 绑定拿 forward 收益（forward 直取 + 免 cat）。LoRA base 即此形态。
    frozen_modules: List[GroupedLinear] = dataclasses.field(default_factory=list)
    # 内部线实体：module id -> (big, big_b)。独立 nn.Parameter，不注册进 module；
    # prepare 时把 G 份权重值搬入（唯一一次真拷贝），DDP 构造时再被 buffer 重绑
    # 到 param_data 视图，旧存储随即释放。
    proxies: Dict[int, Tuple[torch.nn.Parameter, Optional[torch.nn.Parameter]]] = \
        dataclasses.field(default_factory=dict)

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
        frozen_modules = []
        found_grouped = False
        for module in model.modules():
            if not isinstance(module, GroupedLinear):
                continue
            found_grouped = True
            if getattr(module, "large_weight_switch", False):
                raise ValueError("big_leaf DDP prepare 需要原生 LW=0 模型，不能接 LW=1")
            params = [p for p in module.parameters(recurse=False)]
            frozen = not any(p.requires_grad for p in params)
            if not frozen and not all(p.requires_grad for p in params):
                raise ValueError(
                    f"{module}: 只支持整模块可训或整模块冻结，不支持部分参数冻结")
            _validate_megatron_module(module, forward_mode,
                                      allow_module_hooks=frozen)
            (frozen_modules if frozen else modules).append(module)
        if not found_grouped:
            raise ValueError("model 内未发现 GroupedLinear 模块")
        plan = cls(bindings=[], model=model, forward_mode=forward_mode,
                   modules_to_bind=modules, frozen_modules=frozen_modules)
        plan._build_proxies()
        return plan

    def _build_proxies(self) -> None:
        """构造内部线实体并把 G 份权重值搬入（DDP 会按 param.data 拷贝进 param_data）。"""
        for module in self.modules_to_bind:
            w0 = module.weight0.data
            big = torch.nn.Parameter(
                w0.new_empty((module.num_gemms, *w0.shape)), requires_grad=True)
            with torch.no_grad():
                for i in range(module.num_gemms):
                    big[i].copy_(getattr(module, f"weight{i}").data)
            big.allreduce = getattr(module.weight0, "allreduce", True)
            if module.use_bias:
                b0 = module.bias0.data
                big_b = torch.nn.Parameter(
                    b0.new_empty((module.num_gemms, *b0.shape)), requires_grad=True)
                with torch.no_grad():
                    for i in range(module.num_gemms):
                        big_b[i].copy_(getattr(module, f"bias{i}").data)
                big_b.allreduce = getattr(module.bias0, "allreduce", True)
            else:
                big_b = None
            self.proxies[id(module)] = (big, big_b)

    def entities(self) -> List[torch.nn.Parameter]:
        """DDP / dist-opt 可见的参数实体：非 GroupedLinear 可训参数原样 + 内部线实体。

        外部线 weight{i}/bias{i} 被排除（不进 autograd、无 main_grad，由内部线
        共享存储原地更新）。layout 与 `entity_swap()` 的产出必须与此集合一致。
        """
        expert_ids = set()
        for module in self.modules_to_bind:
            for kind in (("weight", "bias") if module.use_bias else ("weight",)):
                for i in range(module.num_gemms):
                    expert_ids.add(id(getattr(module, f"{kind}{i}")))
        others = [p for p in self.model.parameters()
                  if id(p) not in expert_ids and p.requires_grad]
        leafs = []
        for module in self.modules_to_bind:
            big, big_b = self.proxies[id(module)]
            leafs.append(big)
            if big_b is not None:
                leafs.append(big_b)
        return others + leafs

    @contextlib.contextmanager
    def entity_swap(self):
        """临时把 GroupedLinear 注册字典里的 weight{i}/bias{i} 换成内部线实体。

        mcore 的 DDP 参数收集（`distributed_data_parallel.py` 的 all_params 与
        grad hook 注册）与 `get_megatron_optimizer` 的 param_groups 收集都从
        `module.named_parameters()` 读参数，而 `nn.Module.named_parameters()`
        的递归实现**直读子模块的 `_parameters` 字典**、不调用子模块同名方法
        （2026-11-19 实测：方法级覆盖完全无效）——因此换装必须发生在
        `_parameters` 层，这正是 TE LW=1"天然生效"的原因。DDP 的一致性断言
        要求 layout 键 == 该集合，换装窗口必须覆盖这两个边界调用的全过程。

        窗口内 `_parameters` 不含 weight{i}（若被访问会 AttributeError），
        但 DDP 构造与 optimizer 构造都只做泛型参数遍历；`state_dict()` 只在
        窗口外被调用，外部查询/导出/键契约零影响。窗口外恢复原字典。
        """
        saved = []
        for module in self.modules_to_bind:
            big, big_b = self.proxies[id(module)]
            orig_params = module._parameters
            swapped = {}
            for name, p in orig_params.items():
                if name.startswith("weight") or name.startswith("bias"):
                    continue
                swapped[name] = p
            swapped["weight"] = big
            if big_b is not None:
                swapped["bias"] = big_b
            module._parameters = swapped
            saved.append((module, orig_params))
        try:
            yield
        finally:
            for module, orig_params in saved:
                module._parameters = orig_params

    def _owner_map(self):
        owners = {}
        for module in self.modules_to_bind:
            for kind in (("weight", "bias") if module.use_bias else ("weight",)):
                big = self.proxies[id(module)][0 if kind == "weight" else 1]
                owners[id(big)] = (module, kind, 0)
        return owners

    def compute_full_param_layout(
        self,
        params: List[torch.nn.Parameter],
        bucket_size: Optional[int],
        data_parallel_world_size: int,
        ddp_config,
        expert_data_parallel_world_size: Optional[int] = None,
    ) -> FullParamLayout:
        if set(params) != set(self.entities()):
            raise ValueError("compute_full_param_layout 的 params 必须恰好是 plan.entities()")
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
                         f"{module}: big_leaf 实体未全部进入 canonical layout")
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
                big, big_b = self.proxies[id(module)]
                weights = [getattr(module, f"weight{i}")
                           for i in range(module.num_gemms)]
                _require(big.data.shape == (module.num_gemms, *weights[0].shape),
                         f"{module}: 内部线实体形状漂移")
                buffer = self._find_buffer(big)
                if big_b is not None:
                    _require(self._find_buffer(big_b) is buffer,
                             f"{module}: weight/bias 实体跨 buffer，无法构造 big_leaf")
                    _require(big_b.data.shape == (
                        module.num_gemms, *getattr(module, "bias0").shape),
                        f"{module}: bias 实体形状漂移")
                # 外部线重绑：weight{i}/bias{i}.data → 内部线实体存储的切片视图。
                # DDP 构造期间已把实体值拷进 param_data 并把实体 .data 重绑到
                # param_data 视图，因此这里既拿值又共享存储。
                for i, w in enumerate(weights):
                    w.data = big.data[i]
                if big_b is not None:
                    for i in range(module.num_gemms):
                        p = getattr(module, f"bias{i}")
                        p.data = big_b.data[i]
                binding = _LeafBinding(
                    module=module,
                    buf=big.data,
                    bbuf=big_b.data if big_b is not None else None,
                    big=big,
                    big_b=big_b,
                )
                binding.plan = self
                self.bindings.append(binding)
                _install_forward(binding, self.forward_mode)
            # 冻结模块：独立 buf 绑定，只拿 forward 收益（不进 DDP/optimizer）。
            for module in self.frozen_modules:
                _validate_megatron_module(module, self.forward_mode,
                                          allow_module_hooks=True,
                                          allow_main_grad=False)
                binding = _repack_one(module, allow_module_hooks=True)
                binding.plan = self
                self.bindings.append(binding)
                _install_forward(binding, self.forward_mode)
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
            "dist-opt 模式必须保留 DDP 参数实体并调用 get_megatron_optimizer；"
            "不能使用非 dist-opt 的 plan.optimizer_params()")

    def restore(self, detach_storage: bool = False):
        """幂等摘除 big_leaf；weight{i} 保持 param_data 视图（值持续最新）。

        **存储生命周期契约（重要）**：默认 restore **不解除**可训模块
        `weight{i}` 对 DDP buffer 存储的依赖——`weight{i}.data` 仍是内部线
        `big` 实体（即 DDP param_data）的切片视图。因此调用方必须保证
        **DDP 对象存活期 ⩾ restore 后 weight{i} 的使用期**；若 restore 后释放
        了 DDP，可训 `weight{i}` 会悬空。

        restore 后**不能继续训练**：DDP 的梯度实体是内部线，标准反向会落到
        外部线 weight{i} 上而无人规约。与生产回滚策略一致（优先新进程）。

        detach_storage=True：可训模块的 `weight{i}` 也 clone 回独立存储，解除
        对 DDP buffer 的依赖，用于"导出后立即释放 DDP"的场景。代价是每个可训
        GroupedLinear 瞬时多占一份本地专家权重显存。冻结模块用独立 buf，无论
        该开关都必须 clone（否则 buf 随 bindings 释放后 weight{i} 悬空）。
        """
        if not self._active:
            return
        if self._broken:
            raise RuntimeError("big_leaf Megatron plan 此前 restore 失败，请重建模型")
        frozen_ids = {id(m) for m in self.frozen_modules}
        try:
            for binding in self.bindings:
                _uninstall_forward(binding)
                # 冻结绑定的独立 buf 会随 bindings 释放，weight{i} 必须脱离视图；
                # 可训绑定仅在显式 detach_storage 时脱离 DDP param_data 视图。
                if detach_storage or id(binding.module) in frozen_ids:
                    with torch.no_grad():
                        for i in range(binding.module.num_gemms):
                            p = getattr(binding.module, f"weight{i}")
                            p.data = p.data.clone()
                        if binding.bbuf is not None:
                            for i in range(binding.module.num_gemms):
                                p = getattr(binding.module, f"bias{i}")
                                p.data = p.data.clone()
        except Exception:
            self._broken = True
            raise
        self._restore_lora_merges()
        self._restore_load_state_dict()
        self.bindings.clear()
        self.proxies.clear()
        if self.model is not None and "_big_leaf_plan" in self.model.__dict__:
            delattr(self.model, "_big_leaf_plan")
        self._active = False
