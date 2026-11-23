"""fully-reshardable dist-opt optimizer state 的双线格式转换（旧→新 / 新→旧）。

背景（2026-11-19 双线实体改版）：DistributedOptimizer 的参数实体从
per-weight{i}（stock 与旧版 big_leaf 的保存格式，下文称 **expert 格式**）改为
内部线实体 big/big_b（**entity 格式**）。fully-reshardable 的 `param_state`
是 `{param_idx: {state_key: tensor}}` 的**位置字典**，位置序 = 保存侧 DDP buffer
`param_index_map` 的迭代序。两种格式的位置序都是确定性的：

- expert 格式 = mcore 默认 layout 的迭代序（`stock_positions()` 复刻）；
- entity 格式 = big_leaf 自定义 layout 的迭代序（`entity_positions()` 复刻）。

因此**不需要源实例在场**就能迁移：加载侧按源格式构造读取模板、按目标格式重建
位置字典。ShardedTensor 的 key 里带 fqn（`optimizer.state.{state_key}.{fqn}`），
模板按键对齐读取。expert↔entity 的粒度变换是 dim 0 的 stack/slice。

假设（与生产形态一致，测试覆盖）：optimizer=adam，state_keys =
("param", "exp_avg", "exp_avg_sq")，dtype=fp32；非 GroupedLinear 参数两格式
同形直通。
"""
from typing import Dict, List, Optional, Tuple

import torch

_STATE_KEYS = ("param", "exp_avg", "exp_avg_sq")


def _name_map(model):
    """param id → (module_key, kind, expert_i)；非 expert 参数 kind="" 直通。"""
    out = {}
    for name, p in model.named_parameters():
        module_part, _, attr = name.rpartition(".")
        if attr.startswith("weight"):
            out[id(p)] = (module_part, "weight", int(attr[len("weight"):]))
        elif attr.startswith("bias"):
            out[id(p)] = (module_part, "bias", int(attr[len("bias"):]))
        else:
            out[id(p)] = (name, "", None)
    return out


def stock_positions(model, ddp_config, data_parallel_world_size,
                    expert_data_parallel_world_size=None):
    """expert 格式的 param_state 位置序（复刻 mcore 默认 layout 的迭代序）。"""
    from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    params = [p for p in model.parameters() if p.requires_grad]
    groups = group_params_for_buffers(params, ddp_config.grad_reduce_in_fp32)
    layout = DistributedOptimizer.compute_full_param_layout(
        params, None, data_parallel_world_size, ddp_config,
        expert_data_parallel_world_size=expert_data_parallel_world_size)
    by_id = _name_map(model)
    out = []
    for buffer_key in groups:
        for p in layout.layouts[buffer_key].param_index_map:
            module_key, kind, i = by_id[id(p)]
            out.append((module_key, kind, i, tuple(p.shape)))
    return out


def entity_positions(plan, ddp_config, data_parallel_world_size,
                     expert_data_parallel_world_size=None):
    """entity 格式的 param_state 位置序（复刻 big_leaf 自定义 layout 的迭代序）。"""
    from megatron.core.distributed.param_and_grad_buffer import group_params_for_buffers

    entities = plan.entities()
    groups = group_params_for_buffers(
        [p for p in entities if p.requires_grad], ddp_config.grad_reduce_in_fp32)
    layout = plan.compute_full_param_layout(
        entities, None, data_parallel_world_size, ddp_config,
        expert_data_parallel_world_size=expert_data_parallel_world_size)
    by_id = _name_map(plan.model)
    mod_names = {id(m): n for n, m in plan.model.named_modules()}
    out = []
    for buffer_key in groups:
        for p in layout.layouts[buffer_key].param_index_map:
            for module in plan.modules_to_bind:
                big, big_b = plan.proxies[id(module)]
                if p is big:
                    out.append((mod_names[id(module)], "weight", None,
                                tuple(big.data.shape)))
                    break
                if big_b is not None and p is big_b:
                    out.append((mod_names[id(module)], "bias", None,
                                tuple(big_b.data.shape)))
                    break
            else:
                module_key, kind, i = by_id[id(p)]
                out.append((module_key, kind, i, tuple(p.shape)))
    return out


def convert_param_state(state: Dict[int, Dict[str, torch.Tensor]],
                        src_positions: List[Tuple], tgt_positions: List[Tuple],
                        plan=None) -> Dict[int, Dict[str, torch.Tensor]]:
    """把 param_state 从源位置序/粒度转换到目标位置序/粒度。

    state：`{param_idx: {state_key: tensor}}`（fully-reshardable canonical 整形张量）。
    positions：`(module_key, kind, expert_i, shape)`，kind="" 表示直通参数。
    plan 非空时，属于其 modules_to_bind 的 (module_key, kind) 目标按实体粒度重建
    （expert 源按 dim 0 拼接），其余按 expert 粒度切片/直通。
    """
    state_keys = sorted(next(iter(state.values())).keys())
    bound_names = set()
    if plan is not None:
        mod_names = {id(m): n for n, m in plan.model.named_modules()}
        bound_names = {mod_names[id(m)] for m in plan.modules_to_bind}

    src_map = {}  # (module_key, kind) -> {state_key: {expert_i: tensor}}
    for idx, (module_key, kind, i, _shape) in enumerate(src_positions):
        entry = state[idx]
        for sk in state_keys:
            src_map.setdefault((module_key, kind), {}).setdefault(sk, {})[i] = entry[sk]

    out = {}
    for tgt_idx, (module_key, kind, i, _shape) in enumerate(tgt_positions):
        group = src_map[(module_key, kind)]
        entry = {}
        for sk in state_keys:
            src = group[sk]
            if module_key in bound_names:
                # 目标实体 ← per-expert 源：stack 到 (G, *expert_shape) 的实体
                # canonical 名义形状（与 big/big_b.data 一致）。flatten 后与旧
                # cat(dim=0) 字节序完全相同，下游 owned-shard 1D 消费数值不变，
                # 但名义形状自洽，不依赖「下游必然 flatten」的隐式前提。
                indices = sorted(k for k in src if isinstance(k, int))
                entry[sk] = torch.stack([src[j] for j in indices], dim=0)
            elif kind == "":
                entry[sk] = src[None]
            else:
                entry[sk] = src[i] if i in src else src[None][i]
        out[tgt_idx] = entry
    return out


def build_param_state_template(positions: List[Tuple],
                               state_keys: Tuple[str, ...] = _STATE_KEYS,
                               device: str = "cuda"):
    """按位置序构造读取模板（键带 fqn，供 dist_checkpointing.load 按键对齐）。"""
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    template = {}
    for idx, (module_key, kind, i, shape) in enumerate(positions):
        if kind == "":
            fqn = module_key
        elif i is None:
            fqn = f"{module_key}.{kind}"
        else:
            fqn = f"{module_key}.{kind}{i}"
        entry = {}
        for sk in state_keys:
            entry[sk] = ShardedTensor.from_rank_offsets(
                key=f"optimizer.state.{sk}.{fqn}",
                data=torch.empty(shape, dtype=torch.float32, device=device),
            )
        template[idx] = entry
    return template
