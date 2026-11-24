"""fully-reshardable dist-opt optimizer state 的双线格式转换（旧→新 / 新→旧）。

背景（2026-11-19 双线实体改版）：DistributedOptimizer 的参数实体从
per-weight{i}（stock 与旧版 big_leaf 的保存格式，下文称 **expert 格式**）改为
内部线实体 big/big_b（**entity 格式**）。fully-reshardable 的 `param_state`
是 `{param_idx: {state_key: tensor}}` 的**位置字典**，位置序 = 保存侧 DDP buffer
`param_index_map` 的迭代序。两种格式的位置序都是确定性的：

- expert 格式 = mcore 默认 layout 的迭代序（`stock_positions()` 复刻）；
- entity 格式 = big_leaf 自定义 layout 的迭代序（`entity_positions()` 复刻）。

读盘机制（review 2026-11-23 1d 澄清）：迁移的**读盘**依赖构造一个**源格式参考
实例**（fresh model+DDP+optimizer，与目标同架构、格式相反），用 mcore 原生
`sharded_state_dict` 作读模板——不需要原始训练实例。`build_param_state_template`
手搓的裸 `from_rank_offsets` ShardedTensor **缺 fully-reshardable 的真实分片元数据**
（replica_id、buffer 切片 offset、模型 sharded metadata），实测驱动不了真实 load
（"Invalid access pattern"）——它只作**布局对照诊断**（验证 position 推导的 FQN
key/shape 与真实一致），不用于读盘。expert↔entity 的粒度变换是 dim 0 的 stack/slice。

假设（与生产形态一致，测试覆盖）：optimizer=adam，state_keys =
("param", "exp_avg", "exp_avg_sq")，dtype=fp32；非 GroupedLinear 参数两格式
同形直通。**未接入生产 load 的自动格式检测**（当前迁移由调用方显式驱动）。
"""
from typing import Dict, List, Optional, Tuple

import torch

_STATE_KEYS = ("param", "exp_avg", "exp_avg_sq")


def _name_map(model):
    """param id → (module_key, kind, expert_i)；非 expert 参数 kind="" 直通。

    只把带**数字后缀**的 `weight{i}`/`bias{i}` 当专家参数。普通 Linear/LayerNorm
    的 `weight`/`bias` 没有索引，必须直通——否则 `int("")` 会让任何含非
    GroupedLinear 参数的真实模型崩溃（review 2026-11-23 指认）。
    """
    out = {}
    for name, p in model.named_parameters():
        module_part, _, attr = name.rpartition(".")
        if attr.startswith("weight") and attr[len("weight"):].isdigit():
            out[id(p)] = (module_part, "weight", int(attr[len("weight"):]))
        elif attr.startswith("bias") and attr[len("bias"):].isdigit():
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
    for tgt_idx, (module_key, kind, i, declared) in enumerate(tgt_positions):
        group = src_map[(module_key, kind)]
        entry = {}
        for sk in state_keys:
            src = group[sk]
            if module_key in bound_names:
                indices = sorted(k for k in src if isinstance(k, int))
                if indices:
                    # expert → entity：stack 到 (G, *expert_shape) 的实体 canonical
                    # 名义形状（flatten 后与 cat(dim=0) 字节序一致，但形状自洽）。
                    result = torch.stack([src[j] for j in indices], dim=0)
                else:
                    # entity → entity：源只有 None 键，恒等直通（不做无谓 stack）。
                    result = src[None]
            elif kind == "":
                result = src[None]
            else:
                result = src[i] if i in src else src[None][i]
            # 目标声明 shape 是契约：**所有分支**（entity / 直通 / entity→expert）
            # 统一校验（review 三轮 P4：此前只覆盖 entity 分支，直通与 entity→expert
            # 声明与实际不符时静默放过）。不符即拒绝，不隐式产出错误形状。
            if tuple(result.shape) != tuple(declared):
                raise ValueError(
                    f"{module_key}.{kind}: 转换结果形状 {tuple(result.shape)} "
                    f"与目标声明 {tuple(declared)} 不符")
            entry[sk] = result
        out[tgt_idx] = entry
    return out


def migrate_param_state_to_target(raw_param_state, source_format, model,
                                  target_plan, ddp_config, data_parallel_world_size,
                                  expert_data_parallel_world_size=None,
                                  source_plan=None):
    """把按 source_format 读出的 param_state 转换到目标格式（供 optimizer.load_state_dict）。

    生产迁移的可复用入口（review 2026-11-23 1d：此前迁移逻辑内联在测试里，生产
    无可调用 API）。调用方负责用**源格式参考实例**的 `sharded_state_dict`（mcore
    原生机制）读出 `raw_param_state`；本函数完成位置序复刻与 expert↔entity 粒度
    转换（`convert_param_state`）。

    - `source_format="expert"`（stock / per-weight{i}）：位置用 `model` 推导，无需源实例；
    - `source_format="entity"`（big_leaf big/big_b）：位置需 `source_plan` 提供实体
      形状与 layout 序（生产无源时此方向需先构造源 plan，见模块 docstring 边界）。
    - 目标格式由 `target_plan` 推断（非空=entity，None=expert）。
    """
    if source_format == "expert":
        src_positions = stock_positions(model, ddp_config, data_parallel_world_size,
                                        expert_data_parallel_world_size)
    elif source_format == "entity":
        if source_plan is None:
            raise ValueError(
                "source_format='entity' 需要 source_plan 提供实体形状与位置序")
        src_positions = entity_positions(source_plan, ddp_config,
                                         data_parallel_world_size,
                                         expert_data_parallel_world_size)
    else:
        raise ValueError(f"未知 source_format={source_format!r}（期望 'expert'/'entity'）")
    tgt_positions = (entity_positions(target_plan, ddp_config, data_parallel_world_size,
                                      expert_data_parallel_world_size)
                     if target_plan is not None
                     else stock_positions(model, ddp_config, data_parallel_world_size,
                                          expert_data_parallel_world_size))
    return convert_param_state(raw_param_state, src_positions, tgt_positions,
                               plan=target_plan)


def build_param_state_template(positions: List[Tuple],
                               state_keys: Tuple[str, ...] = _STATE_KEYS,
                               device: str = "cuda"):
    """按位置序构造 param_state 布局描述（键带 fqn）。

    **仅作布局对照诊断，不能驱动真实 load**：裸 `from_rank_offsets` 不含
    fully-reshardable 的分片元数据（replica_id / buffer 切片 offset / 模型
    sharded metadata），真实读盘必须改用源格式参考实例的 `sharded_state_dict`
    （见模块 docstring 的读盘机制说明）。
    """
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
