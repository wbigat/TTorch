"""big_leaf 与 stock/LW=1 之间的 optimizer state 双向转换。

转换基于参数名和 model-space 张量，不猜 DistributedOptimizer 的本地 shard。
dist-opt 请先使用 fully-reshardable 格式还原成 model-space state。
"""
import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SourceRef = Tuple[str, Optional[int]]


def _optimizer_param_ids(optimizer: torch.optim.Optimizer) -> set:
    return {id(p) for group in optimizer.param_groups for p in group["params"]}


def _named_map(named_parameters: Iterable[Tuple[str, torch.Tensor]]) -> Dict[int, str]:
    out = {}
    names = set()
    for name, p in named_parameters:
        if name in names:
            raise ValueError(f"重复参数名: {name}")
        if id(p) in out and out[id(p)] != name:
            raise ValueError(f"同一参数对象对应多个名字: {out[id(p)]}, {name}")
        names.add(name)
        out[id(p)] = name
    return out


def _raw_layout(optimizer: torch.optim.Optimizer, raw_state: dict):
    raw_groups = raw_state["param_groups"]
    if len(raw_groups) != len(optimizer.param_groups):
        raise ValueError("optimizer 当前 param_groups 与待转换 state 不一致")
    param_location = {}
    for gi, (group, raw_group) in enumerate(zip(optimizer.param_groups, raw_groups)):
        if len(group["params"]) != len(raw_group["params"]):
            raise ValueError(f"optimizer group {gi} 的参数数与 state 不一致")
        for pi, (p, raw_index) in enumerate(zip(group["params"], raw_group["params"])):
            if id(p) in param_location:
                raise ValueError("同一参数在 optimizer 中重复出现")
            param_location[id(p)] = (gi, pi, raw_index)
    return param_location


def _equal(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.detach().cpu(), b.detach().cpu())
    if type(a) is not type(b):
        return False
    try:
        return bool(a == b)
    except (TypeError, ValueError, RuntimeError):
        return False


def _convert_value(values: Sequence[object], source_params: Sequence[torch.Tensor],
                   slice_indices: Sequence[Optional[int]], target_param: torch.Tensor):
    first = values[0]
    if torch.is_tensor(first):
        if not all(torch.is_tensor(v) for v in values):
            raise ValueError("optimizer state 同时包含 tensor 与非 tensor")
        converted = []
        param_shaped = []
        for value, source_param, slice_index in zip(values, source_params, slice_indices):
            is_param_shaped = value.shape == source_param.shape
            if slice_index is not None and is_param_shaped:
                if value.ndim == 0 or slice_index >= value.shape[0]:
                    raise ValueError("big_leaf state 的专家切片下标越界")
                value = value[slice_index]
                is_param_shaped = True
            converted.append(value)
            param_shaped.append(is_param_shaped)
        if all(param_shaped):
            result = converted[0].detach().clone() if len(converted) == 1 else torch.stack(converted, dim=0)
            if result.shape != target_param.shape:
                raise ValueError(
                    f"state 合并/切片后 shape={tuple(result.shape)} 与目标参数 "
                    f"{tuple(target_param.shape)} 不一致")
            return result
        if not all(_equal(first, v) for v in converted[1:]):
            raise ValueError("非参数形状的 optimizer state 在被合并参数之间不一致")
        return converted[0].detach().clone()
    if isinstance(first, dict):
        if not all(isinstance(v, dict) and set(v) == set(first) for v in values):
            raise ValueError("嵌套 optimizer state 的 dict 结构不一致")
        return {k: _convert_value([v[k] for v in values], source_params, slice_indices, target_param)
                for k in first}
    if isinstance(first, (list, tuple)):
        if not all(type(v) is type(first) and len(v) == len(first) for v in values):
            raise ValueError("嵌套 optimizer state 的序列结构不一致")
        items = [_convert_value([v[i] for v in values], source_params, slice_indices, target_param)
                 for i in range(len(first))]
        return type(first)(items)
    if not all(_equal(first, v) for v in values[1:]):
        raise ValueError("标量 optimizer state 在被合并参数之间不一致")
    return copy.deepcopy(first)


def convert_optimizer_state_dict(
    source_optimizer: torch.optim.Optimizer,
    source_named_parameters: Iterable[Tuple[str, torch.Tensor]],
    target_optimizer: torch.optim.Optimizer,
    target_named_parameters: Iterable[Tuple[str, torch.Tensor]],
    target_to_source: Dict[str, List[SourceRef]],
) -> dict:
    """把 source optimizer.state_dict() 转成 target 可直接 load 的格式。

    target_to_source[target_name] = [(source_name, expert_index), ...]：
    - 多个无 index 的引用表示 stack 到 big_leaf；
    - 一个带 index 的引用表示从 big_leaf 切回某个专家；
    - 一个无 index 的引用表示普通一对一映射。
    """
    source_named_parameters = list(source_named_parameters)
    target_named_parameters = list(target_named_parameters)
    if type(source_optimizer) is not type(target_optimizer):
        raise TypeError(
            f"仅支持同种 optimizer 转换，得到 {type(source_optimizer).__name__} → "
            f"{type(target_optimizer).__name__}")
    source_raw = source_optimizer.state_dict()
    source_names = _named_map(source_named_parameters)
    target_names = _named_map(target_named_parameters)
    source_params = {source_names[id(p)]: p
                     for group in source_optimizer.param_groups for p in group["params"]}
    target_params = {target_names[id(p)]: p
                     for group in target_optimizer.param_groups for p in group["params"]}
    source_location = _raw_layout(source_optimizer, source_raw)
    target_raw = target_optimizer.state_dict()
    target_location = _raw_layout(target_optimizer, target_raw)
    source_param_by_name = {name: p for name, p in source_named_parameters}
    target_param_by_name = {name: p for name, p in target_named_parameters}

    if set(target_params) != set(target_to_source):
        raise ValueError(
            f"转换映射必须恰好覆盖 target optimizer 参数；缺少={set(target_params)-set(target_to_source)}，"
            f"多出={set(target_to_source)-set(target_params)}")

    consumed_sources = set()
    converted_state = {}
    target_group_sources = {gi: set() for gi in range(len(target_optimizer.param_groups))}
    for target_name, refs in target_to_source.items():
        target_param = target_params[target_name]
        target_id = id(target_param_by_name[target_name])
        tgi, _, target_raw_index = target_location[target_id]
        entries = []
        params = []
        slice_indices = []
        for source_name, slice_index in refs:
            if source_name not in source_params:
                raise ValueError(f"source optimizer 中不存在参数 {source_name}")
            consumed_sources.add(source_name)
            source_param = source_param_by_name[source_name]
            sgi, _, source_raw_index = source_location[id(source_param)]
            target_group_sources[tgi].add(sgi)
            entries.append(source_raw["state"].get(source_raw_index, {}))
            params.append(source_param)
            slice_indices.append(slice_index)
        if not any(entries):
            continue
        if not all(entries):
            raise ValueError(f"{target_name} 的源参数只有一部分初始化了 optimizer state")
        keys = set(entries[0])
        if any(set(entry) != keys for entry in entries[1:]):
            raise ValueError(f"{target_name} 的源 optimizer state keys 不一致")
        converted_state[target_raw_index] = {
            key: _convert_value([entry[key] for entry in entries], params,
                                slice_indices, target_param_by_name[target_name])
            for key in keys
        }

    unconsumed = set(source_params) - consumed_sources
    if unconsumed:
        raise ValueError(f"source optimizer state 存在未消费参数: {sorted(unconsumed)}")

    output_groups = []
    for tgi, target_raw_group in enumerate(target_raw["param_groups"]):
        source_group_ids = target_group_sources[tgi]
        if len(source_group_ids) != 1:
            raise ValueError(
                f"target group {tgi} 来自 source groups {sorted(source_group_ids)}，"
                "无法保留一致的 lr/wd/betas 等语义")
        source_group = source_raw["param_groups"][next(iter(source_group_ids))]
        group = {k: copy.deepcopy(v) for k, v in source_group.items() if k != "params"}
        group["params"] = list(target_raw_group["params"])
        output_groups.append(group)
    return {"state": converted_state, "param_groups": output_groups}


def _expert_name(base: str, kind: str, index: Optional[int] = None) -> str:
    suffix = kind if index is None else f"{kind}{index}"
    return f"{base}.{suffix}" if base else suffix


def _leaf_base(plan, model: torch.nn.Module, binding) -> str:
    leaf_name = next(name for name, leaf in plan.named_leafs(model) if leaf is binding.big)
    return leaf_name.rsplit(".", 1)[0] if "." in leaf_name else ""


def mapping_to_big_leaf(plan, model: torch.nn.Module,
                        source_named_parameters: Iterable[Tuple[str, torch.Tensor]],
                        target_named_parameters: Iterable[Tuple[str, torch.Tensor]]):
    """构造 stock/LW=1 → big_leaf 映射；source 有 large_* 时优先一对一。"""
    source_names = {name for name, _ in source_named_parameters}
    mapping = {}
    for target_name, _ in target_named_parameters:
        matched = False
        for b in plan.bindings:
            base = _leaf_base(plan, model, b)
            for kind, leaf in (("weight", b.big), ("bias", b.big_b)):
                if leaf is None or target_name != _expert_name(base, f"large_{kind}"):
                    continue
                matched = True
                if target_name in source_names:
                    mapping[target_name] = [(target_name, None)]
                else:
                    refs = [(_expert_name(base, kind, i), None)
                            for i in range(b.module.num_gemms)]
                    missing = [name for name, _ in refs if name not in source_names]
                    if missing:
                        raise ValueError(f"stock source 缺少专家参数: {missing}")
                    mapping[target_name] = refs
        if not matched:
            mapping[target_name] = [(target_name, None)]
    return mapping


def mapping_from_big_leaf(plan, model: torch.nn.Module,
                          target_named_parameters: Iterable[Tuple[str, torch.Tensor]]):
    """构造 big_leaf → stock 映射。"""
    mapping = {}
    for target_name, _ in target_named_parameters:
        matched = False
        for b in plan.bindings:
            base = _leaf_base(plan, model, b)
            for kind, leaf in (("weight", b.big), ("bias", b.big_b)):
                if leaf is None:
                    continue
                for i in range(b.module.num_gemms):
                    if target_name == _expert_name(base, kind, i):
                        mapping[target_name] = [(_expert_name(base, f"large_{kind}"), i)]
                        matched = True
        if not matched:
            mapping[target_name] = [(target_name, None)]
    return mapping


def convert_to_big_leaf(plan, source_optimizer, source_model,
                        target_optimizer, target_model) -> dict:
    source_named = [(name, p) for name, p in source_model.named_parameters()
                    if id(p) in _optimizer_param_ids(source_optimizer)]
    target_named = [(name, p) for name, p in plan.named_optimizer_params(target_model)
                    if id(p) in _optimizer_param_ids(target_optimizer)]
    mapping = mapping_to_big_leaf(plan, target_model, source_named, target_named)
    return convert_optimizer_state_dict(
        source_optimizer, source_named, target_optimizer, target_named, mapping)


def convert_from_big_leaf(plan, source_optimizer, source_model,
                          target_optimizer, target_model) -> dict:
    source_named = [(name, p) for name, p in plan.named_optimizer_params(source_model)
                    if id(p) in _optimizer_param_ids(source_optimizer)]
    target_named = [(name, p) for name, p in target_model.named_parameters()
                    if id(p) in _optimizer_param_ids(target_optimizer)]
    mapping = mapping_from_big_leaf(plan, source_model, target_named)
    return convert_optimizer_state_dict(
        source_optimizer, source_named, target_optimizer, target_named, mapping)
