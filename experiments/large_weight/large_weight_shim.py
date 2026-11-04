"""XTE large-weight 布局的 per-expert 视图 shim（B2 / B3）。

**缺陷背景（实证 + 源码核实）**

厂商 TE 的 grouped linear 在 ``XTE_GROUPED_GEMM_LARGE_WEIGHT=1`` 下把 G 份
专家权重合并成**单个 3D 参数** ``large_weight``，形状 (num_gemms, out, in)；
``weight{i}`` / ``bias{i}`` **不再注册**。实证（transformer_engine
2.12.0+c9c3d0d9，meta device 构造 num_gemms=4）::

    LW=0  params=['weight0','bias0',...,'weight3','bias3']   weight{i} 可访问
    LW=1  params=['large_weight','large_bias']               weight{i} 无

而 ``mcore_bridge`` 的 HF↔Megatron 转换与 LoRA merge 全程按
``getattr(layer, f'weight{i}')`` 寻址——``bridge/gpt_bridge.py`` 有 12 处
（819/860/867/993/997/1044/1133/1136/1153/1230/1234/1255）。LW=1 时全部
``AttributeError``。生产脚本 ``runtime/script/env.sh:31`` 正是 ``=1``，其注释
已记录该症状："与 mcore_bridge 权重命名曾出现不兼容，no attribute error"。

厂商源码自己标注这是临时方案（``grouped_linear.py:786``：
"large weight is temporary solution，need to be removed in the future"），
且该开关默认 False（``xmegatron_ext/envs.py:68``）——所以本 shim 以
**large_weight 是否真实存在**为门控：LW=0 时自动不接管，零副作用。

**为什么不能只靠 XME 的 __getattr__ shim**

XME 用 ``KlxSymbolRewriter`` 把 ``TEGroupedLinear.__getattr__`` 改写成合成
``weight{i}`` 视图（``xmegatron_ext/plugin.py:113-117``）。读路径够用，但
**写路径不行**：``__getattr__`` 只在属性查找失败时触发，而
``LoraParallelLinear.merge`` 在 ``safe_merge`` 下执行
``weight.data = orig_weights[i]``（``mcore_bridge/tuners/lora.py:538-540``）
—— 这是**重新绑定**那个临时视图对象，不是透写回 large_weight 的存储，
合并结果静默丢失。所以必须：设真实属性（真实属性遮蔽 __getattr__）→
调用原实现 → **把被重绑的值 copy_ 回 large_weight** → 还原层。
"""
import torch

__all__ = ["has_large_weight", "expose_per_expert", "restore_layer",
           "write_back", "iter_expert_layers", "num_experts_of"]


def has_large_weight(layer):
    """该层是否处于 large-weight 布局（本 shim 的唯一门控）。

    判据是"``large_weight`` 是否真实存在且非空"，不读环境变量——厂商在
    ``noop_cat`` 零拷贝成立时也会把 ``large_weight`` 设成小权重的视图
    （``grouped_linear.py:1005``），此时布局其实是 per-expert。以实际属性
    为准，比以开关为准更贴近运行态（待办 C2 记录的"属性存在判据失真"
    即源于混用两种判据）。
    """
    lw = getattr(layer, "large_weight", None)
    return lw is not None and getattr(lw, "numel", lambda: 0)() > 0


def num_experts_of(layer):
    """该层的本地专家数。

    优先取 ``layer.num_gemms``（层自己的事实），其次退到 large_weight 的
    首维。**不用** ``config.num_moe_experts // ep_size`` 推算——那是待办 C8
    「专家数口径不一致」的根因：推算值与层实际切分不符时，按它切分会静默
    产生错误的专家映射。拿不到可信值返回 None（调用方据此不接管）。
    """
    n = getattr(layer, "num_gemms", None)
    lw = getattr(layer, "large_weight", None)
    shape0 = lw.shape[0] if (lw is not None and lw.dim() >= 1) else None
    if n is not None and shape0 is not None and n != shape0:
        return None  # 口径冲突：宁可不接管，不猜
    return n if n is not None else shape0


def _is_real_attr(layer, name):
    """该名字是否为层上**真实存储**的属性。

    不能用 ``hasattr``：XME 把 ``__getattr__`` 改写成合成 ``weight{i}`` 视图，
    对任意专家下标都答 True。把合成视图记进 restore 清单，还原时会被
    ``setattr`` 成真实属性，**永久遮蔽** shim 并固化一份过期视图。
    """
    if name in layer.__dict__:
        return True
    for store in ("_parameters", "_buffers"):
        if name in (layer.__dict__.get(store) or {}):
            return True
    return False


def expose_per_expert(layer, num_experts):
    """把 large_weight/large_bias 的专家切片设为真实属性 weight{i}/bias{i}。

    返回 (restore_items, delete_names) 供 restore_layer 还原。
    用直接索引 ``large[i]`` 而非 ``chunk(n).squeeze(0)``：索引的语义与
    ``num_experts`` 解耦，chunk 在份数与首维不整除时会静默给出错误切分。
    """
    restore_items, delete_names = [], []
    for kind in ("weight", "bias"):
        large = getattr(layer, f"large_{kind}", None)
        # TE 在无 bias 时注册一个零尺寸占位，不是真参数
        if large is None or large.numel() == 0 or large.dim() < 1:
            continue
        if large.shape[0] < num_experts:
            continue  # 形状与专家数不符：该 kind 不接管
        for idx in range(num_experts):
            name = f"{kind}{idx}"
            if _is_real_attr(layer, name):
                restore_items.append((name, getattr(layer, name)))
            else:
                delete_names.append(name)
            setattr(layer, name, large[idx])
    return restore_items, delete_names


def write_back(layer, num_experts):
    """把调用方**重新绑定**过的专家属性写回 large_weight 的对应切片。

    指针比对是为了跳过原地写路径：那里属性仍别名着切片，copy_ 到自己身上
    是无意义的空操作。只有 ``x.data = y`` 这种重绑才需要真正搬运。
    返回实际写回的切片数（0 = 没有发生重绑，属正常情况）。
    """
    n = 0
    for kind in ("weight", "bias"):
        large = getattr(layer, f"large_{kind}", None)
        if large is None or large.numel() == 0 or large.dim() < 1:
            continue
        for idx in range(min(num_experts, large.shape[0])):
            cur = getattr(layer, f"{kind}{idx}", None)
            if cur is None:
                continue
            target = large.data[idx]
            cur_data = getattr(cur, "data", cur)
            if cur_data.data_ptr() == target.data_ptr():
                continue        # 原地写：已经落在 large 存储上
            if cur_data.shape != target.shape:
                continue        # 形状不符：不猜，不写（避免静默污染）
            with torch.no_grad():
                target.copy_(cur_data)
            n += 1
    return n


def restore_layer(layer, restore_items, delete_names):
    """还原 expose_per_expert 的改动（先删临时名，再恢复原有真实属性）。"""
    for name in delete_names:
        try:
            delattr(layer, name)
        except AttributeError:
            pass
    for name, value in restore_items:
        setattr(layer, name, value)


def iter_expert_layers(mg_mlp):
    """产出一个 MLP 下所有可能处于 large-weight 布局的 linear 层。

    覆盖 base layer 与 LoRA 的 A/B —— LoRA 场景下 grouped linear 被
    ``LoraParallelLinear`` 包住，转换要同时看到三者。
    """
    if mg_mlp is None:
        return
    for linear_name in ("linear_fc1", "linear_fc2"):
        linear = getattr(mg_mlp, linear_name, None)
        if linear is None:
            continue
        yield linear
        for lora_name in ("lora_A", "lora_B"):
            lora = getattr(linear, lora_name, None)
            if lora is None:
                continue
            if hasattr(lora, "values"):          # ModuleDict / dict
                yield from lora.values()
        base = getattr(linear, "base_layer", None)
        if base is not None:
            yield base
