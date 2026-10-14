"""真实 SFT 数据面：jsonl -> chat template -> tokenize -> loss mask。

穿透优先，只做单卡功能链路必需的部分：
- 读 LLaMA-Factory alpaca 格式（instruction/input/output，本机数据集形态）；
- 走 tokenizer 自带 chat template（与 swift `--template qwen3_5` 同源，
  模板由模型目录的 chat_template.jinja 提供，不自己拼 prompt）；
- **prompt 段 label=-100**，只对 answer 段算 loss（SFT 的定义性要求，
  错了 loss 看着也在降，但学的是复述输入）；
- 右侧 pad 到 batch 内最长，attention_mask 配套。

明确不做（铺面才需要）：packing / padding_free / 多模态图像分支 /
group_by_length / lazy_tokenize 流式 / channel loss。数据集里有
`images` 字段但本次只取纯文本样本（`n_images == 0`）。
"""
import json

IGNORE_INDEX = -100


def load_alpaca_jsonl(path, limit=None, text_only=True):
    """读 alpaca 格式 jsonl，返回 [{"instruction","input","output"}]。

    text_only：只保留纯文本样本——多模态样本需要 ViT/图像 token 分支，
    不在当前穿透点范围内，混进来会在 tokenize 阶段静默丢图而不报错。
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if text_only and d.get("images"):
                continue
            if not d.get("output"):  # 无答案的样本对 SFT 无意义
                continue
            rows.append({"instruction": d.get("instruction", ""),
                         "input": d.get("input", ""),
                         "output": d["output"]})
            if limit and len(rows) >= limit:
                break
    return rows


def _ids(encoded):
    """取 input_ids：transformers 5.x 的 apply_chat_template 返回
    BatchEncoding，4.x 返回 list[int]——按实测形态兼容，不按记忆假设
    （实测 transformers 5.12.1 返回 BatchEncoding(keys=input_ids,attention_mask)；
    不做这层转换时 len() 得到的是 key 数量 1，表现为"样本长度=1"的静默错误）。
    """
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        return list(encoded["input_ids"])
    return list(encoded)


def encode_sft(tokenizer, row, max_length):
    """单样本 -> (input_ids, labels)。prompt 段 label=IGNORE_INDEX。

    两次 apply_chat_template（prompt-only 与 full）取长度差来定位 answer
    起点：这样 mask 边界由模板自己决定，不依赖我们猜模板里有几个特殊
    token——换模板/换模型不会静默错位。
    """
    user = row["instruction"] + (("\n" + row["input"]) if row["input"] else "")
    prompt_ids = _ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=True))
    full_ids = _ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": user},
         {"role": "assistant", "content": row["output"]}],
        add_generation_prompt=False, tokenize=True))

    # 截断保留答案开头：从右截断会把 answer 整段切掉，loss 变成纯 padding
    full_ids = full_ids[:max_length]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * n_prompt + list(full_ids[n_prompt:])
    return list(full_ids), labels


def collate(batch, pad_token_id):
    """右 pad 到 batch 内最长；pad 位 label=IGNORE_INDEX、attention_mask=0。"""
    import torch
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(lab + [IGNORE_INDEX] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def build_batches(tokenizer, rows, batch_size, max_length):
    """编码全部样本并切 batch。返回 (batches, stats)。

    丢弃 answer 段为空的样本（截断后 label 全是 IGNORE_INDEX）——这类样本
    的 loss 是 NaN，不剔除会污染整个 batch。

    stats 把丢弃量显式暴露出来，**不静默**：本机数据集样本总长中位数
    约 6k token（prompt 段中位数 2.1k），max_length 设小会让绝大多数
    样本的 answer 被整段截掉，表现为"能跑但只在少数短样本上训练"——
    这正是 skill 说的"看着在降的 loss"类陷阱。调用方据 stats 判断
    max_length 是否合理（真实训练脚本用 SEQ_LEN=20480）。
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    encoded, dropped, truncated = [], 0, 0
    for r in rows:
        ids, lab = encode_sft(tokenizer, r, max_length)
        full_len = len(encode_sft(tokenizer, r, 10 ** 9)[0])
        truncated += 1 if full_len > max_length else 0
        if any(x != IGNORE_INDEX for x in lab):
            encoded.append((ids, lab))
        else:
            dropped += 1  # answer 被 max_length 整段截掉
    batches = [collate(encoded[i:i + batch_size], pad_id)
               for i in range(0, len(encoded) - batch_size + 1, batch_size)]
    stats = {"n_rows": len(rows), "n_valid": len(encoded),
             "n_dropped_answer_truncated": dropped, "n_truncated": truncated,
             "max_length": max_length, "n_batches": len(batches)}
    return batches, stats
