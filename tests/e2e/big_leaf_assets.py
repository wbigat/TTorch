"""为生产链路验收准备确定性的 BF16 Qwen3 MoE 初始权重与文本数据。"""
import json
import os
from pathlib import Path


def resolve_tokenizer_source(source=None):
    candidate = source or os.environ.get("TT_BIG_LEAF_TOKENIZER")
    if candidate:
        return str(candidate)
    raise ValueError(
        "首次生成默认验收资产需要 tokenizer：请使用 --tokenizer 或 "
        "TT_BIG_LEAF_TOKENIZER 指定本地路径/HF 模型名；也可通过 --model 与 "
        "--dataset 使用外部 MoE，跳过资产生成")


def ensure_assets(root, tokenizer_source=None):
    root = Path(root)
    model_dir = root / "qwen3_moe_bf16"
    dataset = root / "text_sft.jsonl"
    metadata = root / "metadata.json"
    required = [model_dir / "config.json", model_dir / "model.safetensors",
                model_dir / "tokenizer_config.json", dataset, metadata]
    if all(path.is_file() for path in required):
        return model_dir, dataset, json.loads(metadata.read_text())

    source = resolve_tokenizer_source(tokenizer_source)
    import torch
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        num_experts=16,
        num_experts_per_tok=2,
        moe_intermediate_size=512,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        eos_token_id=tokenizer.eos_token_id,
        dtype=torch.bfloat16,
    )
    torch.manual_seed(42)
    model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    del model

    with dataset.open("w") as writer:
        for index in range(256):
            left = index % 20
            right = (index * 3) % 20
            record = {"messages": [
                {"role": "user", "content": f"Calculate {left} + {right}."},
                {"role": "assistant", "content": f"The answer is {left + right}."},
            ]}
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    info = {
        "model_type": "qwen3_moe",
        "initialization": "deterministic_random_seed_42",
        "dtype": "bfloat16",
        "num_hidden_layers": 4,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "hidden_size": 512,
        "dataset_records": 256,
        "note": "合成初始化整网，仅用于生产链路功能与相对性能验收，不代表预训练模型效果。",
    }
    metadata.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return model_dir, dataset, info
