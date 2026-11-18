"""big_leaf 验收资产的可移植性与缓存依赖边界。"""
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "e2e" / "big_leaf_assets.py"
SPEC = importlib.util.spec_from_file_location("big_leaf_assets_under_test", MODULE_PATH)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def test_explicit_tokenizer_overrides_environment(monkeypatch):
    monkeypatch.setenv("TT_BIG_LEAF_TOKENIZER", "from-env")
    assert assets.resolve_tokenizer_source("from-cli") == "from-cli"
    assert assets.resolve_tokenizer_source() == "from-env"


def test_missing_tokenizer_has_actionable_error(monkeypatch):
    monkeypatch.delenv("TT_BIG_LEAF_TOKENIZER", raising=False)
    with pytest.raises(ValueError, match="--tokenizer"):
        assets.resolve_tokenizer_source()


def test_cached_assets_do_not_require_tokenizer_or_transformers(monkeypatch, tmp_path):
    model_dir = tmp_path / "qwen3_moe_bf16"
    model_dir.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer_config.json"):
        (model_dir / name).touch()
    (tmp_path / "text_sft.jsonl").touch()
    (tmp_path / "metadata.json").write_text(json.dumps({"cached": True}))
    monkeypatch.setattr(assets, "resolve_tokenizer_source",
                        lambda _: pytest.fail("缓存资产不应解析 tokenizer"))
    model, dataset, metadata = assets.ensure_assets(tmp_path)
    assert model == model_dir
    assert dataset == tmp_path / "text_sft.jsonl"
    assert metadata == {"cached": True}
