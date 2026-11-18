"""生产验收脚本的仓库外契约检查。"""
import importlib
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))
production = importlib.import_module("big_leaf_production")


def _scripts(tmp_path, run_text, train_text):
    (tmp_path / "run.sh").write_text(run_text)
    (tmp_path / "train_megatron.sh").write_text(train_text)


def test_production_script_contract_accepts_parameterized_scripts(tmp_path):
    _scripts(
        tmp_path,
        'if [ -n "${DATASET_PATH}" ]; then\nTT_BIG_LEAF\n',
        'TEMPLATE:- SAVE_SAFETENSORS:- ADD_NON_THINKING_PREFIX:- ENABLE_CHANNEL_LOSS:- '
        'TT_BIG_LEAF:-',
    )
    production._validate_production_script_contract(tmp_path)


def test_production_script_contract_rejects_unmodified_scripts(tmp_path):
    _scripts(tmp_path, "", "--save_safetensors true")
    with pytest.raises(ValueError, match="伴随改动"):
        production._validate_production_script_contract(tmp_path)


def test_production_script_contract_requires_big_leaf_entrypoint(tmp_path):
    _scripts(
        tmp_path,
        'if [ -n "${DATASET_PATH}" ]; then\nTT_BIG_LEAF\n',
        'TEMPLATE:- SAVE_SAFETENSORS:- ADD_NON_THINKING_PREFIX:- ENABLE_CHANNEL_LOSS:-',
    )
    with pytest.raises(ValueError, match="TT_BIG_LEAF"):
        production._validate_production_script_contract(tmp_path)


def test_production_script_dir_is_required(monkeypatch):
    monkeypatch.delenv("TRAIN_SCRIPT_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["big_leaf_production.py", "--mode", "smoke"])
    with pytest.raises(SystemExit) as error:
        production.main()
    assert error.value.code == 2
