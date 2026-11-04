"""Four-rank real Swift-Megatron SFT evidence launcher.

Run with torchrun; --comparison-mode and --evidence-dir are consumed here,
all other arguments are passed unchanged to megatron_sft_main.
"""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch


def parameter_digest(models, save_path=None):
    digest = hashlib.sha256()
    state = {}
    count = 0
    for chunk, model in enumerate(models):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            key = f"{chunk}.{name}"
            value = param.detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
            count += value.numel()
            if save_path is not None:
                state[key] = value
    if save_path is not None:
        torch.save(state, save_path)
    return {"sha256": digest.hexdigest(), "trainable_params": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-mode", choices=("xmlir", "tt"), required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    options, training_args = parser.parse_known_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    options.evidence_dir.mkdir(parents=True, exist_ok=True)
    if options.comparison_mode == "tt":
        if os.environ.get("TT_PLUGIN_ENABLE") != "1":
            raise RuntimeError("TT comparison requires TT_PLUGIN_ENABLE=1")
        import ttorch
        ttorch.activate()
        ttorch.assert_healthy()
        if not ttorch.is_active():
            raise RuntimeError("TT activation failed")
    elif "ttorch" in __import__("sys").modules:
        raise RuntimeError("xmlir baseline unexpectedly imported TT")

    from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map

    class EvidenceCallback(MegatronCallback):
        def __init__(self, trainer):
            super().__init__(trainer)
            self.path = options.evidence_dir / f"rank{rank}.jsonl"
            self.stream = self.path.open("w")
            self.step_batches = []
            self.step_seconds = None
            self.started = None
            self.previous_step = None
            collator = trainer.data_collator

            def measured_collator(*args, **kwargs):
                # packing=true 时 row 是 list（已打包的样本序列），不是 dict。
                source_shapes = [
                    ({key: len(row[key]) for key in
                      ("input_ids", "labels", "mm_token_type_ids")
                      if row.get(key) is not None}
                     if isinstance(row, dict) else {"packed_list_len": len(row)})
                    for row in args[0]]
                try:
                    batch = collator(*args, **kwargs)
                except Exception:
                    self.write("collator_error", source_shapes=source_shapes)
                    raise
                ids = batch.get("input_ids")
                labels = batch.get("labels")
                if isinstance(ids, torch.Tensor):
                    if ids.device.type != "cpu":
                        raise RuntimeError("Data fingerprint expects CPU collator output")
                    fingerprint = hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
                    tensors = {}
                    for key in ("labels", "loss_scale", "position_ids", "mm_token_type_ids",
                                "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                        value = batch.get(key)
                        if isinstance(value, torch.Tensor):
                            if value.device.type != "cpu":
                                raise RuntimeError(f"Data fingerprint expects CPU {key}")
                            tensors[key] = {
                                "shape": list(value.shape), "dtype": str(value.dtype),
                                "sha256": hashlib.sha256(value.contiguous().view(torch.uint8)
                                                         .numpy().tobytes()).hexdigest(),
                            }
                    self.step_batches.append({
                        "shape": list(ids.shape), "input_sha256": fingerprint,
                        "input_tokens": ids.numel(), "tensors": tensors,
                        "label_tokens": int((labels != -100).sum())
                        if isinstance(labels, torch.Tensor) else None,
                    })
                return batch

            trainer.data_collator = measured_collator
            train_step = trainer.train_step

            def measured_train_step(*args, **kwargs):
                result = train_step(*args, **kwargs)
                self.update_successful = bool(result[2])
                return result

            trainer.train_step = measured_train_step

        def write(self, event, **fields):
            self.stream.write(json.dumps({"event": event, "rank": rank,
                                          "mode": options.comparison_mode, **fields},
                                         ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()

        def provider_state(self):
            if options.comparison_mode != "tt":
                return None
            from ttorch.provider import manager
            from ttorch.witness import provenance
            provenance.export_jsonl(str(options.evidence_dir / f"provenance.rank{rank}.jsonl"))
            return manager.instance().status()

        def module_state(self):
            import inspect
            counts = Counter()
            wrapped = Counter()
            sources = {}
            for model in self.trainer.unwrapped_models:
                for module in model.modules():
                    name = type(module).__name__
                    if "Norm" in name or "Linear" in name or "Attention" in name:
                        counts[name] += 1
                    if name not in sources and "Norm" in name:
                        original = getattr(module.forward, "_orig_forward", module.forward)
                        sources[name] = {
                            "class_module": type(module).__module__,
                            "eps": float(getattr(module, "eps", 1e-6)),
                            "zero_centered_gamma": getattr(module, "zero_centered_gamma", None),
                            "forward_source": inspect.getsource(original),
                        }
                    if getattr(module, "_tt_provider_wrapped", False):
                        wrapped[getattr(module.forward, "_semantic", "unknown")] += 1
            return {"classes": dict(counts), "wrapped": dict(wrapped), "sources": sources}

        def on_train_begin(self):
            self.write("begin", world_size=torch.distributed.get_world_size(),
                       arguments=training_args, modules=self.module_state(),
                       parameters=parameter_digest(self.trainer.unwrapped_models,
                           options.evidence_dir / "initial_trainable.pt" if rank == 0 else None),
                       config=str(self.trainer.config),
                       provider=self.provider_state())
            torch.cuda.reset_peak_memory_stats()
            self.previous_step = time.perf_counter()

        def on_step_begin(self):
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.begin_interval = self.started - self.previous_step
            self.step_batches = []

        def on_step_end(self):
            torch.cuda.synchronize()
            self.previous_step = time.perf_counter()
            self.step_seconds = self.previous_step - self.started
            self.write("step", step=self.state.iteration, seconds=self.step_seconds,
                       pre_step_seconds=self.begin_interval,
                       update_successful=self.update_successful, batches=self.step_batches)

        def on_log(self, logs):
            self.write("metrics", step=self.state.iteration, metrics=dict(logs))

        def on_train_end(self):
            self.write("end", step=self.state.iteration, modules=self.module_state(),
                       parameters=parameter_digest(self.trainer.unwrapped_models,
                           options.evidence_dir / "final_trainable.pt" if rank == 0 else None),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       provider=self.provider_state())
            self.stream.close()

    megatron_callbacks_map["tt_comparison_evidence"] = EvidenceCallback
    from swift.megatron import megatron_sft_main
    megatron_sft_main(training_args + ["--callbacks", "tt_comparison_evidence"])


if __name__ == "__main__":
    main()
