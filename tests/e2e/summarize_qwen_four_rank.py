"""Summarize completed four-rank ABBA evidence without rounding loss values."""
import argparse
import json
import math
import statistics
from pathlib import Path

import torch


def difference_stats(a, b):
    delta = [y - x for x, y in zip(a, b)]
    return {
        "count": len(delta), "max_abs": max(map(abs, delta)),
        "mean_abs": statistics.mean(map(abs, delta)), "signed_mean": statistics.mean(delta),
        "positive": sum(x > 0 for x in delta), "negative": sum(x < 0 for x in delta),
        "zero": sum(x == 0 for x in delta),
        "max_relative": max(abs(y - x) / max(abs(x), 1e-12) for x, y in zip(a, b)),
    }


def weight_difference(initial_path, a_path, b_path):
    initial = torch.load(initial_path, map_location="cpu", weights_only=True, mmap=True)
    a = torch.load(a_path, map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(b_path, map_location="cpu", weights_only=True, mmap=True)
    if a.keys() != b.keys() or a.keys() != initial.keys():
        raise RuntimeError("Trainable parameter names differ")
    max_abs = 0.0
    diff2 = ref2 = update2 = abs_sum = 0.0
    dot = b2 = 0.0
    elements = different = 0
    for key in a:
        av, bv, iv = a[key].double(), b[key].double(), initial[key].double()
        diff = bv - av
        max_abs = max(max_abs, float(diff.abs().max()))
        abs_sum += float(diff.abs().sum())
        diff2 += float(diff.square().sum())
        ref2 += float(av.square().sum())
        b2 += float(bv.square().sum())
        update2 += float((av - iv).square().sum())
        dot += float((av * bv).sum())
        elements += av.numel()
        different += int((av != bv).sum())
    return {
        "elements": elements, "different_elements": different, "max_abs": max_abs,
        "mean_abs": abs_sum / elements, "relative_l2": math.sqrt(diff2 / max(ref2, 1e-30)),
        "relative_l2_to_training_update": math.sqrt(diff2 / max(update2, 1e-30)),
        "cosine": dot / math.sqrt(max(ref2 * b2, 1e-30)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    options = parser.parse_args()
    names = ("xmlir1", "tt1", "tt2", "xmlir2")
    runs = {}
    rank_steps = {}
    rank_begins = {}
    rank_ends = {}
    for name in names:
        folder = options.directory / name
        if (folder / "exit_code.txt").read_text().strip() != "0":
            raise RuntimeError(f"{name} did not complete successfully")
        rank_steps[name], rank_begins[name], rank_ends[name] = [], [], []
        for rank in range(4):
            rows = [json.loads(line) for line in
                    (folder / "evidence" / f"rank{rank}.jsonl").read_text().splitlines()]
            begin = next(row for row in rows if row["event"] == "begin")
            end = next(row for row in rows if row["event"] == "end")
            steps = [row for row in rows if row["event"] == "step"]
            if len(steps) != end["step"] or begin["world_size"] != 4:
                raise RuntimeError(f"{name}/rank{rank}: incomplete evidence")
            if not all(row["update_successful"] for row in steps):
                raise RuntimeError(f"{name}/rank{rank}: optimizer updates skipped")
            rank_steps[name].append(steps)
            rank_begins[name].append(begin)
            rank_ends[name].append(end)
            if rank == 0:
                runs[name] = {
                    "metrics": [row for row in rows if row["event"] == "metrics"],
                    "modules": end["modules"], "provider": end["provider"],
                }
    count = len(rank_steps[names[0]][0])
    for name in names:
        if any(len(steps) != count for steps in rank_steps[name]):
            raise RuntimeError("Step counts differ")
        if len(runs[name]["metrics"]) != count:
            raise RuntimeError("Missing rank0 metrics")
    inputs_equal = all(
        [row["batches"] for row in rank_steps[name][rank]] ==
        [row["batches"] for row in rank_steps[names[0]][rank]]
        for name in names for rank in range(4))
    initial_hashes = {begin["parameters"]["sha256"] for name in names for begin in rank_begins[name]}
    if not inputs_equal or len(initial_hashes) != 1:
        raise RuntimeError("Initial trainable parameters or per-rank data differ")
    summary = {
        "directory": str(options.directory.resolve()), "steps": count,
        "warmup_steps": options.warmup, "all_rank_input_fingerprints_equal": inputs_equal,
        "initial_trainable_hashes": sorted(initial_hashes),
        "trainable_params": rank_begins[names[0]][0]["parameters"]["trainable_params"],
        "performance": {}, "accuracy": {}, "weights": {},
    }
    for name in names:
        times = [max(rank_steps[name][rank][step]["seconds"] for rank in range(4))
                 for step in range(options.warmup, count)]
        tokens = sum(sum(batch["input_tokens"] for batch in rank_steps[name][rank][step]["batches"])
                     for rank in range(4) for step in range(options.warmup, count))
        label_tokens = sum(sum(batch["label_tokens"] or 0
                               for batch in rank_steps[name][rank][step]["batches"])
                           for rank in range(4) for step in range(options.warmup, count))
        summary["performance"][name] = {
            "step_mean_s": statistics.mean(times), "step_median_s": statistics.median(times),
            "step_p95_s": sorted(times)[math.ceil(len(times) * 0.95) - 1],
            "samples_per_s": 4 / statistics.mean(times),
            "input_tokens_per_s": tokens / sum(times), "label_tokens_per_s": label_tokens / sum(times),
            "input_tokens": tokens, "label_tokens": label_tokens,
            "per_rank_mean_s": [statistics.mean(row["seconds"] for row in steps[options.warmup:])
                                for steps in rank_steps[name]],
            "peak_allocated_gib": [end["peak_allocated_bytes"] / 2**30 for end in rank_ends[name]],
            "peak_reserved_gib": [end["peak_reserved_bytes"] / 2**30 for end in rank_ends[name]],
            "loss_first": runs[name]["metrics"][0]["metrics"]["loss"],
            "loss_last": runs[name]["metrics"][-1]["metrics"]["loss"],
            "loss_first20_mean": statistics.mean(row["metrics"]["loss"] for row in runs[name]["metrics"][:20]),
            "loss_last20_mean": statistics.mean(row["metrics"]["loss"] for row in runs[name]["metrics"][-20:]),
            "final_parameters_equal_across_ranks": len({end["parameters"]["sha256"]
                                                       for end in rank_ends[name]}) == 1,
            "wrapped": runs[name]["modules"]["wrapped"], "provider": runs[name]["provider"],
        }
    for a, b in (("xmlir1", "tt1"), ("xmlir1", "tt2"), ("xmlir1", "xmlir2"), ("tt1", "tt2")):
        key = f"{a}_vs_{b}"
        summary["accuracy"][key] = {}
        for metric in ("loss", "grad_norm"):
            av = [row["metrics"][metric] for row in runs[a]["metrics"]]
            bv = [row["metrics"][metric] for row in runs[b]["metrics"]]
            summary["accuracy"][key][metric] = difference_stats(av, bv)
            summary["accuracy"][key][f"{metric}_windows"] = {
                f"{start+1}-{end}": difference_stats(av[start:end], bv[start:end])
                for start, end in ((0, 20), (20, min(100, count)), (100, count)) if end > start
            }
        summary["weights"][key] = weight_difference(
            options.directory / a / "evidence/initial_trainable.pt",
            options.directory / a / "evidence/final_trainable.pt",
            options.directory / b / "evidence/final_trainable.pt")
    xmlir_s = statistics.mean(summary["performance"][name]["step_mean_s"] for name in ("xmlir1", "xmlir2"))
    tt_s = statistics.mean(summary["performance"][name]["step_mean_s"] for name in ("tt1", "tt2"))
    summary["pooled_performance"] = {
        "xmlir_step_s": xmlir_s, "tt_step_s": tt_s,
        "tt_latency_gap_percent": 100 * (tt_s / xmlir_s - 1),
        "tt_speedup": xmlir_s / tt_s,
        "tt_throughput_gap_percent": 100 * (xmlir_s / tt_s - 1),
    }
    target = options.directory / "summary.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("steps", "all_rank_input_fingerprints_equal", "pooled_performance", "accuracy", "weights")},
                     ensure_ascii=False, indent=2))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
