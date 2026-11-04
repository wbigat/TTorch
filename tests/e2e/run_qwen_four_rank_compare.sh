#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${COMPARISON_ENV:-/root/dl/runtime/script/env.sh}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export NNODES=1 NPROC_PER_NODE=4 WORLD_GPUS=4
export XPYTORCH_RUN_ENHANCE=0
export BKCL_USE_RDMA=0 BKCL_RDMA_VERBS=0 BKCL_RDMA_FORCE_TREE=0 BKCL_FORCE_L3_RDMA=0
export XME_PLUGIN_ENABLE=0 XSWIFT_PLUGIN_ENABLE=0
export HF_HUB_OFFLINE=1 MODELSCOPE_OFFLINE=1
unset TT_ACTIVATED_PID

OUT=${COMPARISON_OUTPUT:-${ROOT}/.benchmarks/qwen4_20261029}
STEPS=${COMPARISON_STEPS:-40}
mkdir -p "${OUT}"
git -C "${ROOT}" rev-parse HEAD > "${OUT}/commit.txt"
cp "${COMPARISON_ENV:-/root/dl/runtime/script/env.sh}" "${OUT}/base_environment.sh"

TRAIN_ARGS=(
    --model "${TT_QWEN_MODEL:-/apdcephfs_bj36/share_306186457/download/huggingface/Qwen/Qwen3.5-0.8B}"
    --dataset "${TT_SFT_DATA:-/cfs_hunyuanaide/172076/caption_merged_messages_sample150_llamafactory_bj.jsonl}"
    --template qwen3_5
    --tuner_type lora --target_modules all-linear
    --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05
    --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1
    --context_parallel_size 1 --expert_model_parallel_size 1 --sequence_parallel false
    --micro_batch_size 1 --global_batch_size 4 --train_iters "${STEPS}"
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1
    --lr 5e-5 --lr_warmup_fraction 0.05 --min_lr 5e-6
    --logging_steps 1 --save_steps 10000 --eval_steps 10000 --eval_iters 0
    --save_safetensors true --merge_lora false --no_save_optim true --no_save_rng true
    --max_length 8192 --truncation_strategy delete
    --attention_backend flash --padding_free true --packing false --group_by_length false
    --seed 42 --cross_entropy_loss_fusion true --cross_entropy_fusion_impl te
    --loss_scale ignore_empty_think --add_non_thinking_prefix true
    --split_dataset_ratio 0 --lazy_tokenize false --load_from_cache_file true
    --enable_channel_loss true --finetune true
    --freeze_llm false --freeze_vit true --freeze_aligner true
    --dataloader_num_workers 0 --dataset_num_proc 4 --check_model false
    --report_to tensorboard
)
printf '%q ' "${TRAIN_ARGS[@]}" > "${OUT}/training_arguments.txt"
printf '\n' >> "${OUT}/training_arguments.txt"

for RUN in ${COMPARISON_RUNS:-xmlir1 tt1 tt2 xmlir2}; do
    if [[ ${RUN} == xmlir* ]]; then
        MODE=xmlir
        export TT_PLUGIN_ENABLE=0 TT_PROVIDER_AUTO_WRAP=0 TT_RESTORE_VENDOR=none
    elif [[ ${RUN} == tt* ]]; then
        MODE=tt
        export TT_PLUGIN_ENABLE=1 TT_PROVIDER_AUTO_WRAP=1 TT_RESTORE_VENDOR=perf
    else
        echo "Invalid comparison run: ${RUN}" >&2
        exit 2
    fi
    mkdir -p "${OUT}/${RUN}"
    echo "$(date -Iseconds) START ${RUN}" | tee -a "${OUT}/progress.log"
    set +e
    torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29529 \
        "${ROOT}/tests/e2e/compare_qwen_four_rank.py" \
        --comparison-mode "${MODE}" --evidence-dir "${OUT}/${RUN}/evidence" \
        "${TRAIN_ARGS[@]}" --output_dir "${OUT}/${RUN}/checkpoint" \
        > "${OUT}/${RUN}/train.log" 2>&1
    STATUS=$?
    set -e
    printf '%s\n' "${STATUS}" > "${OUT}/${RUN}/exit_code.txt"
    echo "$(date -Iseconds) END ${RUN} exit=${STATUS}" | tee -a "${OUT}/progress.log"
    if [[ ${STATUS} -ne 0 ]]; then
        tail -n 60 "${OUT}/${RUN}/train.log"
        exit "${STATUS}"
    fi
done
