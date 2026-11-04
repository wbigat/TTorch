#!/bin/bash
# B1 训练级验证：真实 swift-megatron sft，CP=2 + packing（THD 路径）。
#
# 红绿结构：
#   RUN A（xmlir，TT 关）：若生产配置真实触发 THD non-batched fallback，
#                         无补丁保护应崩于 "cp_group must be provided"。
#   RUN B（tt，TT 开）：同配置应正常训练，补丁兜底。
# 若 A 也正常训练完 → 该配置没触发 fallback，训练级验证对该配置不适用
# （算子级多卡验证 b1_thd_rope_e2e.py 的 repro-cp/verify-cp 为准）。
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "${COMPARISON_ENV:-/root/dl/runtime/script/env.sh}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
export XPYTORCH_RUN_ENHANCE=0
export BKCL_USE_RDMA=0 BKCL_RDMA_VERBS=0 BKCL_RDMA_FORCE_TREE=0 BKCL_FORCE_L3_RDMA=0
export XME_PLUGIN_ENABLE=0 XSWIFT_PLUGIN_ENABLE=0
export HF_HUB_OFFLINE=1 MODELSCOPE_OFFLINE=1
unset TT_ACTIVATED_PID

OUT=${B1_OUTPUT:-${ROOT}/.benchmarks/b1_cp_$(date +%Y%m%d_%H%M%S)}
STEPS=${B1_STEPS:-5}
mkdir -p "${OUT}"
git -C "${ROOT}" rev-parse HEAD > "${OUT}/commit.txt" 2>/dev/null || true

TRAIN_ARGS=(
    --model "${TT_QWEN_MODEL:-/apdcephfs_bj36/share_306186457/download/huggingface/Qwen/Qwen3.5-0.8B}"
    --dataset "${TT_SFT_DATA:-/cfs_hunyuanaide/172076/caption_merged_messages_sample150_llamafactory_bj.jsonl}"
    --template qwen3_5
    --tuner_type lora --target_modules all-linear
    --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05
    --tensor_model_parallel_size 1 --pipeline_model_parallel_size 1
    --context_parallel_size "${B1_CP:-2}" --expert_model_parallel_size 1 --sequence_parallel false
    --micro_batch_size 1 --global_batch_size 4 --train_iters "${STEPS}"
    --lr 5e-5 --lr_warmup_fraction 0.05 --min_lr 5e-6
    --logging_steps 1 --save_steps 10000 --eval_steps 10000 --eval_iters 0
    --save_safetensors true --merge_lora false --no_save_optim true --no_save_rng true
    --max_length 8192 --truncation_strategy delete
    --attention_backend flash --padding_free true --packing true --group_by_length false
    --seed 42 --cross_entropy_loss_fusion true --cross_entropy_fusion_impl te
    --split_dataset_ratio 0 --lazy_tokenize false --load_from_cache_file true
    --finetune true --dataloader_num_workers 0 --dataset_num_proc 4 --check_model false
)

for RUN in xmlir tt; do
    if [[ ${RUN} == xmlir ]]; then
        export TT_PLUGIN_ENABLE=0 TT_PROVIDER_AUTO_WRAP=0 TT_RESTORE_VENDOR=none
    else
        export TT_PLUGIN_ENABLE=1 TT_PROVIDER_AUTO_WRAP=1 TT_RESTORE_VENDOR=perf
    fi
    mkdir -p "${OUT}/${RUN}"
    echo "$(date -Iseconds) START ${RUN} cp=${B1_CP:-2} steps=${STEPS}" | tee -a "${OUT}/progress.log"
    torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29541 \
        "${ROOT}/tests/e2e/compare_qwen_four_rank.py" \
        --comparison-mode "${RUN}" --evidence-dir "${OUT}/${RUN}/evidence" \
        "${TRAIN_ARGS[@]}" --output_dir "${OUT}/${RUN}/checkpoint" \
        > "${OUT}/${RUN}/train.log" 2>&1
    STATUS=$?
    echo "${STATUS}" > "${OUT}/${RUN}/exit_code.txt"
    echo "$(date -Iseconds) END ${RUN} exit=${STATUS}" | tee -a "${OUT}/progress.log"
done

echo "=== B1 训练级验证汇总 ==="
for RUN in xmlir tt; do
    STATUS=$(cat "${OUT}/${RUN}/exit_code.txt")
    FB=$(grep -c "batched rope not applicable" "${OUT}/${RUN}/train.log" || true)
    CPG=$(grep -c "cp_group must be provided" "${OUT}/${RUN}/train.log" || true)
    LOSS=$(grep -oE "lm loss: [0-9.E+-]+" "${OUT}/${RUN}/train.log" | tail -1 || true)
    echo "${RUN}: exit=${STATUS} fallback警告=${FB} cp_group报错=${CPG} ${LOSS}"
done
echo "日志目录: ${OUT}"
