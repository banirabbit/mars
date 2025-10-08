#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate servicereco
export WANDB_MODE=offline
MODEL_NAME="BAAI/bge-base-en-v1.5"
TRAIN_DATA="/home/yinzijie/code/servicerag/src/retrieval/data/train_examples_0.55.jsonl"
OUTPUT_DIR="/home/yinzijie/code/servicerag/src/retrieval/output/finetuned_bge_singlegpu_$(date +%Y-%m-%d_%H-%M)_0.55"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  -m FlagEmbedding.finetune.embedder.encoder_only.base \
  --model_name_or_path ${MODEL_NAME} \
  --train_data ${TRAIN_DATA} \
  --output_dir ${OUTPUT_DIR} \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --train_group_size 4 \
  --learning_rate 1e-5 \
  --query_max_len 128 \
  --passage_max_len 512 \
  --temperature 0.02 \
  --negatives_cross_device \
  --normalize_embeddings True \
  --dataloader_drop_last True \
  --logging_steps 10 \
  --save_steps 500

echo "✅ 训练完成，模型保存到: ${OUTPUT_DIR}"
