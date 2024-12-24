#!/bin/bash
#SBATCH -A pi_allan_umass_edu
#SBATCH --job-name=pythia6.9b
#SBATCH -c 8
#SBATCH --nodes=1
#SBATCH --mem=400G 
#SBATCH --partition=superpod-a100
#SBATCH --gpus=a100:8 
#SBATCH -t 168:00:00 
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --mail-type=ALL 



# # Generate proper hostfile format for DeepSpeed
# HOSTFILE="hostfile_${SLURM_JOB_ID}"
# scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
#     echo "$host slots=8" >> $HOSTFILE
# done



# Setup the Conda environment and working directory
source /project/pi_allan_umass_edu/anijasure/rankllama3/miniconda3/etc/profile.d/conda.sh
conda activate finetune_env
export HF_HOME=/project/pi_allan_umass_edu/anijasure/rankllama3/cache

# export PYTHONPATH=/project/pi_allan_umass_edu/anijasure/rankllama3/reranking_finetune


python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"

module load cuda/12.6


# Log allocated resources
# echo "Number of nodes allocated: $SLURM_NNODES"
# echo "Node list: $SLURM_NODELIST"

export TRITON_CACHE_DIR=./trition_cache

deepspeed --master_port 60000 --num_gpus 1 --module train_pythia \
  --deepspeed ./deepspeed/ds_zero3_config.json \
  --output_dir ./model_outputs/aug_model_pythia_6_9b \
  --model_name_or_path EleutherAI/pythia-6.9b \
  --lora \
  --lora_target_modules query_key_value,dense,dense_h_to_4h,dense_4h_to_h \
  --save_steps 50 \
  --dataset_name Tevatron/msmarco-passage-aug \
  --query_prefix "query: " \
  --passage_prefix "document: " \
  --bf16 \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --rerank_max_len $(( 32 + 164 )) \
  --num_train_epochs 1 \
  --logging_steps 100 \
  --gradient_accumulation_steps 4

