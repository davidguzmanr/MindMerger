#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mail.utoronto.ca
#SBATCH --output=americas-mindmerger-%j.out
#SBATCH --error=americas-mindmerger-%j.out

#############################################################
# install the environment by loading in python and required packages
module load python/3.10.13
module load gcc/12.3
module load cuda/12.2
module load arrow/17.0.0 

source AfricanLLM-env/bin/activate
export HF_HOME=~/scratch/huggingface
#############################################################

# Redirect all output and errors from this point forward
exec > americas-mindmerger-${SLURM_JOB_ID}.out 2>&1

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo $HF_HOME

cd MindMerger
git checkout AmericasMindMerger

echo "mapping stage"
deepspeed run_training.py \
    --llm_path  meta-llama/Llama-2-7b-hf \
    --mt_path facebook/nllb-200-distilled-600M \
    --stage_name mapping \
    --task nli \
    --augmentation False \
    --save_name AmericasMindMergerLlama2 \
    --train_num 20000 \
    --dev_size 3000 \
    --train_batch_size 2 \
    --train_micro_batch_size_per_gpu 2 \
    --epoch_num 1 \
    --max_seq_len 200 \
    --max_gen_len 200 \
    --eval_batch_size 2

echo "augmentation stage"
deepspeed run_training.py \
    --llm_path  meta-llama/Llama-2-7b-hf \
    --mt_path facebook/nllb-200-distilled-600M \
    --stage_name augmentation \
    --task nli \
    --augmentation True \
    --save_name AmericasMindMergerLlama2 \
    --train_num 10000 \
    --dev_size 100 \
    --train_batch_size 1 \
    --train_micro_batch_size_per_gpu 1 \
    --epoch_num 1 \
    --max_seq_len 512 \
    --max_gen_len 512 \
    --eval_batch_size 1

echo "evaluation stage"
deepspeed run_evaluation.py \
    --task nli \
    --llm_path meta-llama/Llama-2-7b-hf \
    --mt_path facebook/nllb-200-distilled-600M \
    --augmentation True \
    --eval_batch_size 1 \
    --init_checkpoint outputs/AmericasMindMergerLlama2/nli/augmentation/pytorch_model.bin \
    --save_name AmericasMindMergerLlama2
