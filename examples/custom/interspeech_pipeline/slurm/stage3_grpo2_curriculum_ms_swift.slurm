#!/bin/bash
#SBATCH --account=OD-237777
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=16:00:00
#SBATCH --output=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.out
#SBATCH --error=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/logs/%x_%j.err

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
conda activate deepfake

cd /scratch3/che489/Ha/interspeech/training/ms-swift
bash examples/custom/interspeech_pipeline/scripts/04_stage3_grpo2_curriculum.sh

