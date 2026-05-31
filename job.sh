#!/bin/bash
#SBATCH --job-name=medical_qa
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medqa

cd /users/student/pg/pg23/shubhranil/Medical-QA

# python rag/store.py --subset "pqa_labeled"                                                      # Create mode
python rag/inference.py --question "What are the symptoms of diabetes?" --num_references 5        # Retrieve mode
