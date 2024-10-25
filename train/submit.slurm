#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=1:59:59

#SBATCH -c 12
#SBATCH --mem=150G
#SBATCH --constraint="gpu32"

#SBATCH --output=slurm_logs/%j.txt
#SBATCH --job-name=SYNGEN

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user ra578108@ucf.edu

python train.py
