#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=6-00:29:00

#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH --constraint="gpu32|gpu80"

#SBATCH --output=../../slurm_logs/%j.txt
#SBATCH --job-name=SYNGEN

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user ra578108@ucf.edu


python inference.py #train.py
