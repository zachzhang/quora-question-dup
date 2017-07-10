#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=zachzhang
#SBATCH --time=4:00:00


module purge
module load nltk/3.2.2 
module swap python/intel  python3/intel/3.5.3
module load scikit-learn/intel/0.18.1
module load tensorflow/python3.5/1.1.0 

cd /home/zz1409/Quora/question-answer

python3 train.py
