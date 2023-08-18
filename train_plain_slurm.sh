#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=train_ood_%A_%a.out
#SBATCH --array=1-5

scontrol show job $SLURM_JOB_ID

declare -a dsets=("MNIST" "FMNIST" "SVHN" "CIFAR10" "CIFAR100")

# Plain
for dset in "${dsets[@]}";
do
    python train.py --dataset $dset --method plain --randseed $SLURM_ARRAY_TASK_ID
done
