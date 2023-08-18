#!/bin/bash
declare -a dsets=("MNIST" "FMNIST" "SVHN" "CIFAR10" "CIFAR100")

# Plain
for dset in "${dsets[@]}";
do
    for i in {1..5}  # Five random seeds
    do
        python train.py --dataset $dset --method plain --randseed $i
    done
done
