#!/bin/bash
declare -a dsets=("MNIST" "FMNIST" "SVHN" "CIFAR10" "CIFAR100")
declare -a ood_methods=("noneclass" "oe")

# OOD methods
for dset in "${dsets[@]}";
do
    for ood_method in "${ood_methods[@]}";
    do
        for i in {1..5}  # Five random seeds
        do
            python train.py --dataset $dset --method $ood_method --ood_data imagenet --randseed $i
        done
    done
done
