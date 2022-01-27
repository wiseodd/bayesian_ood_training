#!/bin/bash

# # Plain (can be used for Laplace approxs.)
python train.py --dataset MNIST
python train.py --dataset FMNIST
python train.py --dataset SVHN
python train.py --dataset CIFAR10
python train.py --dataset CIFAR100


# # MAP-OOD, for Laplace
python train.py --dataset MNIST --method dirlik
python train.py --dataset FMNIST --method dirlik
python train.py --dataset SVHN --method dirlik
python train.py --dataset CIFAR10 --method dirlik
python train.py --dataset CIFAR100 --method dirlik

python train.py --dataset MNIST --method noneclass
python train.py --dataset FMNIST --method noneclass
python train.py --dataset SVHN --method noneclass
python train.py --dataset CIFAR10 --method noneclass
python train.py --dataset CIFAR100 --method noneclass


# # OE
python train.py --dataset MNIST --oe
python train.py --dataset FMNIST --oe
python train.py --dataset SVHN --oe
python train.py --dataset CIFAR10 --oe
python train.py --dataset CIFAR100 --oe

# # DE
for i in {1..4}
do
    python train.py --dataset MNIST --de
done

for i in {1..4}
do
    python train.py --dataset FMNIST --de
done

for i in {1..4}
do
    python train.py --dataset SVHN --de
done

for i in {1..4}
do
    python train.py --dataset CIFAR10 --de
done

for i in {1..4}
do
    python train.py --dataset CIFAR100 --de
done


# # VB
python train_vb.py --dataset MNIST
python train_vb.py --dataset FMNIST
python train_vb.py --dataset SVHN
python train_vb.py --dataset CIFAR10
python train_vb.py --dataset CIFAR100


# VB with OOD
python train_vb.py --dataset MNIST --method dirlik
python train_vb.py --dataset FMNIST --method dirlik
python train_vb.py --dataset SVHN --method dirlik
python train_vb.py --dataset CIFAR10 --method dirlik
python train_vb.py --dataset CIFAR100 --method dirlik

python train_vb.py --dataset MNIST --method oe
python train_vb.py --dataset FMNIST --method oe
python train_vb.py --dataset SVHN --method oe
python train_vb.py --dataset CIFAR10 --method oe
python train_vb.py --dataset CIFAR100 --method oe

python train_vb.py --dataset MNIST --method noneclass
python train_vb.py --dataset FMNIST --method noneclass
python train_vb.py --dataset SVHN --method noneclass
python train_vb.py --dataset CIFAR10 --method noneclass
python train_vb.py --dataset CIFAR100 --method noneclass


## SMOOTH NOISE
## ----------------------------------

python train.py --dataset MNIST --oe --ood_data smooth
python train.py --dataset FMNIST --oe --ood_data smooth
python train.py --dataset SVHN --oe --ood_data smooth
python train.py --dataset CIFAR10 --oe --ood_data smooth
python train.py --dataset CIFAR100 --oe --ood_data smooth

python train.py --dataset MNIST --method noneclass --ood_data smooth
python train.py --dataset FMNIST --method noneclass --ood_data smooth
python train.py --dataset SVHN --method noneclass --ood_data smooth
python train.py --dataset CIFAR10 --method noneclass --ood_data smooth
python train.py --dataset CIFAR100 --method noneclass --ood_data smooth

python train.py --dataset MNIST --method dirlik --ood_data smooth
python train.py --dataset FMNIST --method dirlik --ood_data smooth
python train.py --dataset SVHN --method dirlik --ood_data smooth
python train.py --dataset CIFAR10 --method dirlik --ood_data smooth
python train.py --dataset CIFAR100 --method dirlik --ood_data smooth

python train_vb.py --dataset MNIST --method oe --ood_data smooth
python train_vb.py --dataset FMNIST --method oe --ood_data smooth
python train_vb.py --dataset SVHN --method oe --ood_data smooth
python train_vb.py --dataset CIFAR10 --method oe --ood_data smooth
python train_vb.py --dataset CIFAR100 --method oe --ood_data smooth

python train_vb.py --dataset MNIST --method noneclass --ood_data smooth
python train_vb.py --dataset FMNIST --method noneclass --ood_data smooth
python train_vb.py --dataset SVHN --method noneclass --ood_data smooth
python train_vb.py --dataset CIFAR10 --method noneclass --ood_data smooth
python train_vb.py --dataset CIFAR100 --method noneclass --ood_data smooth

python train_vb.py --dataset MNIST --method dirlik --ood_data smooth
python train_vb.py --dataset FMNIST --method dirlik --ood_data smooth
python train_vb.py --dataset SVHN --method dirlik --ood_data smooth
python train_vb.py --dataset CIFAR10 --method dirlik --ood_data smooth
python train_vb.py --dataset CIFAR100 --method dirlik --ood_data smooth

