#!/bin/bash


## BBB - Plain
python train_bbb.py --dataset CIFAR10
python train_bbb.py --dataset CIFAR100
python train_bbb.py --dataset SVHN


## BBB with OOD
python train_bbb.py --dataset CIFAR10 --method noneclass
python train_bbb.py --dataset CIFAR100 --method noneclass
python train_bbb.py --dataset SVHN --method noneclass


## CSGHMC - Plain
python train_csghmc.py --dataset CIFAR10
python train_csghmc.py --dataset CIFAR100
python train_csghmc.py --dataset SVHN


## CSGHMC with OOD
python train_csghmc.py --dataset CIFAR10 --method noneclass
python train_csghmc.py --dataset CIFAR100 --method noneclass
python train_csghmc.py --dataset SVHN --method noneclass


## DE with OOD
python train_de_ood.py --dataset CIFAR10 --method noneclass
python train_de_ood.py --dataset CIFAR100 --method noneclass
