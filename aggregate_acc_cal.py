import numpy as np
import pickle
import os, sys, argparse
from collections import defaultdict
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'smooth'])
parser.add_argument('--nlp', action='store_true', default=False)
parser.add_argument('--aux_results', action='store_true', default=False)
args = parser.parse_args()

path = './results/'

if args.nlp:
    path += 'NLP/'
else:
    path += f'OOD/{args.ood_data}/'

_, _, filenames = next(os.walk(path))

if not args.nlp:
    if not args.aux_results:
        methods = [
            'MAP-plain', 'DE-plain', 'MAP-oe',
            'LLVB-plain', 'LLVB-noneclass', 'LLVB-dirlik', 'LLVB-mixed', 'LLVB-oe',
            'DLA-plain', 'DLA-noneclass', 'DLA-dirlik', 'DLA-mixed', 'DLA-oe'
        ]
        datasets = ['MNIST', 'FMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']
    else:
        methods = [
            'BBB-plain', 'BBB-noneclass', 'CSGHMC-plain', 'CSGHMC-noneclass', 'DE-noneclass'
        ]
        datasets = ['CIFAR10', 'CIFAR100']
else:
    methods = [
        'MAP-plain', 'DE-plain', 'MAP-oe',
        'LLLA-plain', 'LLLA-noneclass', 'LLLA-dirlik', 'LLLA-mixed', 'LLLA-oe'
    ]
    # datasets = ['SST', '20NG', 'TREC']
    datasets = ['SST', 'TREC']

method_strs = {
    # Baselines
    'MAP-plain': 'MAP', 'MAP-oe': 'OE', 'DE-plain': 'DE',
    #
    # VB
    'LLVB-plain': 'VB', 'LLVB-dirlik': '+SL', 'LLVB-mixed': '+ML',
    'LLVB-noneclass': '+NC', 'LLVB-oe': '+OE',
    #
    # LA
    'DLA-plain': 'LA', 'DLA-dirlik': '+SL', 'DLA-mixed': '+ML',
    'DLA-noneclass': '+NC', 'DLA-oe': '+OE',
    #
    # For NLP
    'LLLA-plain': 'LA', 'LLLA-noneclass': '+NC', 'LLLA-dirlik': '+DL',
    'LLLA-mixed': '+ML', 'LLLA-oe': '+OE',
    # Aux.
    'BBB-plain': 'Flipout', 'CSGHMC-plain': 'CSGHMC',
    'DE-noneclass': '+NC', 'BBB-noneclass': '+NC', 'CSGHMC-noneclass': '+NC'
}

dataset_strs = {
    'MNIST': 'MNIST', 'FMNIST': 'F-MNIST', 'CIFAR10': 'CIFAR-10',
    'SVHN': 'SVHN', 'CIFAR100': 'CIFAR-100'
}


def process_data(type):
    table_means = {method: [] for method in methods}
    table_stds = {method: [] for method in methods}

    for i, dset in enumerate(datasets):
        full_path = f'{path+dset}'
        if args.aux_results:
            full_path += '/aux'

        vals = np.load(f'{full_path}/{type}.npy', allow_pickle=True).item()
        vals = pd.DataFrame(vals)
        vals = vals.drop(columns=[col for col in vals.columns if col not in methods])

        means = {}
        for col in vals:
            # Mean over repetitions
            means[col] = [np.mean(val) for val in vals[col].values]
            vals[col] = means[col]

        for method in methods:
            table_means[method].append(vals[method].mean())
            table_stds[method].append(vals[method].sem())

    return table_means, table_stds


acc_means, acc_stds = process_data('acc')
ece_means, ece_stds = process_data('cal')

for i, method in enumerate(methods):
    acc_meanstd = zip(acc_means[method], acc_stds[method])
    ece_meanstd = zip(ece_means[method], ece_stds[method])

    # val_str = ' & '.join([f'{m:.1f}$\\pm${s:.1f}' for m, s in meanstd])

    val_str = ' & '.join([
        f'{m1:.1f}$\\pm${v1:.1f}\\,/\\,{m2:.1f}$\\pm${v2:.1f}'
        for (m1, v1), (m2, v2) in zip(acc_meanstd, ece_meanstd)
    ])
    print(f'{method_strs[method]} & {val_str} \\\\')

    # Separating baselines, VB, and LA
    if i in [2, 7]:
        print('\n\\midrule\n')
