import numpy as np
import pickle
import os, sys, argparse
from collections import defaultdict
import pandas as pd
import scipy.stats as st


parser = argparse.ArgumentParser()
parser.add_argument('--type', default='auroc', choices=['auroc', 'auprc', 'fpr95', 'mmc', 'acc', 'cal', 'acc_ece'])
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
parser.add_argument('--averaged', action='store_true', default=False)
parser.add_argument('--nlp', action='store_true', default=False)
parser.add_argument('--aux_results', action='store_true', default=False)
args = parser.parse_args()

path = './results/'

if args.nlp:
    path += 'NLP/'
else:
    path += f'OOD/{args.ood_data}/'

methods_base = [
    'MAP-plain', 'DE-plain', 'MAP-oe'
]
methods_image_essential = [
    'LLVB-plain', 'LLVB-mixed',
    'DLA-plain', 'DLA-mixed',
]
methods_image_all = [
    # 'MAP-dirlik', 'MAP-mixed', 'MAP-noneclass',
    'LLVB-plain', 'LLVB-noneclass', 'LLVB-noneclass-extra', 'LLVB-dirlik', 'LLVB-mixed', 'LLVB-oe',
    'DLA-plain', 'DLA-noneclass', 'DLA-noneclass-extra', 'DLA-dirlik', 'DLA-mixed', 'DLA-oe'
]
methods_nlp = [
    # 'LLVB-plain', 'LLVB-ood'
    'LLLA-plain', 'LLLA-noneclass', 'LLLA-noneclass-extra', 'LLLA-dirlik', 'LLLA-mixed', 'LLLA-oe',
]
# For auxiliary results
methods_aux = [
    'BBB-plain', 'BBB-noneclass', 'CSGHMC-plain', 'CSGHMC-noneclass', 'DE-noneclass',
]

methods = methods_base

if args.nlp:
    methods += methods_nlp
    datasets = ['SST', '20NG', 'TREC']
else:
    if not args.aux_results:
        methods += methods_image_all
        datasets = ['MNIST', 'FMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']
    else:
        methods = methods_aux
        datasets = ['CIFAR10', 'CIFAR100']

if args.type in ['acc', 'cal'] and ('LLVB-noneclass-extra' in methods or 'DLA-noneclass-extra' in methods):
    methods.remove('LLVB-noneclass-extra')
    methods.remove('DLA-noneclass-extra')

test_dsets = {
    'MNIST': ['FMNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10', 'UniformNoise', 'Noise'],
    'FMNIST': ['MNIST', 'EMNIST', 'KMNIST', 'GrayCIFAR10', 'UniformNoise', 'Noise'],
    'SVHN': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D', 'UniformNoise', 'Noise'],
    'CIFAR10': ['SVHN', 'LSUN', 'CIFAR100', 'FMNIST3D', 'UniformNoise', 'Noise'],
    'CIFAR100': ['CIFAR10', 'LSUN', 'CIFAR100', 'FMNIST3D', 'UniformNoise', 'Noise'],
}

method_strs = {
    'MAP-plain': 'MAP', 'MAP-oe': 'OE', 'DE-plain': 'DE',
    #
    'DLA-plain': 'LA', 'DLA-dirlik': '+SL', 'DLA-mixed': '+ML', 'DLA-oe': '+OE',
    'DLA-noneclass': '+NC-1', 'DLA-noneclass-extra': '+NC-2',
    #
    'LLVB-plain': 'VB', 'LLVB-dirlik': '+SL', 'LLVB-oe': '+OE',
    'LLVB-mixed': '+ML', 'LLVB-noneclass': '+NC-1', 'LLVB-noneclass-extra': '+NC-2',
    # For NLP
    'LLLA-plain': 'LA', 'LLLA-noneclass': '+NC-1', 'LLLA-noneclass-extra': '+NC-2', 'LLLA-dirlik': '+DL', 'LLLA-mixed': '+ML', 'LLLA-oe': '+OE',
    # Aux.
    'BBB-plain': 'Flipout', 'CSGHMC-plain': 'CSGHMC',
    'DE-noneclass': '+NC', 'BBB-noneclass': '+NC', 'CSGHMC-noneclass': '+NC'
}

dataset_strs = {
    'MNIST': 'MNIST', 'FMNIST': 'F-MNIST', 'CIFAR10': 'CIFAR-10',
    'SVHN': 'SVHN', 'CIFAR100': 'CIFAR-100', 'EMNIST': 'E-MNIST', 'KMNIST': 'K-MNIST',
    'UniformNoise': 'Uniform', 'Noise': 'Smooth', 'FarAway': 'Asymptotic',
    'GrayCIFAR10': 'CIFAR-Gr', 'LSUN': 'LSUN-CR', 'FMNIST3D': 'FMNIST-3D',
    '20NG': '20NG', 'SST': 'SST', 'TREC': 'TREC',
    'IMDB': 'IMDB', 'SNLI': 'SNLI', 'Multi30k': 'Multi30k', 'WMT16': 'WMT16'
}

if args.type in ['acc', 'cal']:
    args.averaged = True

table_means = {method: [] for method in methods}
table_stds = {method: [] for method in methods}

for i, dset in enumerate(datasets):
    fname = f'{path+dset}'

    if args.aux_results:
        fname += '/aux'

    vals = np.load(f'{fname}/{args.type}.npy', allow_pickle=True).item()
    vals = pd.DataFrame(vals)
    vals = vals.drop(columns=[col for col in vals.columns if col not in methods])

    if args.type not in ['acc', 'mmc', 'cal']:
        if not args.nlp :
            vals = vals.drop(index=[idx for idx in vals.index if idx not in test_dsets[dset]])

    if not args.averaged:
        vals = pd.DataFrame(vals).transpose()  # Dataset-major

        if args.aux_results:
            # Reorder methods
            vals = vals.transpose()[methods_aux].transpose()

    means = {}
    stds = {}

    for col in vals:
        # Mean over repetitions
        if not args.aux_results:
            means[col] = [np.mean(val) for val in vals[col].values]
            stds[col] = [st.sem(val) for val in vals[col].values]
        else:
            means[col] = [np.mean(v) for v in vals[col]]
            stds[col] = [st.sem(v) for v in vals[col]]

    df_means = vals.copy()
    df_stds = vals.copy()
    for col in vals:
        df_means[col] = means[col]
        df_stds[col] = stds[col]

    def print_bold(dset_name, means, stds, mark_bold=False):
        if mark_bold:
            means = [round(m, 1) for m in means]
            top_means = np.max(means) if args.type in ['auroc', 'auprc'] else np.min(means)
            tops = np.argwhere(means == top_means).flatten()
            bolds = [True if j in tops else False for j, _ in enumerate(means)]
        else:
            bolds = [False]*len(means)

        str = f'{dataset_strs[dset_name]} & '
        str += ' & '.join([
            f'\\textbf{{{m:.1f}}}$\\pm${s:.1f}' if bold else f'{m:.1f}$\\pm${s:.1f}'
            for m, s, bold in zip(means, stds, bolds)
        ])
        str += ' \\\\'
        print(str)

    if not args.averaged:
        if args.aux_results:
            # Rearrange methods
            means = df_means.transpose()[methods_aux].transpose()
            stds = df_stds.transpose()[methods_aux].transpose()

        # Print LaTex code
        if args.type in ['auroc', 'auprc', 'fpr95']:
            print(f'\\textbf{{{dataset_strs[dset]}}} & & & & & & \\\\')
        else:  # MMC
            str = f'\\textbf{{{dataset_strs[dset]}}} & '

            try:
                str += ' & '.join([f'{v_:.1f}' for v_ in means[dset]])
            except KeyError:
                str += ' & '.join([f'{v_:.1f}' for v_ in means[dset.lower()]])

            str += ' \\\\'
            print(str)

        for k in df_means.keys():
            k = k.upper() if k in ['sst', '20ng', 'trec'] else k
            if args.type == 'mmc' and k == dset:
                continue
            if k == 'FarAway':
                continue
            if args.ood_data == 'smooth' and k == 'Noise':
                continue
            print_bold(k, df_means[k], df_stds[k])

        if i < len(datasets)-1:
            print('\n\\midrule\n')
    else:  # Averaged
        for method in methods:
            table_means[method].append(vals[method].mean())
            table_stds[method].append(vals[method].sem())

if args.averaged:
    for i, method in enumerate(methods):
        # meanstd = zip(table_means[method], table_stds[method])
        # val_str = ' & '.join([f'{m:.1f}$\\pm${s:.1f}' for m, s in meanstd])
        val_str = ' & '.join([f'{m:.1f}' for m in table_means[method]])
        print(f'{method_strs[method]} & & {val_str} \\\\')

        # Separating baselines, VB, and LA
        if i in ([2] if args.nlp else [2, 8]):
            print('\n\\midrule\n')
