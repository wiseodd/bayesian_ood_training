import numpy as np
import pickle
import os, sys, argparse
from collections import defaultdict
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', default='CIFAR-10-C', choices=['CIFAR-10-C', 'rot-MNIST', 'trans-MNIST'],
    '--methods', default='all', choices=['LA', 'VB', 'all']
)
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'uniform', 'smooth'])
args = parser.parse_args()

path = f'./results/{args.dataset}/{args.ood_data}/'
_, _, filenames = next(os.walk(path))

methods_baselines = [
    'MAP-plain', 'DE-plain', 'MAP-oe', # 'MAP-dirlik', 'MAP-mixed', 'MAP-noneclass',
]
methods_VB = ['LLVB-plain', 'LLVB-noneclass', 'LLVB-dirlik', 'LLVB-mixed']
methods_LA = ['DLA-plain', 'DLA-noneclass', 'DLA-dirlik', 'DLA-mixed']

if args.methods == 'LA':
    methods = methods_LA
elif args.methods == 'VB':
    methods = methods_VB
else:
    methods = methods_baselines + methods_VB[-1:] + methods_LA[-1:]

# metrics = ['loglik', 'ece', 'brier', 'acc', 'mmc']
metrics = ['loglik', 'ece', 'brier', 'acc']
metric2str = {'acc': 'Acc.', 'mmc': 'MMC', 'ece': 'ECE',
              'brier': 'Brier', 'loglik': 'LogLik'}

method_strs = {
    'MAP-plain': 'MAP', 'MAP-oe': 'OE', 'DE-plain': 'DE',
    'MAP-dirlik': 'MAP-DL', 'MAP-mixed': 'MAP-MX', 'MAP-noneclass': 'MAP-NC-1',
    'DLA-plain': 'LA', 'DLA-dirlik': 'LA-SL', 'DLA-mixed': 'LA-ML', 'DLA-noneclass': 'LA-NC',
    'LLVB-plain': 'VB', 'LLVB-dirlik': 'VB-SL', 'LLVB-mixed': 'VB-ML', 'LLVB-noneclass': 'VB-NC'
}

N = 10000  # Test size for both MNIST and CIFAR-10

for method in methods:
    res = defaultdict(list)
    means = []
    stds = []

    for metric in metrics:
        vals = np.load(f'{path}/{metric}.npy', allow_pickle=True).item()
        decimal_place = 1 if metric not in ['loglik', 'brier'] else 3

        if args.dataset == 'CIFAR-10-C':
            # For all distortions and severities (80 in total)
            for distortion in vals[method].keys():
                # Ignore clean data
                if distortion == 'clean':
                    continue

                for severity in vals[method][distortion].keys():
                    # The value is a list, resulting from repetition
                    val = np.mean(vals[method][distortion][severity])

        elif args.dataset == 'rot-MNIST':
            for angle in range(15, 181, 15):  # Exclude clean test results
                val = np.mean(vals[method][str(angle)])
        elif args.dataset == 'trans-MNIST':
            for shift in range(2, 15, 2):  # Exclude clean
                val = np.mean(vals[method][str(shift)])

        if metric == 'loglik':
            val /= -N

        res[metric].append(val)

        # Average over all of them
        means.append(round(np.mean(res[metric]), decimal_place))
        stds.append(round(np.std(res[metric]), decimal_place))

    means, stds = np.array(means), np.array(stds)

    string = f'{method_strs[method]} & '
    string += ' & '.join([
        f'{mean}' for mean, std in zip(means, stds)
    ])
    string += ' \\\\'
    print(string)
