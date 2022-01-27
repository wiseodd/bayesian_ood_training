import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
import os, sys, argparse
sns.set_palette('colorblind')


parser = argparse.ArgumentParser()
parser.add_argument('--type', default='LA', choices=['LA', 'VB', 'all'])
parser.add_argument('--ood_dset', default='imagenet', choices=['imagenet', 'smooth'])
args = parser.parse_args()

methods_base = [
    'MAP-plain', 'DE-plain', 'MAP-oe',
    'MAP-dirlik', 'MAP-mixed', 'MAP-noneclass'
]
methods_VB = [
    'LLVB-plain',
    'LLVB-noneclass', #'LLVB-noneclass-extra',
    'LLVB-dirlik', 'LLVB-mixed'
]
methods_LA = [
    'DLA-plain',
    'DLA-noneclass', #'DLA-noneclass-extra',
    'DLA-dirlik', 'DLA-mixed'
]

if args.type == 'LA':
    methods = methods_LA
elif args.type == 'VB':
    methods = methods_VB
else:
    methods = methods_base + methods_VB + methods_LA

metric2str = {'acc': 'Acc.', 'mmc': 'MMC', 'ece': 'ECE',
              'brier': 'Brier', 'loglik': 'LogLik'}

method_strs = {
    'MAP-plain': 'MAP', 'MAP-oe': 'OE', 'DE-plain': 'DE',
    'MAP-dirlik': 'MAP-SL', 'MAP-mixed': 'MAP-ML', 'MAP-noneclass': 'MAP-NC',
    'DLA-plain': 'LA', 'DLA-dirlik': 'LA-SL', 'DLA-mixed': 'LA-ML',
    'DLA-noneclass': 'LA-NC', 'LLVB-plain': 'VB', 'LLVB-dirlik': 'VB-SL',
    'LLVB-mixed': 'VB-ML', 'LLVB-noneclass': 'VB-NC',
    'MAP-noneclass-extra': 'MAP-NC-2', 'DLA-noneclass-extra': 'LA-NC-2',
    'LLVB-noneclass-extra': 'VB-NC-2'
}

path = f'results/CIFAR-10-C/{args.ood_dset}'
N = 10000  # n test points


def plot(metric='ece'):
    metric_str = metric2str[metric]
    data = {'Method': [], 'Severity': [], metric_str: []}
    vals = np.load(f'{path}/{metric}.npy', allow_pickle=True).item()

    for method in methods:
        for distortion in vals[method].keys():
            if distortion == 'clean':
                continue

            for severity in vals[method][distortion].keys():
                data['Method'].append(method_strs[method])
                data['Severity'].append(int(severity))

                val = vals[method][distortion][severity][0]

                if metric == 'loglik':
                    val /= -N

                data[metric_str].append(val)


    df = pd.DataFrame(data)

    df_filtered = df
    # df_filtered = df[df['Method'].isin(methods)]
    # df_filtered['Method'] = df_filtered['Method'].replace('DLA', 'LA')
    # df_filtered['Method'] = df_filtered['Method'].replace('LLVB', 'VB')

    sns.boxplot(
        data=df_filtered, x='Severity', y=metric_str, hue='Method', fliersize=0, width=0.5
    )

    dir_name = f'figs/CIFAR-10-C/{args.ood_dset}'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    tikzplotlib.save(f'{dir_name}/cifar10c_{args.type}_{metric}.tex')
    plt.savefig(f'{dir_name}/cifar10c_{args.type}_{metric}.pdf', bbox_inches='tight')
    plt.close()


plot(metric='loglik')
plot(metric='ece')
plot(metric='brier')
plot(metric='mmc')
