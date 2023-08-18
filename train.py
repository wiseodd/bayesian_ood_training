import torch
from torch import optim
from torch import distributions as dists
from models.models import LeNetMadry
from models import wideresnet
from util.evaluation import *
import util.dataloaders as dl
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import os
import math
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "FMNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='CIFAR10')
parser.add_argument('--method', choices=['noneclass', 'oe', 'plain'], default='plain')
parser.add_argument('--ood_data', default='imagenet', choices=['imagenet', 'smooth', 'uniform'])
parser.add_argument('--randseed', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

assert args.dataset in ['MNIST', 'FMNIST', 'CIFAR10', 'SVHN', 'CIFAR100'], 'Invalid dataset.'

batch_size = 128
n_epochs = 100
path = './pretrained_models'

OOD_TRAINING = args.method in ['noneclass', 'oe']

if OOD_TRAINING:
    path += f'/{args.ood_data}'

model_suffix = f'_{args.method}_{args.randseed}'

train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
num_classes = 100 if args.dataset == 'CIFAR100' else 10

if args.method == 'noneclass':
    num_classes += 1

print(len(train_loader.dataset), len(test_loader.dataset))

if args.dataset in ['MNIST', 'FMNIST']:
    model = LeNetMadry(num_classes)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
else:
    depth = 16
    widen_factor = 4
    model = wideresnet.WideResNet(depth, widen_factor, num_classes)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

model.cuda()
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction='mean')

## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(train_loader))
pbar = trange(n_epochs)

## For automatic-mixed-precision
scaler = amp.GradScaler()

if OOD_TRAINING and args.ood_data == 'imagenet':
    ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)

for epoch in pbar:
    if OOD_TRAINING:
        # Get new data every epoch to avoid overfitting to noises
        if args.ood_data == 'imagenet':
            # Induce a randomness in the OOD batch since num_ood_data >> num_indist_data
            # The shuffling of ood_loader only happens when all batches are already yielded
            ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))
        elif args.ood_data == 'smooth':
            ood_loader = dl.Noise(train=True, dataset=args.dataset, batch_size=batch_size)
        elif args.ood_data == 'uniform':
            ood_loader = dl.UniformNoise(train=True, dataset=args.dataset, size=len(train_loader.dataset), batch_size=batch_size)

        data_iter = enumerate(zip(train_loader, ood_loader))
    else:
        data_iter = enumerate(train_loader)

    train_loss = 0

    for batch_idx, data in data_iter:
        model.train()
        opt.zero_grad()

        if OOD_TRAINING:
            (x_in, y_in), (x_out, _) = data
            m = len(x_in)  # Batch size
            x_out = x_out[:m]  # To ensure the same batch size
            if args.method == 'noneclass':
                # Last class, zero-indexed
                y_out = (num_classes-1) * torch.ones(m).long()
            else:
                y_out = -1 * torch.ones(m).long()  # Just a dummy, we don't need this
        else:
            x_in, y_in = data
            m = len(x_in)  # Batch size

        x_in, y_in = x_in.cuda(non_blocking=True), y_in.long().cuda(non_blocking=True)

        if OOD_TRAINING:
            x_out, y_out = x_out.cuda(non_blocking=True), y_out.cuda(non_blocking=True)
            x = torch.cat([x_in, x_out], dim=0)
            y = torch.cat([y_in, y_out], dim=0)
        else:
            x, y = x_in, y_in

        with amp.autocast():
            outputs = model(x).squeeze()

            if args.method in ['plain', 'noneclass']:
                loss = criterion(outputs, y)
            elif args.method == 'oe':
                loss = criterion(outputs[:m], y_in)
                # Sum all log-probs directly (0.5 following Hendrycks)
                loss += -0.5*1/num_classes*torch.log_softmax(outputs[m:], 1).mean()

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss = 0.9*train_loss + 0.1*loss.item()

    model.eval()
    pred = predict(test_loader, model).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; loss: {train_loss:.3f}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )

if not os.path.exists(path):
    os.makedirs(path)

torch.save(model.state_dict(), f'{path}/{args.dataset}{model_suffix}.pt')

## Try loading and testing
model.load_state_dict(torch.load(f'{path}/{args.dataset}{model_suffix}.pt'))
model.eval()

print()

## In-distribution
py_in = predict(test_loader, model).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')
