import os
import torch
import torch.nn.functional as F
from torch import optim
from bayes.vb import bbb
import util.dataloaders as dl
import util.evaluation as evalutil
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import math
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CIFAR10', choices=['SVHN', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--estimator', default='flipout', choices=['reparam', 'flipout'])
parser.add_argument('--method', choices=['noneclass', 'plain'], default='plain')
parser.add_argument('--var0', type=float, default=1, help='Gaussian prior variance. If None, it will be computed to emulate weight decay')
parser.add_argument('--tau', type=float, default=0.1, help='Tempering parameter for the KL-term')
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

batch_size = 128
n_epochs = 100
EPS = 1e-30
path = './pretrained_models'

OOD_TRAINING = args.method in ['noneclass']
model_suffix = f'_{args.method}'

train_loader = dl.datasets_dict[args.dataset](train=True, batch_size=batch_size)
test_loader = dl.datasets_dict[args.dataset](train=False, augm_flag=False)
ood_loader = dl.ImageNet32(train=True, dataset=args.dataset, batch_size=batch_size)

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(test_loader.dataset))

num_data = len(train_loader.dataset)
num_classes = 100 if args.dataset == 'CIFAR100' else 10

if OOD_TRAINING:
    num_data *= 2  # OOD dataset doubles the data count
    num_classes += 1  # Additional 'none' class

if args.var0 is None:
    args.var0 = 1/(5e-4*len(train_loader.dataset))

    if args.method != 'plain':
        prior_prec *= 2  # Effective dataset size is doubled due to OOD data


model = bbb.WideResNetBBB(16, 4, num_classes, var0=args.var0, estimator=args.estimator)
model.cuda()
model.train()
print(f'Num. params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

n_epochs = 100
pbar = trange(n_epochs)

opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0, nesterov=True)
## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs*len(train_loader))

## For automatic-mixed-precision
scaler = amp.GradScaler()


for epoch in pbar:
    train_loss = 0

    if OOD_TRAINING:
        # Induce a randomness in the OOD batch since num_ood_data >> num_indist_data
        # The shuffling of ood_loader only happens when all batches are already yielded
        ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

    for batch_idx, ((x_in, y_in), (x_out, _)) in enumerate(zip(train_loader, ood_loader)):
        model.train()
        opt.zero_grad()
        
        m = len(x_in)  # Current batch size
        x_in, y_in = x_in.cuda(non_blocking=True), y_in.long().cuda(non_blocking=True)

        if OOD_TRAINING:
            x_out = x_out[:m].cuda(non_blocking=True)  # To make the length the same as x_in
            x = torch.cat([x_in, x_out], dim=0)

            # Noneclass: set x_out's target to be the last, additional, class
            # Last class, zero-indexed
            y_out = (num_classes-1) * torch.ones(m, device='cuda').long()
            y = torch.cat([y_in, y_out])
        else:
            x = x_in; y = y_in

        with amp.autocast():
            out, kl = model(x)
            # Scaled negative-ELBO with 1 MC sample
            # See Graves 2011 as to why the KL-term is scaled that way
            # and notice that we use mean instead of sum; tau is the tempering parameter
            loss = F.cross_entropy(out.squeeze(), y) + args.tau/num_data*kl

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        train_loss = 0.9*train_loss + 0.1*loss.item()

    model.eval()
    pred = evalutil.predict(test_loader, model, n_samples=1).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; ELBO: {train_loss:.3f}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )


if not os.path.exists(path):
    os.makedirs(path)

print(f'{path}/{args.dataset}{model_suffix}_bbb_{args.estimator}.pt')
torch.save(model.state_dict(), f'{path}/{args.dataset}{model_suffix}_bbb_{args.estimator}.pt')

## Try loading and testing
model.load_state_dict(torch.load(f'{path}/{args.dataset}{model_suffix}_bbb_{args.estimator}.pt'))
model.eval()

print()

## In-distribution
py_in = evalutil.predict(test_loader, model, n_samples=10).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')
