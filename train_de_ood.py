import os
import torch
import torch.nn.functional as F
from torch import optim
from models.wideresnet import WideResNet
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
parser.add_argument('--method', choices=['noneclass', 'plain'], default='plain')
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

batch_size = 128
n_epochs = 100
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

models = [WideResNet(16, 4, num_classes) for _ in range(5)]
print(f'Num. params: 5*({sum(p.numel() for p in models[0].parameters() if p.requires_grad):,})')

for model in models:
    model.cuda()
    model.train()

n_epochs = 100
pbar = trange(n_epochs)

opts = [optim.SGD(models[k].parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True) for k in range(5)]
## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheds = [optim.lr_scheduler.CosineAnnealingLR(opts[k], T_max=n_epochs*len(train_loader)) for k in range(5)]
## For automatic-mixed-precision
scaler = amp.GradScaler()


for epoch in pbar:
    train_loss = 0

    if OOD_TRAINING:
        # Induce a randomness in the OOD batch since num_ood_data >> num_indist_data
        # The shuffling of ood_loader only happens when all batches are already yielded
        ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

    for batch_idx, ((x_in, y_in), (x_out, _)) in enumerate(zip(train_loader, ood_loader)):
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

        for model, opt, sched in zip(models, opts, scheds):
            model.train()
            opt.zero_grad()

            with amp.autocast():
                out = model(x)
                loss = F.cross_entropy(out.squeeze(), y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            train_loss = 0.9*train_loss + 0.1*loss.item()

    models[0].eval()
    pred = evalutil.predict(test_loader, models[0], n_samples=1).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets)*100
    mmc_val = pred.max(-1).mean()*100

    pbar.set_description(
        f'[Epoch: {epoch+1}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )


if not os.path.exists(path):
    os.makedirs(path)

save_name = f'{path}/{args.dataset}{model_suffix}_de.pt'
print(save_name)
torch.save([model.state_dict() for model in models], save_name)

## Try loading and testing
state_dicts = torch.load(save_name)
models = []

for state_dict in state_dicts:
    _model = WideResNet(16, 4, num_classes)
    _model.load_state_dict(state_dict)
    models.append(_model.cuda().eval())

print()

## In-distribution
py_in = evalutil.predict_ensemble(test_loader, models).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
print(f'Accuracy: {acc_in:.1f}')
