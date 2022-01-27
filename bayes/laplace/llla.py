import torch
from torch import nn, autograd
from torch.nn import functional as F
from math import *
from tqdm import tqdm, trange
import numpy as np
from util import misc


def get_hessian(model, train_loader, ood_loader, num_classes, type='plain', prec=1):
    W = list(model.parameters())[-2]
    b = list(model.parameters())[-1]
    m, n = W.shape

    lossfunc = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

    W = list(model.modules())[-1].weight  # last-layer weight
    b = list(model.modules())[-1].bias    # last-layer bias
    H_W, H_b = 0, 0

    for batch in tqdm(train_loader):
        model.zero_grad()
        logits = model(batch.text.t().cuda())

        if type == 'dirlik':
            alpha = prec*F.log_softmax(logits).exp()
            dist_in = torch.distributions.Dirichlet(alpha)
            loss = -dist_in.log_prob(dist_in.sample()).mean()
        else:
            if num_classes == 2:
                y = torch.distributions.Bernoulli(logits=logits).sample()
            else:
                y = torch.distributions.Categorical(logits=logits).sample()
            loss = lossfunc(logits, y)

        if ood_loader is not None:
            batch_out = next(ood_loader)
            logits_out = model(batch_out.text.t().cuda())

            if type in ['noneclass', 'oe']:
                if num_classes == 2:
                    y_out = torch.distributions.Bernoulli(logits=logits_out).sample()
                else:
                    y_out = torch.distributions.Categorical(logits=logits_out).sample()
                loss += lossfunc(logits_out, y_out)
            elif type in ['dirlik', 'mixed']:
                alpha = prec*F.log_softmax(logits_out).exp()
                dist_out = torch.distributions.Dirichlet(alpha)
                loss += -dist_out.log_prob(dist_out.sample()).mean()

        W_grad, b_grad = autograd.grad(loss, [W, b])

        H_W += W_grad**2
        H_b += b_grad**2

    mean_W = W.t()
    mean_b = b
    H_W = H_W.t()

    return [mean_W, mean_b, H_W, H_b]


torch.no_grad()
def estimate_variance(var0, params):
    tau = 1/var0
    mean_W, mean_b, H_W, H_b = params

    # Add prior and invert
    var_W = 1 / (H_W + tau)
    var_b = 1 / (H_b + tau)

    return [mean_W, mean_b, var_W, var_b]


def gridsearch_var0(model, hessians, data_loader, interval, n_classes):
    vals, var0s = [], []
    pbar = tqdm(interval)

    for var0 in pbar:
        params = estimate_variance(var0, hessians)

        preds, y = [], []
        for batch in iter(data_loader):
            preds.append(predict_batch(batch.text.t().cuda(), model, *params, n_samples=5))
            y.append(batch.label - 1)

        preds = torch.cat(preds, 0).cpu().numpy()
        y = torch.cat(y, 0).cpu().numpy()

        # Brier score loss
        y_onehot = misc.get_one_hot(y, n_classes)
        loss = np.mean(np.linalg.norm(preds - y_onehot, ord=2, axis=1)**2)

        vals.append(loss)
        var0s.append(var0)

        pbar.set_description(f'var0: {var0:.2e}, Loss: {loss:.3f}')

    best_var0 = var0s[np.argmin(np.nan_to_num(vals, nan=9999))]

    return best_var0


@torch.no_grad()
def predict_batch(x, model, mean_W, mean_b, var_W, var_b, n_samples=100):
    phi = model.features(x)

    # MC-integral
    py = 0
    for _ in range(n_samples):
        W_sample = mean_W + torch.sqrt(var_W) * torch.randn(*mean_W.shape, device='cuda')
        b_sample = mean_b + torch.sqrt(var_b) * torch.randn(*mean_b.shape, device='cuda')
        py += torch.softmax(phi @ W_sample + b_sample, 1)
    py /= n_samples

    return py
