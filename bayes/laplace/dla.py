import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as np
from math import *
from util.evaluation import predict_laplace
from util import misc


class DiagLaplace(nn.Module):
    """
    Taken, with modification, from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py
    """

    def __init__(self, base_model):
        super().__init__()

        self.n_classes = len(list(base_model.modules())[-1].bias)
        self.net = base_model
        self.params = []
        self.net.apply(lambda module: dla_parameters(module, self.params))
        self.hessians = None
        self.n_params = sum(p.numel() for p in base_model.parameters())

    def forward(self, x):
        return self.net.forward(x)

    def forward_sample(self, x):
        self.sample()
        return self.net.forward(x)

    def sample(self, scale=1, require_grad=False):
        for module, name in self.params:
            mean = module.__getattr__(f'{name}_mean')
            var = module.__getattr__(f'{name}_var')

            eps = torch.randn(*mean.shape, device='cuda')
            w = mean + scale * torch.sqrt(var) * eps

            if require_grad:
                w.requires_grad_()

            module.__setattr__(name, w)
            getattr(module, name)
        else:
            for module, name in self.params:
                mean = module.__getattr__(f'{name}_mean')
                var = module.__getattr__(f'{name}_var')

    def estimate_variance(self, var0):
        tau = 1/var0

        for module, name in self.params:
            h = self.hessians[(module, name)].clone()
            var = (1 / (h + tau))
            module.__getattr__(f'{name}_var').copy_(var)

    def get_hessian(self, train_loader, ood_loader=None, mode='noneclass'):
        assert mode in ['plain', 'noneclass', 'dirlik', 'mixed', 'oe']
        criterion = nn.CrossEntropyLoss()
        diag_hess = dict()

        if ood_loader is not None:
            ood_loader = iter(ood_loader)

        for module, name in self.params:
            var = module.__getattr__(f'{name}_var')
            diag_hess[(module, name)] = torch.zeros_like(var)

        # Populate parameters with the means
        self.sample(scale=0, require_grad=True)

        for x, _ in tqdm(train_loader):
            x = x.cuda()
            m = len(x)

            if ood_loader is not None:
                x_out, _ = next(ood_loader)
                x_out = x_out[:len(x)].cuda()  # Make sure the same length as x
                x = torch.cat([x, x_out], 0)

            out = self(x).squeeze()

            if ood_loader is None:
                y_in = torch.distributions.Categorical(logits=out[:m]).sample()
                loss = criterion(out[:m], y_in)
            else:
                if mode in ['noneclass', 'oe']:
                    y_in = torch.distributions.Categorical(logits=out[:m]).sample()
                    y_out = torch.distributions.Categorical(logits=out[m:]).sample()
                    loss = criterion(out[:m], y_in)
                    loss += criterion(out[m:], y_out)
                elif mode == 'mixed':
                    y_in = torch.distributions.Categorical(logits=out[:m]).sample()
                    dist_out = torch.distributions.Dirichlet(1*torch.softmax(out[m:], -1))
                    loss = criterion(out[:m], y_in)
                    loss += -dist_out.log_prob(dist_out.sample()).mean()
                else:  # Dirichlet likelihood
                    prec = 100
                    probs = torch.softmax(out, 1)
                    dist_in = torch.distributions.Dirichlet(prec*probs[:m] + 1e-10)
                    dist_out = torch.distributions.Dirichlet(prec*probs[m:] + 1e-10)
                    loss = -dist_in.log_prob(dist_in.sample()).mean()
                    loss += -dist_out.log_prob(dist_out.sample()).mean()

            self.net.zero_grad()
            loss.backward()

            for module, name in self.params:
                grad = module.__getattr__(name).grad
                diag_hess[(module, name)] += grad**2

        self.hessians = diag_hess

        return diag_hess

    def gridsearch_var0(self, val_loader, interval, lam=1):
        vals, var0s = [], []
        pbar = tqdm(interval)

        for var0 in pbar:
            try:
                self.estimate_variance(var0)

                preds_in, y_in = predict_laplace(val_loader, self, n_samples=5, return_targets=True)

                # Brier score loss
                preds_in, y_in = preds_in.cpu().numpy(), y_in.cpu().numpy()
                y_in_oh = misc.get_one_hot(y_in, self.n_classes)
                loss = np.mean(np.linalg.norm(preds_in - y_in_oh, ord=2, axis=1)**2)
            except:
                loss = inf

            vals.append(loss)
            var0s.append(var0)

            # pbar.set_description(f'var0: {var0:.3f}, Loss-in: {loss_in:.3f}, Loss-out: {loss_out:.3f}, Loss: {loss:.3f}')
            pbar.set_description(f'var0: {var0:.2e}, Loss: {loss:.3f}')

        best_var0 = var0s[np.argmin(vals)]

        return best_var0


def dla_parameters(module, params):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            # print(module, name)
            continue

        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer(f'{name}_mean', data)
        module.register_buffer(f'{name}_var', data.new(data.size()).zero_())
        module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))


