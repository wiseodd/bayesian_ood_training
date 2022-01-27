import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as d
import copy
from math import *


class LLVB(nn.Module):

    def __init__(self, base_model, prior_var=1):
        """
        base_model : Any NN with `features(x)` method and attribute `feature_extractor`
        prior_var  : Scalar variance for the last-layer weight W_L
        """
        super().__init__()

        self.base_model = copy.deepcopy(base_model)
        self.base_model.feature_extractor = True

        last_layer = list(self.base_model.modules())[-1]

        self.n_classes, self.n_features = last_layer.weight.shape

        # Diagonal Gaussian variational posterior for W_L
        self.m_W = nn.Parameter(last_layer.weight.data.clone().flatten())
        self.logvar_W = nn.Parameter(torch.zeros_like(self.m_W) - 6)

        # Diagonal Gaussian variational posterior for b_L
        self.m_b = nn.Parameter(last_layer.bias.data.clone())
        self.logvar_b = nn.Parameter(torch.zeros(self.n_classes) - 6)

        self.var_params = [self.m_W, self.logvar_W, self.m_b, self.logvar_b]

        self.prior_var = prior_var

    def forward(self, x, n_samples=1):
        """
        In training mode, also outputs KL-div
        """
        features = self.base_model(x)[None, :, :]  # 1 x mb_size x m

        # Variational posteriors
        var_W = F.softplus(self.logvar_W)
        var_b = F.softplus(self.logvar_b)

        # Sample weights
        W = sample_diag_gaussian(self.m_W, var_W, n_samples)
        W = W.reshape(n_samples, self.n_classes, self.n_features)  # n_samples x m x n

        b = sample_diag_gaussian(self.m_b, var_b, n_samples)
        b = b[:, None, :] # n_samples x 1 x n

        # Get output
        out = features @ W.transpose(1,2) + b  # n_samples x mb_size x n

        if not self.training:
            return out

        # KL-divergence
        dkl_W = kld_normal_diag_iso(
            self.m_W, var_W, torch.zeros_like(self.m_W), self.prior_var
        )
        dkl_b = kld_normal_diag_iso(
            self.m_b, var_b, torch.zeros_like(self.m_b), self.prior_var
        )

        dkl = dkl_W + dkl_b

        return out, dkl



def sample_diag_gaussian(mean, var, n_samples=1):
    e = torch.randn(n_samples, *mean.shape, device='cuda')
    return mean[None, :] + e * var.sqrt()[None, :]


def kld_normal_diag_iso(mean1, var1, mean2, var2):
    """
    KL-div b/w p1 := diagonal Gaussian and p2 := isotropic Gaussian
    """
    assert len(mean1.shape) == 1 and len(var1.shape) == 1
    assert np.isscalar(var2)

    n = len(mean1)
    dkl = n*log(var2) - torch.sum(torch.log(var1)) - n
    dkl += torch.sum(var1/var2) + torch.sum(mean1**2)/var2
    dkl *= 1/2

    return dkl
