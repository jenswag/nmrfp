# ======================================================================================================================
# nmrfp: equivariant.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   Definition of invariant and equivariant layers.
#
# Author: Jens Wagner
# Revised: 30.10.2025
#
# Ref.:
# [1] Zaheer et al.: NeurIPS (2017). https://doi.org/10.48550/arXiv.1703.06114
# [2] Soelch et al.: ICANN (2019). https://doi.org/10.1007/978-3-030-30487-4_35
# [3] https://github.com/dpernes/deepsets-digitsum/blob/master/deepsetlayers.py
# ======================================================================================================================

# Import Packages
import math
import torch
from torch import nn
from torch.nn import init


# Invariant layer
class InvLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, reduction='sum'):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)

        self.reduction = reduction

        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))

        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        init.xavier_uniform_(self.beta)

        if self.bias is not None:

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X, mask=None):

        N, M, _ = X.shape
        device = X.device

        if mask is None:
            mask = torch.ones(N, M).byte().to(device)

        if self.reduction == 'mean':
            sizes = mask.float().sum(dim=1).unsqueeze(1)
            Z = X * mask.unsqueeze(2).float()
            y = (Z.sum(dim=1) @ self.beta)/sizes

        elif self.reduction == 'sum':
            Z = X * mask.unsqueeze(2).float()
            y = Z.sum(dim=1) @ self.beta

        elif self.reduction == 'max':
            Z = X.clone()
            Z[~mask] = float('-Inf')
            y = Z.max(dim=1)[0] @ self.beta

        else:  # min
            Z = X.clone()
            Z[~mask] = float('Inf')
            y = Z.min(dim=1)[0] @ self.beta

        if self.bias is not None:
            y += self.bias

        return y

    def extra_repr(self):

        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)


# Equivariant layer
class EquivLinear(InvLinear):

    def __init__(self, in_features, out_features, bias=True, reduction='sum'):

        super(EquivLinear, self).__init__(in_features, out_features,
                                          bias=bias, reduction=reduction)

        self.alpha = nn.Parameter(torch.Tensor(self.in_features,
                                               self.out_features))

        self.reset_parameters()

    def reset_parameters(self):

        super(EquivLinear, self).reset_parameters()
        if hasattr(self, 'alpha'):
            init.xavier_uniform_(self.alpha)

    def forward(self, X, mask=None):

        device = X.device
        N, M, _ = X.shape

        if mask is None:
            mask = torch.ones(N, M).byte().type(torch.bool).to(device)

        Y = torch.zeros(N, M, self.out_features).to(device)
        h_inv = super(EquivLinear, self).forward(X, mask=mask)
        Y[mask] = (X @ self.alpha + h_inv.unsqueeze(1))[mask]

        return Y
