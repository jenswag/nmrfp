# ======================================================================================================================
# nmrfp: dsm.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   Definition of DSM.
#
# Author: Jens Wagner
# Revised: 30.10.2025
#
# Ref.:
# [1] Zaheer et al.: NeurIPS (2017). https://doi.org/10.48550/arXiv.1703.06114
# [2] Soelch et al.: ICANN (2019). https://doi.org/10.1007/978-3-030-30487-4_35
# ======================================================================================================================

# Import Packages and Modules
import torch
from torch import nn
import torch.nn.functional as func

from Model.layers.equivariant import EquivLinear


# Define DSM
class DSM(nn.Module):

    def __init__(self, F, C, phiX=2, nodesX=8, phi=1, sig=0, rho=1, nodes=256):
        super(DSM, self).__init__()

        self.F = F  # Features
        self.C = C  # Classes
        self.nodes = nodes  # Nodes Phi, Sig and Rho
        self.nodesX = nodesX    # Nodes Phi_C, Phi_H

        # Encoder phi
        layers = [nn.SiLU(), nn.Linear(self.nodesX, self.nodesX)] * phiX
        self.phiC = nn.Sequential(nn.Linear(2, self.nodesX),
                                  *layers)

        layers = [nn.SiLU(), nn.Linear(self.nodesX, self.nodesX)] * phiX
        self.phiH = nn.Sequential(nn.Linear(1, self.nodesX),
                                  *layers)

        layers = [nn.Linear(self.nodes, self.nodes), nn.SiLU()] * phi
        self.phi = nn.Sequential(nn.Linear(self.nodesX, self.nodes),
                                 nn.SiLU(),
                                 *layers)

        # Equivariant network sigma
        self.sig = nn.ModuleList([EquivLinear(self.nodes, self.nodes) for _ in range(sig + 1)])

        # Decoder rho
        layers = [nn.SiLU(), nn.Linear(self.nodes, self.nodes)] * rho
        self.rho = nn.Sequential(*layers,
                                 nn.SiLU(),
                                 nn.Linear(self.nodes, self.C))

    def reset_parameters(self):

        for layer in self.phi:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.sig:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for layer in self.rho:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, X, labile, fmask, smask):

        # Encoding
        C = self.phiC(X[:, :, 0:2])
        H = self.phiH(X[:, :, 2:].unsqueeze(-1))

        mask = ~fmask.unsqueeze(-1).repeat(1, 1, 1, self.nodesX)[:, :, 2:]
        H = torch.mul(H, mask)

        Y = torch.add(C, H.sum(-2))     # Aggregation
        Y = self.phi(Y)

        # Equivariant network
        for i, layer in enumerate(self.sig):

            Y = layer(Y, smask)

            if not i == len(self.sig) - 1:
                Y = func.silu(Y)

        # Add labile H
        z = labile.unsqueeze(-1).unsqueeze(-1)
        z = z.repeat(1, Y.size(-2), Y.size(-1))
        Y = torch.add(Y, z)

        Y = self.rho(Y)     # Decoding

        return Y
