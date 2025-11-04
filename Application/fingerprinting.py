# ======================================================================================================================
# nmrfp: fingerprinting.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   Application of DSM for NMR fingerprinting of mixtures.
#
# Author: Jens Wagner
# Revised: 30.10.2025
#
# Input:
#   - ./input/{name}.xlsx: Excel file containing spectral data of mixture
#
# Parameters:
#   - file (str): Excel file identifier (default = 'Mixture_I')
#   - corr (float): parameters [m, b] of linear regression for correlation of chemical shifts 1H and 13C
#                   (default = [m = 0.05171687110147821, b = 0.23901273390989086])
#
# Output:
#   - ./output/{name}.xlsx: Excel file containing NMR fingerprinting results for mixture
#
# ======================================================================================================================

# Import Packages and Modules
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as func
import os
from pathlib import Path

from Application.functions.constants import mapDEPT, names
from Model.dsm import DSM

# ----------------------------------------------------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
file = 'Mixture_I'    # Input file

corr = [0.05171687110147821,    # Parameters [m, b] of linear regression for correlation of chemical shifts 1H and 13C
        0.23901273390989086]

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZATION
# ----------------------------------------------------------------------------------------------------------------------

token = -42     # Token

# ----------------------------------------------------------------------------------------------------------------------
# READ INPUT
# ----------------------------------------------------------------------------------------------------------------------
path = Path(__file__).resolve().parent / 'input' / f'{file}.xlsx'
assert os.path.isfile(path), 'Input file not found.'

# print Info
print('NMR Fingerprinting of {}\n'
      '--------------------\n'.format(file))

df = pd.read_excel(path, header=None, index_col=0)  # Create dataframe

# Chemical shifts 1H
peaksH = [str(x).split() for x in df.loc['Chemical shifts 1H'][~df.loc['Chemical shifts 1H'].isnull()]]
for Hid, H in enumerate(peaksH):
    for hid, h in enumerate(H):
        peaksH[Hid][hid] = float(h.strip(','))

# Labile 1H
labile = df.loc['Labile 1H']
labile = labile[~pd.isnull(labile)].to_list()[0]

# HSQC connections
hsqc = df.loc['HSQC connections']
hsqc = hsqc[~pd.isnull(hsqc)]
hsqc = [int(c - 1) if not isinstance(c, str) else c for c in hsqc]

# Chemical shifts 13C
peaksC = df.loc['Chemical shifts 13C'].to_numpy().astype(float)
peaksC = peaksC[~np.isnan(peaksC)]

# Signal areas 13C
areasC = df.loc['Signal areas 13C'].to_numpy().astype(float)
areasC = areasC[~np.isnan(areasC)]

# Intensities DEPT 90
dept90 = df.loc['Intensities DEPT 90']
dept90 = dept90[~pd.isnull(dept90)].to_list()
dept90 = [str(int(d)) if not isinstance(d, str) else d for d in dept90]

# Intensities DEPT 135
dept135 = df.loc['Intensities DEPT 135']
dept135 = dept135[~pd.isnull(dept135)].to_list()
dept135 = [str(int(d)) if not isinstance(d, str) else d for d in dept135]

# Check Input
assert len(peaksH) == len(hsqc), "Check Input for 'Chemical shifts 1H' and 'HSQC connections'."
assert len(peaksC) == len(areasC) == len(dept90) == len(dept135), ("Check Input for 'Chemical shifts 13C',"
                                                                   " 'Signal areas 13C', 'Intensities DEPT 90' and "
                                                                   "'Intensities DEPT 135'.")
assert (labile.lower() == 'no' or labile.lower() == 'yes'), "'Labile 1H' can only be 'Yes' or 'No'."

# ----------------------------------------------------------------------------------------------------------------------
# PARSING
# ----------------------------------------------------------------------------------------------------------------------
# Determine DEPT, H_{x} and labile H
dept = np.array([mapDEPT[d90 + d135] for d90, d135 in zip(dept90, dept135)])   # DEPT

prot = np.array([3 - d for d in dept])   # H_{X}
labile = 0 if labile.lower() == 'no' else 1     # Labile H

# Determine mole fraction
molC = [a / np.sum(areasC) for a in areasC]

# Prepare data
X = np.ones((len(peaksC), 5)) * token

# Insert data
X[:, 0] = peaksC    # Chemical shift 13C
X[:, 1] = dept  # DEPT

# Chemical shifts 1H
for c, h in zip(hsqc, peaksH):
    if isinstance(c, int):

        if len(h) == 1:
            X[c, 2:2 + prot[c]] = h[0]

        elif len(h) > 1 and len(h) == prot[c]:
            X[c, 2:2 + prot[c]] = h

# Check input
for x in X:

    # Correct missing 1H by Correlation
    if len(x[2:][x[2:] != token]) != 3 - x[1]:
        x[2:2 + int(x[1] + 1)] = np.round(corr[0] * x[0] + corr[1], 2)

# Prepare masks
smask = np.ones(len(peaksC))
fmask = np.zeros(X.shape)
fmask[X == token] = 1

# ----------------------------------------------------------------------------------------------------------------------
# NMR FINGERPRINTING
# ----------------------------------------------------------------------------------------------------------------------
# Prepare input
X = torch.Tensor([X.tolist()])
labile = torch.Tensor([labile])
smask = torch.Tensor([smask.tolist()]).type(torch.bool)
fmask = torch.Tensor([fmask.tolist()]).type(torch.bool)

# Load model
base = Path(__file__).resolve().parent.parent / 'Model' / 'parameters'
path = base / os.listdir(base)[0]
params = torch.load(path, map_location=torch.device('cpu'))
model = DSM(5, 13, phiX=2, nodesX=8, phi=1, sig=0, rho=1, nodes=256)
model.load_state_dict(params)

# Prediction
model.eval()
Y = model(X, labile, fmask, smask)

Y = func.softmax(Y, -1)     # Softmax
_, pred = torch.max(Y, -1)  # Predictions
pred = pred[-1]
predH = [names[y].replace('x', str(p)) for y, p in zip(pred, prot)]     # Prediction with H_[x}

# Print predictions
print('Predictions\n'
      '----------\n'
      '{}\n'
      '{}\n\n'
      'Confidences\n'
      '----------'.format(', '.join([p for p in predH]), ', '.join(['{:.2f}'.format(x) for x in molC])))

for y in Y[-1].tolist():
    print([round(x, 2) for x in y])

# ----------------------------------------------------------------------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------------------------------------------------------------------
# Create output
out = {'Chemical shift 13C': peaksC.tolist(),
       'Predicted group': predH,
       'Confidence': [round(max(y),2) for y in Y[-1].tolist()],
       'Mole fraction / mol/mol': molC}

# Create dataframe
out = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in out.items()]))
out = out.transpose()

# Write to Excel
out.to_excel(Path(__file__).resolve().parent / 'output/{}.xlsx'.format(file), header=False)
