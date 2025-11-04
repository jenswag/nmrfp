# ======================================================================================================================
# nmrfp: produce.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   Train DSM on prepared NMR data sets.
#
# Author: Jens Wagner
# Revised: 30.10.2025
#
# Input:
#   - ./sets/sets_{partition}_{degree}_{seed}.json: prepared sets for DSM training
#
# Parameters:
#   - file (str): data set file identifier (default = 'sets_0.9_3_42')
#   - layersX (int): nucleus networks depth (Phi_C, Phi_H; default = 2)
#   - nodesX (int): hidden nodes per nucleus network (Phi_C, Phi_H; default = 8)
#   - layers (list[int]): number of layers for [Phi, Sigma, Rho] (default = [1, 0, 1])
#   - nodes (int): hidden nodes for Phi, Sigma, and Rho (default = 256)
#   - epochs (int): maximum training epochs (default = 5000)
#   - bs (int): batch size (default = 1)
#   - lr (float): learning rate (default = 1e-4)
#   - decay (float): weight decay for optimizer (default = 5e-4)
#   - patience (int): epochs without improvement before LR reduction (default = 20)
#   - early (int): epochs without improvement before early stop (default = 30)
#   - com (str): run comment appended to filenames (default = 'final')
#
# Output:
#   - ./save/<model>_<timestamp>_<com>/
#       ├── <model>.pt               # Trained model weights (best epoch)
#       └── log.txt                  # Training summary and configuration
#
# ======================================================================================================================

# Import Packages and Modules
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
import os
import shutil
from datetime import datetime
from pathlib import Path

from Production.functions.reproducibility import seeds
from Production.functions.routines import training, validating

from Model.dsm import DSM

# ----------------------------------------------------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
# Data
file = 'sets_0.9_3_42'

# Model
layersX = 2  # Network depth Phi_C, Phi_H: layers = 1 + layersX
nodesX = 8  # Width for Phi_C, Phi_H
layers = [1, 0, 1]  # Phi, Sigma, Rho: layers = 1 + layers[i]
nodes = 256  # Width for Phi, Sigma, Rho

# Training
epochs = 5000  # Max epochs
bs = 1  # Batch size
lr = 1e-4  # Learning rate
decay = 5e-4  # Weight decay
patience = 20  # Plateau patience for LR scheduler (factor 0.1)
early = 30  # Early stop patience after no validation score improvement

com = 'final'  # Comment for result files

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZATION
# ----------------------------------------------------------------------------------------------------------------------
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'  # Compute device
cuda = torch.cuda.is_available()

# Load data bundle
data = json.load(open(Path(__file__).resolve().parent / 'sets/{}.json'.format(file), 'r'))

seed = data['seed']  # Reproducibility seed
partition = data['partition']  # Train/validation split fraction
degree = data['degree']  # Mixture degree used in data generation
token = [data['token1'], data['token2']]  # Token values

train = data['train']
valid = data['valid']
alls = data['alls']

F = len(train['sets'][0][0])  # Feature size
C = len(train['targets'][0][0])  # Number of classes

# Fix randomness and wrap datasets
seeds(seed)
train = DataLoader(TensorDataset(torch.Tensor(train['sets']).to(device),
                                 torch.Tensor(train['targets']).to(device),
                                 torch.Tensor(train['labile']).to(device),
                                 torch.Tensor(train['fmasks']).type(torch.bool).to(device),
                                 torch.Tensor(train['smasks']).type(torch.bool).to(device)),
                   batch_size=bs, shuffle=True)  # Training loader

valid = [torch.Tensor(valid['sets']).to(device),
         torch.Tensor(valid['targets']).to(device),
         torch.Tensor(valid['labile']).to(device),
         torch.Tensor(valid['fmasks']).type(torch.bool).to(device),
         torch.Tensor(valid['smasks']).type(torch.bool).to(device)]

# ----------------------------------------------------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------------------------------------------------
model = DSM(F, C, phiX=layersX, nodesX=nodesX, phi=layers[0], sig=layers[1], rho=layers[2],
            nodes=nodes).to(device)  # Initialize DSM

lossF = nn.CrossEntropyLoss()  # Loss fucntion
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)  # Optimizer
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1, verbose=True)  # Scheduler

# ----------------------------------------------------------------------------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------------------------------------------------------------------------
# Start marker
st = datetime.now()
print('Start: {}\n'.format(st.strftime('%d/%m/%Y %H:%M:%S')))

# Data info
print('Data\n'
      '----------------\n'
      'File: {}\n'
      'Seed: {}\n'
      'Partition: {}\n'
      'Mixture degree: {}\n'
      'Token: {}\n'.format(file, seed, partition, degree,
                           ', '.join(['{:.3f}'.format(num) for num in token])))

# Model info
print('Model\n'
      '----------------\n'
      'Layers invariant DSM: {}\t\t# 1 + n\n'
      'Nodes invariant DSM: {}\n'
      'Layers equivariant DSM: {}\t\t# 1 + n\n'
      'Nodes equivariant DSM: {}\n'.format(layersX, nodesX,
                                           ', '.join(['{:d}'.format(num) for num in layers]), nodes))

# Training info
print('Training\n'
      '----------------\n'
      'Cuda: {}\n'
      'Epochs: {}\n'
      'Batch size: {}\n'
      'Learning rate: {}\n'
      'Weight decay: {}\n'
      'Patience: {}\n'
      'Cancel early: {}\n'.format(cuda, epochs, bs, lr, decay, patience, early))

# Storage for scores
[Tloss, tF1macro, tF1micro, tF1i, Vloss, vF1macro, vF1micro, vF1i] = [[] for _ in range(8)]  # Figure data

# Epoch loop
vmin = 1e10  # Best validation initialization
fname = 'DSM_' + st.strftime('%Y%m%d_%H-%M') + '_' + com + '.pt'  # Checkpoint name
for e in (range(epochs)):

    print('Epoch {} / {}'.format(e + 1, epochs))

    # Train and validate
    tloss, tf1macro, tf1micro, tf1i = training(train, model, lossF, optimizer, False, device, token)
    vloss, vf1macro, vf1micro, vf1i = validating(valid, model, lossF, False, token)

    # Log metrics
    Tloss.append(tloss)
    tF1macro.append(tf1macro)
    tF1micro.append(tf1micro)
    tF1i.append(tf1i.tolist())

    Vloss.append(vloss)
    vF1macro.append(vf1macro)
    vF1micro.append(vf1micro)
    vF1i.append(vf1i.tolist())

    # Early stopping and checkpoint
    if vloss < vmin:

        vmin = vloss
        imin = e
        torch.save(model.state_dict(), Path(__file__).resolve().parent / './save/' / fname)

        ecount = 0

    else:

        ecount += 1
        if ecount >= early:
            print('Training stopped at epoch {}. Minimal loss at epoch {}.\n'.format(e + 1, imin))
            break

    # Scheduler update
    scheduler.step(vloss)

# Print training snapshot at best epoch
print('Training\n'
      '----------------\n'
      'Avg. Loss: {}\n'
      'Avg. F1-score (Macro): {}\n'
      'Avg. F1-score (Micro): {}\n'
      'Avg. F1-score (Indiv.): '.format(Tloss[imin], tF1macro[imin], tF1micro[imin]), tF1i[imin])

# Print validation snapshot at best epoch
print('\nValidation\n'
      '----------------\n'
      'Avg. Loss: {}\n'
      'Avg. F1-score (Macro): {}\n'
      'Avg. F1-score (Micro): {}\n'
      'Avg. F1-score (Indiv.): '.format(Vloss[imin], vF1macro[imin], vF1micro[imin]), vF1i[imin])

rt = str(datetime.now() - st)  # Wall time
print('\nRun time: {}'.format(rt))

# ----------------------------------------------------------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------------------------------------------------------

# Create results directory and move best model
path = (Path(__file__).resolve().parent / 'save' / 
        f"DSM_{st.strftime('%Y-%m-%d_%H-%M')}_Loss_{np.round(Vloss[imin], 2)}{f'_{com}' if com else ''}")
if os.path.isdir(path):
    shutil.rmtree(path)
os.makedirs(path)
os.rename(Path(__file__).resolve().parent / 'save' / fname, path / fname)  # Move checkpoint

# Log file with run configuration and metrics
with open(path / 'log.txt', 'w') as f:
    f.write('{}\n\n'
            'Data\n'
            '----------------\n'
            'File: {}\n'
            'Seed: {}\n'
            'Partition: {}\n'
            'Mixture degree: {}\n'
            'Token: {}\n\n'
            'Model\n'
            '----------------\n'
            'Layers invariant DSM: {}\t\t# 1 + n\n'
            'Nodes invariant DSM: {}\n'
            'Layers equivariant DSM: {}\t\t# 1 + n\n'
            'Nodes equivariant DSM: {}\n\n'
            'Training\n'
            '----------------\n'
            'Cuda: {}\n'
            'Epochs: {}\n'
            'Batch size: {}\n'
            'Learning rate: {}\n'
            'Weight decay: {}\n'
            'Patience: {}\n'
            'Cancel early: {}\n\n'
            'Avg. Loss: {}\n'
            'F1-score (Macro): {}\n'
            'F1-score (Micro): {}\n'
            'F1-score (Indiv.): {}\n\n'
            'Validation\n'
            '----------------\n'
            'Avg. Loss: {}\n'
            'F1-score (Macro): {}\n'
            'F1-score (Micro): {}\n'
            'F1-score (Indiv.): {}\n\n'
            'Run time: {}\n'.format(st.strftime('%d/%m/%Y %H:%M:%S'), file, seed, partition, degree,
                                    token, layersX, nodesX,
                                    ', '.join(['{:d}'.format(num) for num in layers]), nodes, cuda, epochs, bs, lr,
                                    decay, patience, early, Tloss[imin], tF1macro[imin], tF1micro[imin],
                                    ', '.join(['{:.2f}'.format(num) for num in tF1i[imin]]), Vloss[imin],
                                    vF1macro[imin], vF1micro[imin],
                                    ', '.join(['{:.2f}'.format(num) for num in vF1i[imin]]), rt))
    f.close()
