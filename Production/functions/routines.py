# ======================================================================================================================
# nmrfp: routines.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   Functions to train and validate DSM.
#
# Author: Jens Wagner
# Revised: 30.10.2025
#
# ======================================================================================================================

# Import Packages
import torch
from sklearn.metrics import f1_score as F1


# Define Training
def training(data, model, lossF, optimizer, out, device, token):

    model.train()

    if out:
        print('Training:')

    tloss = 0   # Total Loss
    tlab, plab = [], []  # True and Predicted labels

    for batch, (X, Y, labile, fmask, smask) in enumerate(data):

        X, Y, labile, fmask, smask = X.to(device), Y.to(device), labile.to(device), fmask.to(device), smask.to(device)

        # Compute Prediction error
        pred = model(X, labile, fmask, smask)     # Prediction
        loss = lossF(pred[smask], Y[smask])   # Loss

        tloss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()    # Update Parameters
        optimizer.zero_grad()   # Reset Gradients

        _, true = torch.max(Y[smask], dim=-1)
        tlab += true.tolist()
        _, lab = torch.max(pred[smask], dim=-1)
        plab += lab.tolist()

    mloss = tloss / len(data)  # Mean Loss
    f1macro = F1(tlab, plab, average='macro')  # F1 macro
    f1micro = F1(tlab, plab, average='micro')  # F1 micro
    f1i = F1(tlab, plab, average=None)  # Individual F1

    if out:

        # Print Info
        print(f'Avg. loss: {mloss:>8f}')
        print(f'F1-score (Macro): {f1macro:>8f}')
        print(f'F1-score (Micro): {f1micro:>8f}')
        print('F1-score (Indiv.):', f1i)

    mloss = tloss / len(data)  # Mean Loss

    return mloss, f1macro, f1micro, f1i


# Define Validating
def validating(data, model, lossF, out, token):

    model.eval()
    [X, Y, labile, fmask, smask] = data

    # Compute Prediction error
    pred = model(X, labile, fmask, smask)  # Prediction
    loss = lossF(pred[smask], Y[smask])  # Loss

    _, tlab = torch.max(Y[smask], dim=-1)
    tlab = tlab.tolist()
    _, plab = torch.max(pred[smask], dim=-1)
    plab = plab.tolist()

    mloss = loss.item() / len(data)   # Mean Loss
    f1macro = F1(tlab, plab, average='macro')   # F1 macro
    f1micro = F1(tlab, plab, average='micro')   # F1 micro
    f1i = F1(tlab, plab, average=None)  # Individual F1

    if out:

        # Print Info
        print(f'\nValidation: '
              f'\nAvg. loss: {mloss:>8f}')
        print(f'F1-score (Macro): {f1macro:>8f}')
        print(f'F1-score (Micro): {f1micro:>8f}')
        print('F1-score (Indiv.):', f1i)
        print()

    return mloss, f1macro, f1micro, f1i
