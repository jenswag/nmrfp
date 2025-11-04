# ======================================================================================================================
# nmrfp: prepare.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   Prepare augmented NMR spectra data for DSM model training and validation.
#
# Author: Jens Wagner
# Revised: 30.10.2025
#
# Input:
#   - ./data/data.json: JSON file containing augmented NMR data
#
# Parameters:
#   - seed (int): random seed for reproducibility (default = 42)
#   - partition (float): training data fraction (default = 0.9)
#   - degree (int): degree of synthetic mixtures (default = 3)
#
# Output:
#   - ./sets/sets_<degree>_<seed>.json: prepared sets for DSM training
#
# ======================================================================================================================

# Import Packages and Modules
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

from Production.functions.reproducibility import seeds

# ----------------------------------------------------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------------------------------------------------
seed = 42  # Random seed
partition = 0.9  # Train/validation split (validation = 1 - partition)
degree = 3  # Degree of synthetic mixtures

# ----------------------------------------------------------------------------------------------------------------------
# INITIALIZATION
# ----------------------------------------------------------------------------------------------------------------------
com = '{}_{}_{}'.format(str(partition), str(degree), str(seed))  # Identifier for output file

token1 = np.nan  # Placeholder
token2 = -42  # Replacement NaN

# ----------------------------------------------------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------------------------------------------------
with open(Path(__file__).resolve().parent / 'data/data.json') as f:
    data = json.load(f)

data = pd.DataFrame(data)  # Convert JSON into a DataFrame

# Initialize storage lists
sets = []  # Feature sets
fmasks = []  # Feature masks
targets = []  # Target groups
labile = []  # Labile H indicator

# ----------------------------------------------------------------------------------------------------------------------
# BUILD FEATURE SETS
# ----------------------------------------------------------------------------------------------------------------------
for i, row in data.iterrows():

    # Label presence of labile H
    labile.append(1 if row.LabileH else 0)

    set = []  # Feature set for one molecule
    fmask = []  # Mask for valid features
    target = row.Groups.copy()
    corr = 0  # Correction index for removed peaks

    # Construct feature vectors per 13C peak
    for idxC, peakC in enumerate(row.ShiftC):

        # Retrieve 1H shifts connected to the 13C peak
        peaksH = np.array(row.ShiftH)[np.nonzero(row.Connections[idxC])]

        # Skip invalid connections
        if None in peaksH:
            target.pop(idxC - corr)
            corr += 1
            continue

        # Feature vector: [Shift13C, DEPT, Shift1H (+ placeholder)]
        vec = [
            peakC,
            row.DEPT[idxC],
            *[x for x in peaksH],
            *[token1 for _ in range(3 - len(peaksH))]  # Pad missing 1H values
        ]
        set.append(vec)

        # Feature mask: 1 for NaN (masked), 0 for valid values
        vec = np.array(vec)
        mvec = vec.copy()
        mvec[np.isnan(vec)] = 1
        mvec[~np.isnan(vec)] = 0
        fmask.append(mvec.astype(int).tolist())

    sets.append(set)
    fmasks.append(fmask)
    targets.append(target)

# ----------------------------------------------------------------------------------------------------------------------
# PAD SETS TO UNIFORM LENGTH
# ----------------------------------------------------------------------------------------------------------------------
smasks = []  # Set masks

maxlen = len(max(sets, key=len))  # Length of the largest feature set

for idx in range(len(sets)):

    # Set mask: 1 for valid entries, 0 for padding
    smasks.append(np.hstack((
        np.ones(len(sets[idx])),
        np.zeros(maxlen - len(sets[idx]))
    )).astype(int).tolist())

    # Pad sets, targets, and masks to equal length
    while len(sets[idx]) < maxlen:
        sets[idx].append([token1] * len(sets[0][0]))
        targets[idx].append([token1] * len(targets[0][0]))
        fmasks[idx].append([0] * len(sets[0][0]))

# ----------------------------------------------------------------------------------------------------------------------
# SPLIT INTO TRAINING AND VALIDATION
# ----------------------------------------------------------------------------------------------------------------------
sets = np.array(sets)
fmasks = np.array(fmasks)
smasks = np.array(smasks)
targets = np.array(targets)
labile = np.array(labile)

# Create validation mask
vmask = np.zeros(len(sets)).astype(bool)
rand = np.random.choice(len(sets), round(len(sets) * (1 - partition)), replace=False)
vmask[rand] = True

# Validation subsets
vsets = sets[vmask]
vfmasks = fmasks[vmask]
vsmasks = smasks[vmask]
vtargets = targets[vmask]
vlabile = labile[vmask]

# Training subsets
tsets = sets[~vmask]
tfmasks = fmasks[~vmask]
tsmasks = smasks[~vmask]
ttargets = targets[~vmask]
tlabile = labile[~vmask]

# ----------------------------------------------------------------------------------------------------------------------
# BUILD SYNTHETIC MIXTURES
# ----------------------------------------------------------------------------------------------------------------------
for deg in range(2, degree + 1):

    # Prepare mixture masks
    seeds(seed + deg)
    rand = np.random.choice(len(tsets), len(tsets), replace=False)
    mmask = np.zeros(len(tsets)).astype(bool)

    while len(rand) % deg != 0:
        rand = rand[:-1]

    mmasks = [mmask.copy() for _ in range(deg)]
    rands = np.array_split(rand, deg)

    for mmask, rand in zip(mmasks, rands):
        mmask[rand] = True

    # Stack mixtures for training
    msets = np.hstack([tsets[mmasks[idx]] for idx in range(deg)])
    mfmasks = np.hstack([tfmasks[mmasks[idx]] for idx in range(deg)])
    msmasks = np.hstack([tsmasks[mmasks[idx]] for idx in range(deg)])
    mtargets = np.hstack([ttargets[mmasks[idx]] for idx in range(deg)])
    mlabile = np.sum(np.stack([tlabile[mmasks[idx]] for idx in range(deg)]), axis=0)
    mlabile[mlabile > 1] = 1  # Cap at 1 if multiple labile Hs

    # Append placeholder arrays to match dimensions
    tsets = np.hstack((tsets, *[np.ones(tsets.shape) * token1 for _ in range(deg - 1)]))
    tfmasks = np.hstack((tfmasks, *[np.ones(tfmasks.shape) for _ in range(deg - 1)]))
    tsmasks = np.hstack((tsmasks, *[np.zeros(tsmasks.shape) for _ in range(deg - 1)]))
    ttargets = np.hstack((ttargets, *[np.ones(ttargets.shape) * token1 for _ in range(deg - 1)]))

    # Combine original and synthetic mixtures
    tsets = np.concatenate((tsets, msets))
    tfmasks = np.concatenate((tfmasks, mfmasks))
    tsmasks = np.concatenate((tsmasks, msmasks))
    ttargets = np.concatenate((ttargets, mtargets))
    tlabile = np.concatenate((tlabile, mlabile))

# ----------------------------------------------------------------------------------------------------------------------
# FINALIZE AND SAVE
# ----------------------------------------------------------------------------------------------------------------------
# Replace NaN placeholders before saving
sets[np.isnan(sets)] = token2
tsets[np.isnan(tsets)] = token2
vsets[np.isnan(vsets)] = token2

train = {
    'sets': tsets.tolist(),
    'fmasks': tfmasks.tolist(),
    'smasks': tsmasks.tolist(),
    'targets': ttargets.tolist(),
    'labile': tlabile.tolist()
}

valid = {
    'sets': vsets.tolist(),
    'fmasks': vfmasks.tolist(),
    'smasks': vsmasks.tolist(),
    'targets': vtargets.tolist(),
    'labile': vlabile.tolist()
}

alls = {
    'sets': sets.tolist(),
    'fmasks': fmasks.tolist(),
    'smasks': smasks.tolist(),
    'targets': targets.tolist(),
    'labile': labile.tolist()
}

out = {
    'train': train,
    'valid': valid,
    'alls': alls,
    'seed': seed,
    'degree': degree,
    'partition': partition,
    'token1': token1,
    'token2': token2
}

with open(Path(__file__).resolve().parent / 'sets/sets_{}.json'.format(com), 'w') as f:
    json.dump(out, f)
    f.close()
