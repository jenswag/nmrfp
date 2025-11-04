# ======================================================================================================================
# nmrfp: reproducibility.py
# ----------------------------------------------------------------------------------------------------------------------
# Purpose:
#   Function to ensure reproducibility.
#
# Author: Jens Wagner
# Revised: 30.10.2025
#
# ======================================================================================================================

# Import Packages
import random
import numpy as np
import torch


# Set Seed for Reproducibility
def seeds(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
