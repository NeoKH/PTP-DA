#!/usr/bin/env python3
# coding:utf8
"""
@file      generals.py
@author    Hao Kong
@date      2023-11-24
@github    https://github.com/NeoKH/PTP-DA
"""

import random
import torch
import numpy as np

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

