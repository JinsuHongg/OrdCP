import torch
import numpy as np
import torch.nn as nn

def cross_entropy(output, target):
    logp_out = torch.log(output)
    inv_logp_out = torch.log(1 - output)
    return torch.sum(-(target * logp_out) - ((1-target)*inv_logp_out)) / len(output)
