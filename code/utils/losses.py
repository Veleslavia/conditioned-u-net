import torch
import torch.nn as nn

from itertools import product


class MSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error Loss
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSLELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):

        ret = (torch.log(input + 1) - torch.log(target + 1)) ** 2
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret


class MPELoss(nn.Module):
    """
    Mean Power Error Loss
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean', power=2):
        super(MPELoss, self).__init__()
        self.reduction = reduction
        self.power = power

    def forward(self, input, target):

        ret = (input - target) ** self.power
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret


EPS = 1e-8


def remove_dc(signal):
    """Normalized to zero mean"""
    mean = torch.mean(signal, dim=-1, keepdim=True)
    signal = signal - mean
    return signal


def pow_p_norm(signal):
    """Compute 2 Norm"""
    return torch.pow(torch.norm(signal, p=2, dim=-1, keepdim=True), 2)


def pow_norm(s1, s2):
    return torch.sum(s1 * s2, dim=-1, keepdim=True)


class SISDRLoss(nn.Module):
    """SI-SDR Error Loss
    # Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(SISDRLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):

        input_nz, target_nz = input[target != 0], target[target != 0]
        target_scaled = pow_norm(input_nz, target_nz) * target_nz / (pow_p_norm(target_nz) + EPS)
        noise = input_nz - target_scaled
        sdr = 10 * torch.log10(pow_p_norm(target_scaled) / (pow_p_norm(noise) + EPS) + EPS)

        if self.reduction != 'none':
            sdr = torch.mean(sdr) if self.reduction == 'mean' else torch.sum(sdr)
        return -sdr


class MSESDRLoss(nn.Module):
    """
        Combined MSE (all) and SDR (non-zero values only) loss
    """

    def __init__(self, size_average=None, reduce=None, reduction='mean', power=2):
        super(MSESDRLoss, self).__init__()
        self.reduction = reduction
        self.power = power

    def forward(self, input, target):

        mse = (input - target) ** self.power

        input_nz, target_nz = input[target != 0], target[target != 0]
        target_scaled = pow_norm(input_nz, target_nz) * target_nz / (pow_p_norm(target_nz) + EPS)
        noise = input_nz - target_scaled
        sdr = 10 * torch.log10(pow_p_norm(target_scaled) / (pow_p_norm(noise) + EPS) + EPS)

        if self.reduction != 'none':
            sdr = torch.mean(sdr) if self.reduction == 'mean' else torch.sum(sdr)
            mse = torch.mean(mse) if self.reduction == 'mean' else torch.sum(mse)

        return mse - sdr

