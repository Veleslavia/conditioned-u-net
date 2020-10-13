import torch


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


def si_sd_sdr(predicted, target):
    predicted_nz, target_nz = predicted, target
    target_scaled = pow_norm(predicted_nz, target_nz) * target_nz / (pow_p_norm(target_nz) + EPS)
    si_noise = target_scaled - predicted_nz
    sd_noise = target_nz - predicted_nz
    si_sdr = 10 * torch.log10(pow_p_norm(target_scaled) / (pow_p_norm(si_noise) + EPS) + EPS)
    sd_sdr = 10 * torch.log10(pow_p_norm(target_scaled) / (pow_p_norm(sd_noise) + EPS) + EPS)
    return si_sdr, sd_sdr

