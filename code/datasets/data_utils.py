import os

import numpy as np

import librosa
import torch
import torch.nn.functional as F
import norbert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _save_segments_to_songs(songs, output_dir, sr):
    for piece_name in songs.keys():
        piece_dir = os.path.join(output_dir, piece_name)
        if not os.path.exists(piece_dir):
            os.mkdir(piece_dir)

        for source_idx in range(len(songs[piece_name])):
            source_name = songs[piece_name][source_idx][0]
            unsorted_data = songs[piece_name][source_idx][1]
            source_wav = np.concatenate([data for idx, data in (sorted(unsorted_data, key=lambda x: x[0]))])
            librosa.output.write_wav(os.path.join(piece_dir, '{:02d}_'.format(source_idx) + source_name),
                                     source_wav, sr)


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


def warpgrid(N_MIX, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((N_MIX, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def preprocess_with_norbert(complex_stft_mix, predicted_magnitudes):
    # v: np.ndarray [shape=(nb_frames, nb_bins, {1,nb_channels}, nb_sources)]
    # x: np.ndarray [complex, shape=(nb_frames, nb_bins, nb_channels)]

    complex_stft_mix = complex_stft_mix.detach().data.cpu()
    complex_stft_mix_numpy = np.array(complex_stft_mix[:, :, :, :, 0]) + 1j * np.array(complex_stft_mix[:, :, :, :, 1])
    complex_stft_mix_numpy = complex_stft_mix_numpy.transpose([0, 3, 2, 1])

    predicted_magnitudes = predicted_magnitudes.detach().data.cpu()
    predicted_magnitudes = np.array(predicted_magnitudes).transpose(3, 2, 0, 1)
    predicted_complex_stft = norbert.wiener(predicted_magnitudes, complex_stft_mix_numpy[0])
    real, imag = np.real(predicted_complex_stft), np.imag(predicted_complex_stft)
    real = real.transpose(2, 3, 1, 0)
    imag = imag.transpose(2, 3, 1, 0)

    torch_predicted_stft = torch.stack((torch.tensor(real), torch.tensor(imag)), dim=4).float()
    return torch_predicted_stft


def reconstruct_with_masks(est_masks, stft_all, log_resample=True, stft_frame=1022,
                           norbert=False):
    """
    :param log_mix: log-magnitude log-sampled stft of the mix
    :param est_masks: estimated masks
    :param gt_masks: gt masks
    :param stft_all: raw stft data, sources and mix, magnitude and phase
    :return:

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    audio_len = 65535
    stft_hop = 256

    phase_mix = stft_all[:, 0, 1, :, :]

    # unwarp log scale
    N = 13
    B = est_masks.size(0)

    if log_resample:
        grid_unwarp = warpgrid(B, stft_frame // 2 + 1, est_masks[0].size(2), warp=False)
        pred_masks_linear = F.grid_sample(est_masks.detach(), torch.FloatTensor(grid_unwarp).to(device))
    else:
        pred_masks_linear = est_masks.detach()

    # convert into numpy
    mag_mix = stft_all[:, 0, 0, :, :].unsqueeze_(1).float().to(device)
    phase_mix = phase_mix.unsqueeze_(1).float().to(device)
    mix_real, mix_im = inv_magphase(mag_mix, phase_mix)
    mix_complex_stft = torch.stack((mix_real, mix_im), dim=4).float().to(device)

    real_orig, im_orig = inv_magphase(stft_all[:, 1:, 0, :, :], stft_all[:, 1:, 1, :, :])
    all_orig_stft_matrix = torch.stack((real_orig, im_orig), dim=4).float().to(device)

    all_predicted_mags = mag_mix*pred_masks_linear.float()

    if norbert:
        all_predicted_stft_matrix = preprocess_with_norbert(mix_complex_stft, all_predicted_mags)
        all_predicted_stft_matrix = all_predicted_stft_matrix.to(device)
    else:
        all_phase_mix = phase_mix.repeat((1, N, 1, 1))
        real, im = inv_magphase(all_predicted_mags, all_phase_mix)
        all_predicted_stft_matrix = torch.stack((real, im), dim=4)

    window = torch.hann_window(stft_frame).to(device)
    all_pred_wav = [None for j in range(B)]
    all_orig_wav = [None for j in range(B)]
    # loop over each sample
    for j in range(B):
        all_pred_wav[j] = istft(stft_matrix=all_predicted_stft_matrix[j],
                                n_fft=stft_frame, hop_length=stft_hop, window=window,
                                length=audio_len)
        all_orig_wav[j] = istft(stft_matrix=all_orig_stft_matrix[j],
                                n_fft=stft_frame, hop_length=stft_hop, window=window,
                                length=audio_len)

    return torch.stack(all_pred_wav), torch.stack(all_orig_wav)


def _log_resample(data, log_sample_n=256, segment_len=256):
    # log scale
    data_new = torch.zeros((len(data), 14, 2, log_sample_n, segment_len)).to(device)
    grid_warp = torch.FloatTensor(warpgrid(data.shape[1], log_sample_n, segment_len, warp=True)).to(device)
    for sample_idx in range(len(data)):
        data_new[sample_idx] = F.grid_sample(data[sample_idx], grid_warp)
    return data_new


def _compute_masks(specs, ratio_masks=False):

    sources = specs[:, 1:, 0, :, :]
    mix = specs[:, 0, 0, :, :]
    mix.unsqueeze_(1)

    if not ratio_masks:
        # ideal binary masks
        gt_masks = (sources >= (mix-sources)).float()
    else:
        gt_masks = sources / torch.sum(sources, dim=1).unsqueeze_(1)

    return mix, gt_masks


def _get_spectrograms(sources, window=None, stft_frame=1022, stft_hop=256,
                      mock_spec_size=(14, 2, 512, 256), non_zero_ids=None):
    if not window:
        window = torch.hann_window(stft_frame).to(device)

    eps = 1e-4
    bs = len(sources)
    specs = torch.zeros(bs, *mock_spec_size).to(device)
    for sample_idx, sample in enumerate(sources):
        if non_zero_ids:
            for idx in non_zero_ids:
                source_stft = torch.stft(sample[idx], n_fft=stft_frame, hop_length=stft_hop, window=window)
                mag, phase = magphase(source_stft)
                specs[sample_idx, idx, 0, :, :] = mag + eps
                specs[sample_idx, idx, 1, :, :] = phase
        else:
            for source_idx, source in enumerate(sample):
                source_stft = torch.stft(source, n_fft=stft_frame, hop_length=stft_hop, window=window)
                mag, phase = magphase(source_stft)
                specs[sample_idx, source_idx, 0, :, :] = mag + eps
                specs[sample_idx, source_idx, 1, :, :] = phase
    return specs


# copy from torchaudio for compatibility
def istft(stft_matrix,          # type: Tensor
          n_fft,                # type: int
          hop_length=None,      # type: Optional[int]
          win_length=None,      # type: Optional[int]
          window=None,          # type: Optional[Tensor]
          center=True,          # type: bool
          pad_mode='reflect',   # type: str
          normalized=False,     # type: bool
          onesided=True,        # type: bool
          length=None           # type: Optional[int]
          ):
    # type: (...) -> Tensor
    r"""Inverse short time Fourier Transform. This is expected to be the inverse of torch.stft.
    It has the same parameters (+ additional optional parameter of ``length``) and it should return the
    least squares estimation of the original signal. The algorithm will check using the NOLA condition (
    nonzero overlap).

    Important consideration in the parameters ``window`` and ``center`` so that the envelop
    created by the summation of all the windows is never zero at certain point in time. Specifically,
    :math:`\sum_{t=-\infty}^{\infty} w^2[n-t\times hop\_length] \cancel{=} 0`.

    Since stft discards elements at the end of the signal if they do not fit in a frame, the
    istft may return a shorter signal than the original signal (can occur if ``center`` is False
    since the signal isn't padded).

    If ``center`` is True, then there will be padding e.g. 'constant', 'reflect', etc. Left padding
    can be trimmed off exactly because they can be calculated but right padding cannot be calculated
    without additional information.

    Example: Suppose the last window is:
    [17, 18, 0, 0, 0] vs [18, 0, 0, 0, 0]

    The n_frames, hop_length, win_length are all the same which prevents the calculation of right padding.
    These additional values could be zeros or a reflection of the signal so providing ``length``
    could be useful. If ``length`` is ``None`` then padding will be aggressively removed
    (some loss of signal).

    [1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
    IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.

    Args:
        stft_matrix (torch.Tensor): Output of stft where each row of a channel is a frequency and each
            column is a window. it has a size of either (channel, fft_size, n_frames, 2) or (
            fft_size, n_frames, 2)
        n_fft (int): Size of Fourier transform
        hop_length (Optional[int]): The distance between neighboring sliding window frames.
            (Default: ``win_length // 4``)
        win_length (Optional[int]): The size of window frame and STFT filter. (Default: ``n_fft``)
        window (Optional[torch.Tensor]): The optional window function.
            (Default: ``torch.ones(win_length)``)
        center (bool): Whether ``input`` was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (str): Controls the padding method used when ``center`` is True. (Default:
            ``'reflect'``)
        normalized (bool): Whether the STFT was normalized. (Default: ``False``)
        onesided (bool): Whether the STFT is onesided. (Default: ``True``)
        length (Optional[int]): The amount to trim the signal by (i.e. the
            original signal length). (Default: whole signal)

    Returns:
        torch.Tensor: Least squares estimation of the original signal of size
        (channel, signal_length) or (signal_length)
    """
    stft_matrix_dim = stft_matrix.dim()
    assert 3 <= stft_matrix_dim <= 4, ('Incorrect stft dimension: %d' % (stft_matrix_dim))

    if stft_matrix_dim == 3:
        # add a channel dimension
        stft_matrix = stft_matrix.unsqueeze(0)

    device = stft_matrix.device
    fft_size = stft_matrix.size(1)
    assert (onesided and n_fft // 2 + 1 == fft_size) or (not onesided and n_fft == fft_size), (
        'one_sided implies that n_fft // 2 + 1 == fft_size and not one_sided implies n_fft == fft_size. ' +
        'Given values were onesided: %s, n_fft: %d, fft_size: %d' % ('True' if onesided else False, n_fft, fft_size))

    # use stft defaults for Optionals
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # There must be overlap
    assert 0 < hop_length <= win_length
    assert 0 < win_length <= n_fft

    if window is None:
        window = torch.ones(win_length)

    assert window.dim() == 1 and window.size(0) == win_length

    if win_length != n_fft:
        # center window with pad left and right zeros
        left = (n_fft - win_length) // 2
        window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        assert window.size(0) == n_fft
    # win_length and n_fft are synonymous from here on

    stft_matrix = stft_matrix.transpose(1, 2)  # size (channel, n_frames, fft_size, 2)
    stft_matrix = torch.irfft(stft_matrix, 1, normalized,
                              onesided, signal_sizes=(n_fft,))  # size (channel, n_frames, n_fft)

    assert stft_matrix.size(2) == n_fft
    n_frames = stft_matrix.size(1)

    ytmp = stft_matrix * window.view(1, 1, n_fft)  # size (channel, n_frames, n_fft)
    # each column of a channel is a frame which needs to be overlap added at the right place
    ytmp = ytmp.transpose(1, 2)  # size (channel, n_fft, n_frames)

    eye = torch.eye(n_fft, requires_grad=False,
                    device=device).unsqueeze(1)  # size (n_fft, 1, n_fft)

    # this does overlap add where the frames of ytmp are added such that the i'th frame of
    # ytmp is added starting at i*hop_length in the output
    y = torch.nn.functional.conv_transpose1d(
        ytmp, eye, stride=hop_length, padding=0)  # size (channel, 1, expected_signal_len)

    # do the same for the window function
    window_sq = window.pow(2).view(n_fft, 1).repeat((1, n_frames)).unsqueeze(0)  # size (1, n_fft, n_frames)
    window_envelop = torch.nn.functional.conv_transpose1d(
        window_sq, eye, stride=hop_length, padding=0)  # size (1, 1, expected_signal_len)

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    assert y.size(2) == expected_signal_len
    assert window_envelop.size(2) == expected_signal_len

    half_n_fft = n_fft // 2
    # we need to trim the front padding away if center
    start = half_n_fft if center else 0
    end = -half_n_fft if length is None else start + length

    y = y[:, :, start:end]
    window_envelop = window_envelop[:, :, start:end]

    # check NOLA non-zero overlap condition
    window_envelop_lowest = window_envelop.abs().min()
    assert window_envelop_lowest > 1e-11, ('window overlap add min: %f' % (window_envelop_lowest))

    y = (y / window_envelop).squeeze(1)  # size (channel, expected_signal_len)

    if stft_matrix_dim == 3:  # remove the channel dimension
        y = y.squeeze(0)
    return y


def complex_norm(complex_tensor, power=1.0):
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (torch.Tensor): Tensor shape of `(*, complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        torch.Tensor: Power of the normed input tensor. Shape of `(*, )`
    """
    if power == 1.0:
        return torch.norm(complex_tensor, 2, -1)
    return torch.norm(complex_tensor, 2, -1).pow(power)


def angle(complex_tensor):
    r"""Compute the angle of complex tensor input.

    Args:
        complex_tensor (torch.Tensor): Tensor shape of `(*, complex=2)`

    Return:
        torch.Tensor: Angle of a complex tensor. Shape of `(*, )`
    """
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])


def magphase(complex_tensor, power=1.):
    r"""Separate a complex-valued spectrogram with shape `(*, 2)` into its magnitude and phase.

    Args:
        complex_tensor (torch.Tensor): Tensor shape of `(*, complex=2)`
        power (float): Power of the norm. (Default: `1.0`)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The magnitude and phase of the complex tensor
    """
    mag = complex_norm(complex_tensor, power)
    phase = angle(complex_tensor)
    return mag, phase


def inv_magphase(mag, phase):
    """

    :param mag: magnitude of a complex spectrogram
    :param phase: phase of a complex spectrogram
    :return: Tuple[real, im] real and imaginary part of the complex spectrogram
    """
    real = mag * torch.cos(phase)
    im = mag * torch.sin(phase)
    return real, im


# from https://github.com/hmartelb/Pix2Pix-Timbre-Transfer/blob/master/code/data.py
def amplitude_to_db(mag, amin=1/(2**16), normalize=True):
    mag_db = 20*torch.log1p(mag/amin)
    if normalize:
        mag_db /= 20*torch.log1p(torch.tensor(1/amin))
    return mag_db


def db_to_amplitude(mag_db, amin=1/(2**16), normalize=True):
    if normalize:
        mag_db *= 20*np.log1p(1/amin)
    return amin*np.expm1(mag_db/20)


def rescale(x):
    # rescale x of size (bs, 1, h, w) to -1, 1
    for i in range(x.shape[0]):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    x[torch.isnan(x)] = 0  # if an entire column is zero, division by 0 will cause NaNs
    x = 2 * x - 1
    return x


def add_noise(x):
    return x + torch.randn_like(x)
