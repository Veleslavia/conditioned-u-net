from pathlib import Path
import numpy as np

import fire
import librosa
from mir_eval.separation import bss_eval_sources


def si_sdr(reference, estimation):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    325.10914850346956
    >>> si_sdr(reference, reference * 2)
    325.10914850346956
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.370460603257728
    >>> si_sdr(reference, reference * 2 + 1)
    6.370460603257728
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    assert reference.dtype == np.float32, reference.dtype
    assert estimation.dtype == np.float32, estimation.dtype

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


def calc_metrcs(est_path='/home/olga/vimss_torch/eval/00001/urmp_4', gt_path='/home/olga/Datasets/urmpv2/all'):
    gt_path = Path(gt_path)
    est_path = Path(est_path)
    segment_len = 65535
    sdr_all, sir_all, sar_all, si_sdr_all = list(), list(), list(), list()

    for dirname in sorted(gt_path.glob('*')):
        n_valid_sources = len(dirname.stem.split('_')) - 2
        preds_wav = [None for n in range(n_valid_sources)]
        gts_wav = [None for n in range(n_valid_sources)]
        for source_idx, source_filename in enumerate(sorted(dirname.glob('AuSep_*.wav'))):
            source_key = source_filename.stem.split('_')[2]

            pred_source_wavs = list()
            for audio_filename in sorted(est_path.glob(dirname.stem + f'/{source_key}_*.wav')):
                pred_source_wavs.append(librosa.core.load(str(audio_filename), sr=11025, mono=True)[0])
            preds_wav[source_idx] = pred_source_wavs #np.concatenate(pred_source_wavs)

            gts_source_wav, _ = librosa.core.load(str(source_filename), sr=11025, mono=True)
            gts_wav[source_idx] = gts_source_wav # gts_source_wav[:len(preds_wav[source_idx])]

        for segment_id in range(len(preds_wav[0])):
            segment_gt_wav = np.array(gts_wav)[:, segment_id*segment_len:(segment_id+1)*segment_len]
            segment_pred_wav = np.array(preds_wav)[:, segment_id]

            valid = True
            for source_id, gt_source in enumerate(segment_gt_wav):
                valid *= np.sum(np.abs(gt_source)) > 1e-5
                valid *= np.sum(np.abs(segment_pred_wav[source_id])) > 1e-5
            if valid:
                sdr, sir, sar, _ = bss_eval_sources(
                    np.asarray(segment_gt_wav),
                    np.asarray(segment_pred_wav),
                    False)
                si_sdr_ = si_sdr(np.asarray(segment_gt_wav), np.asarray(segment_pred_wav))
                sdr_all.append(sdr)
                sir_all.append(sir)
                sar_all.append(sar)
                si_sdr_all.append(si_sdr_)
                if np.mean(sdr) > 2 and all(sdr > 0):
                    print(sdr, source_filename, segment_id)

    np.save(est_path / 'sdr_all', sdr_all)
    np.save(est_path / 'sir_all', sir_all)
    np.save(est_path / 'sar_all', sar_all)
    np.save(est_path / 'si_sdr_all', si_sdr_all)
    print(est_path)
    print(f"$ {np.concatenate(sdr_all).mean():.2f}\pm{np.concatenate(sdr_all).std():.2f} $ & "
          f"$ {np.concatenate(sir_all).mean():.2f}\pm{np.concatenate(sir_all).std():.2f} $ & "
          f"$ {np.concatenate(sar_all).mean():.2f}\pm{np.concatenate(sar_all).std():.2f} $ & "
          f"$ {np.concatenate(si_sdr_all).mean():.2f}\pm{np.concatenate(si_sdr_all).std():.2f} $")


if __name__ == '__main__':
    fire.Fire(calc_metrcs)
