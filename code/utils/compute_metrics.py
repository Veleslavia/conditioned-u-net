import os
import glob
import numpy as np

import fire
import librosa
import mir_eval

from si_sdr import si_sdr


def load_audio(target_path, eval_path, piece_name, source_names):
    target_source_data = list()
    estimated_source_data = list()
    for source_name in source_names:
        target_source_data.append(librosa.load(
            glob.glob(os.path.join(target_path, piece_name, '*' + source_name))[0])[0])
        estimated_source_data.append(librosa.load(
            glob.glob(os.path.join(eval_path, piece_name, '*' + source_name))[0])[0])
    return np.array(target_source_data), np.array(estimated_source_data)


def one_piece_filenames(filenames):
    source_filenames = list()
    old_key = filenames[0].split('/')[-2]
    for filename in filenames:
        key = filename.split('/')[-2]
        if key == old_key:
            source_filenames.append(filename.split('/')[-1])
        else:
            yield old_key, source_filenames
            source_filenames = [filename.split('/')[-1]]
            old_key = key


def get_cutted_target(target_audio):
    num_samples = 16384
    mix_with_padding = 147442
    offset = (mix_with_padding - num_samples) // 2
    start_idx = offset
    last_valid = target_audio.shape[1] - offset - 1
    end_idx = start_idx + ((last_valid - start_idx) // num_samples) * num_samples
    return target_audio[:, start_idx: end_idx]


def compute_metrics(target_audio_path: str, estimated_audio_path: str):
    """
    Compute and save metrics

    :param target_audio_path: Path to target ground truth audio, e.g. /home/olga/Datasets/urmpv2/test
    :param estimated_audio_path: Path to estimated audio sources, e.g. /home/olga/vimss_torch/eval/00001
    :return:
    """

    SDR, SIR, SAR, SI_SDR = list(range(4))
    metrics = [list() for _ in range(4)]

    for piece_name, source_names in one_piece_filenames(glob.glob(os.path.join(target_audio_path, "*/AuSep*.wav"))):
        target_audio, estimated_audio = load_audio(target_audio_path, estimated_audio_path, piece_name, source_names)

        cutted_target_audio = get_cutted_target(target_audio)
        s_sdr, s_sir, s_sar, _ = mir_eval.separation.bss_eval_sources(cutted_target_audio,
                                                                      estimated_audio,
                                                                      compute_permutation=False)
        s_si_sdr = si_sdr(cutted_target_audio, estimated_audio)
        metrics[SDR].append(s_sdr)
        metrics[SIR].append(s_sir)
        metrics[SAR].append(s_sar)
        metrics[SI_SDR].append(s_si_sdr)
        print("Metrics for piece {piece_name}: \nSDR {s_sdr} \nSIR {s_sir} \nSAR {s_sar} \nSI-SDR {s_si_sdr}".format(
            piece_name=piece_name, s_sdr=s_sdr, s_sir=s_sir, s_sar=s_sar, s_si_sdr=s_si_sdr
        ))

    print("Metrics computed")

    print("Mean SDR: ", np.concatenate(metrics[SDR]).ravel().mean())
    print("Mean SIR: ", np.concatenate(metrics[SIR]).ravel().mean())
    print("Mean SAR: ", np.concatenate(metrics[SAR]).ravel().mean())
    print("Mean SI-SDR: ", np.concatenate(metrics[SI_SDR]).ravel().mean())

    metrics_data_path = os.path.join(estimated_audio_path, 'metrics.npy')
    np.save(metrics_data_path, np.array(metrics))
    print("Metrics saved at ", metrics_data_path)


def eval_silent_frames(true_source, predicted_source, window_size: int, hop_size: int, eval_incomplete_last_frame=False,
                       eps_for_silent_target=True):
    """
    :param true_source: true source signal in the time domain, numpy array with shape (T,)
    :param predicted_source: predicted source signal in the time domain, numpy array with shape (T,)
    :param window_size: length (in samples) of the window used for the framewise bss_eval metrics computation
    :param hop_size: hop size (in samples) used for the framewise bss_eval metrics computation
    :param eval_incomplete_last_frame: if True, takes last frame into account even if it is shorter than the window,
    default: False
    :param eps_for_silent_target: if True, returns a value also if target source is silent, set to False for exact same
    behaviour as explained in the paper "Weakly Informed Audio Source Separation", default: True
    :return: pes: numpy array containing PES values for all applicable frames
             eps: numpy array containing EPS values for all applicable frames
             silent_true_source_frames: list of indices of frames with silent target source
             silent_prediction_frames: list of indices of frames with silent predicted source
    """

    # check inputs
    assert true_source.ndim == 1, "true source array has too many dimensions, expected shape is (T,)"
    assert predicted_source.ndim == 1, "predicted source array has too many dimensions, expected shape is (T,)"
    assert len(true_source) == len(predicted_source), "true source and predicted source must have same length"

    # compute number of evaluation frames
    number_eval_frames = int(np.ceil((len(true_source) - window_size) / hop_size)) + 1

    last_frame_incomplete = False
    if len(true_source) % hop_size != 0:
        last_frame_incomplete = True

    # values for each frame will be gathered here
    pes_list = []
    eps_list = []

    # indices of frames with silence will be gathered here
    silent_true_source_frames = []
    silent_prediction_frames = []

    for n in range(number_eval_frames):

        # evaluate last frame if applicable
        if n == number_eval_frames - 1 and last_frame_incomplete:
            if eval_incomplete_last_frame:
                prediction_window = predicted_source[n * hop_size:]
                true_window = true_source[n * hop_size:]
            else:
                continue

        # evaluate other frames
        else:
            prediction_window = predicted_source[n * hop_size: n * hop_size + window_size]
            true_window = true_source[n * hop_size: n * hop_size + window_size]

        # compute Predicted Energy at Silence (PES)
        if sum(abs(true_window)) == 0:
            pes = 10 * np.log10(sum(prediction_window ** 2) + 10 ** (-12))
            pes_list.append(pes)
            silent_true_source_frames.append(n)

        # compute Energy at Predicted Silence (EPS)
        if eps_for_silent_target:
            if sum(abs(prediction_window)) == 0:
                true_source_energy_at_silent_prediction = 10 * np.log10(sum(true_window ** 2) + 10 ** (-12))
                eps_list.append(true_source_energy_at_silent_prediction)
                silent_prediction_frames.append(n)

        else:
            if sum(abs(prediction_window)) == 0 and sum(abs(true_window)) != 0:
                true_source_energy_at_silent_prediction = 10 * np.log10(sum(true_window ** 2) + 10 ** (-12))
                eps_list.append(true_source_energy_at_silent_prediction)
                silent_prediction_frames.append(n)

    pes = np.asarray(pes_list)
    eps = np.asarray(eps_list)

    return pes, eps, silent_true_source_frames, silent_prediction_frames


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    fire.Fire(compute_metrics)
