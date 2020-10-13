from collections import namedtuple, defaultdict
import os
from pathlib import Path

import glob

from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.functional as TF

source_map = {
    'mix': 0,
    'bn': 1,
    'vc': 2,
    'cl': 3,
    'db': 4,
    'fl': 5,
    'hn': 6,
    'ob': 7,
    'sax': 8,
    'tbn': 9,
    'tpt': 10,
    'tba': 11,
    'va': 12,
    'vn': 13}

URMPIndex = namedtuple('URMPIndex', ['datafile', 'offset'])


def _get_source_id_from_filename(filename: str):
    stem = Path(filename).stem
    if stem.startswith('AuSep'):
        instrument_key = stem.split('_')[2]
        return source_map[instrument_key]
    elif stem.startswith('AuMix'):
        return 0
    else:
        print(filename)
        raise IndexError


class URMP(torch.utils.data.Dataset):

    def __init__(self):
        self.segment_len = 256
        self.orig_freq_n = 512
        self.log_sample_n = 256
        self.channels_n = 2     # amplitude and phase
        self.sr = 11025
        self.n_instruments = 13
        self.load_specs = True

    def __len__(self):
        return len(self.data_idx)

    def _load_stft_meta_files(self, dataset_dir, phase=False):
        key = '*/*.amp.npy' if not phase else '*/*.phase.npy'
        stft_filenames = glob.glob(os.path.join(dataset_dir, key))

        stft_meta = defaultdict(list)
        for filename in stft_filenames:
            piece_name = Path(filename).parent.stem
            stft_meta[piece_name].append(filename)
        return stft_meta

    def _load_data_from_meta(self, meta: dict):
        data = defaultdict(dict)
        # FixMe reduced memory!!!!
        for piece_name, filenames in list(meta.items()):
            for filename in filenames:
                source_idx = _get_source_id_from_filename(filename)
                data[piece_name][str(source_idx)] = np.load(filename)
            for key in set(str(value) for value in source_map.values()) - data[piece_name].keys():
                data[piece_name][key] = np.zeros(data[piece_name]['0'].shape)
        return data

    def _make_data_idx(self):
        index = list()
        # form an index as list of pairs (piece_name, segment_idx)
        for piece_name, piece_data in self.amp_data.items():
            for i in range(piece_data['0'].shape[1] // self.segment_len):
                index.append((piece_name, i))
        return index

    def save_predicted(self, output_dir, wav_predicted, item_idx, visual_probs=None):
        import librosa
        import pathlib

        train_source_map = ['bn', 'vc', 'cl', 'db', 'fl', 'hn', 'ob', 'sax', 'tbn', 'tpt', 'tba', 'va', 'vn']

        filename, segment_id = self.data_idx[item_idx]
        if not os.path.exists(pathlib.Path(output_dir) / filename):
            os.mkdir(pathlib.Path(output_dir) / filename)

        if visual_probs is not None:
            np.save(pathlib.Path(output_dir) / filename / f'visual_{segment_id:02d}.npy', visual_probs)

        for source_idx, predicted in enumerate(wav_predicted):
            source_name = train_source_map[source_idx]
            predicted_path = pathlib.Path(output_dir) / filename / f'{source_name}_{segment_id:02d}.wav'
            librosa.output.write_wav(predicted_path, predicted, self.sr)


class URMPSpec(URMP):

    def __init__(self, dataset_dir, context=False):
        super(URMPSpec, self).__init__()
        amp_meta = self._load_stft_meta_files(dataset_dir)
        phase_meta = self._load_stft_meta_files(dataset_dir, phase=True)
        self.amp_data = self._load_data_from_meta(amp_meta)
        self.phase_data = self._load_data_from_meta(phase_meta)
        self.data_idx = self._make_data_idx()
        self.multimodal = False
        self.context = context

    def __getitem__(self, item):
        # get piece_name from the index
        piece_name, segment_idx = self.data_idx[item]
        # prepare indexes
        aux_output = np.zeros(self.n_instruments)

        # get segments of the spectrograms according to the index
        data_slice = np.zeros((len(source_map), self.channels_n, self.orig_freq_n, self.segment_len))
        segment_boundaries = slice(segment_idx*self.segment_len, (segment_idx+1)*self.segment_len)
        for key in self.amp_data[piece_name].keys():
            aux_output[int(key)-1] = 1
            data_slice[int(key), 0, :, :] = self.amp_data[piece_name][key][:, segment_boundaries]
            data_slice[int(key), 1, :, :] = self.phase_data[piece_name][key][:, segment_boundaries]

        model_input = [data_slice, aux_output] if self.context else data_slice
        return model_input, [], aux_output


class URMPImg(URMP):
    pass


class URMPMM(URMP):
    def __init__(self, dataset_dir,
                 context=True, n_visual_frames=15):

        super(URMPMM, self).__init__()
        self.dataset_dir = Path(dataset_dir)
        amp_meta = self._load_stft_meta_files(dataset_dir)
        phase_meta = self._load_stft_meta_files(dataset_dir, phase=True)

        self.amp_data = self._load_data_from_meta(amp_meta)
        self.phase_data = self._load_data_from_meta(phase_meta)
        self.data_idx = self._make_data_idx()
        self.multimodal = True
        self.context = context
        self.max_sources_in_mix = 7
        self.audio_len = 65535

        self.visual_shape = (3, 224, 224)
        self.n_visual_frames = n_visual_frames
        self.video_fr = 25
        self.frames_load_fr = int(self.video_fr*6/self.n_visual_frames)
        self.inverted_source_map = dict(map(reversed, source_map.items()))

    def _load_frames(self, piece_name, segment_id):
        # 1 segment has length 256 = 25*65535/11025 = 148.6 frames per segment
        # self.frames_load_fr
        # prepare frames
        frames = torch.zeros(self.max_sources_in_mix, self.n_visual_frames, *self.visual_shape)
        first_image_idx = int(segment_id * self.video_fr * self.audio_len / self.sr) + 1
        image_dirname = self.dataset_dir / (piece_name + '/frames/')

        for idx, source_pair in enumerate(sorted([(source_map[a], a) for a in piece_name.split('_')[2:]])):
            source_name = source_pair[1]
            for i in range(self.n_visual_frames):
                filename = image_dirname / (source_name + f'_{(first_image_idx+i*self.frames_load_fr):04d}.jpg')
                if filename.exists():
                    image = Image.open(filename)
                    frames[idx][i] = TF.to_tensor(image)
                else:
                    print(filename)

        return frames

    def __getitem__(self, item):
        # get piece_name from the index
        piece_name, segment_idx = self.data_idx[item]
        # prepare indexes
        aux_output = np.zeros(self.n_instruments)

        # get segments of the spectrograms according to the index
        data_slice = np.zeros((len(source_map), self.channels_n, self.orig_freq_n, self.segment_len))
        segment_boundaries = slice(segment_idx*self.segment_len, (segment_idx+1)*self.segment_len)
        for key in self.amp_data[piece_name].keys():
            aux_output[int(key)-1] = 1
            data_slice[int(key), 0, :, :] = self.amp_data[piece_name][key][:, segment_boundaries]
            data_slice[int(key), 1, :, :] = self.phase_data[piece_name][key][:, segment_boundaries]

        frames = self._load_frames(piece_name, segment_idx)

        return [data_slice, frames], [], aux_output


class URMPMMFeatures(URMP):
    def __init__(self, dataset_dir,
                 context=True, n_visual_frames=15):

        super(URMPMMFeatures, self).__init__()
        self.dataset_dir = Path(dataset_dir)
        amp_meta = self._load_stft_meta_files(dataset_dir)
        phase_meta = self._load_stft_meta_files(dataset_dir, phase=True)

        self.amp_data = self._load_data_from_meta(amp_meta)
        self.phase_data = self._load_data_from_meta(phase_meta)
        self.data_idx = self._make_data_idx()
        self.multimodal = True
        self.context = context
        self.max_sources_in_mix = 7
        self.audio_len = 65535

        self.visual_shape = (3, 224, 224)
        self.n_visual_frames = n_visual_frames
        self.video_fr = 25
        self.frames_load_fr = int(self.video_fr*6/self.n_visual_frames)
        self.inverted_source_map = dict(map(reversed, source_map.items()))

    def _load_visual_features(self, piece_name, segment_id):
        # 1 segment has length 256 = 25*65535/11025 = 148.6 frames per segment
        # self.frames_load_fr
        # prepare frames
        frames = torch.zeros(self.max_sources_in_mix, self.n_visual_frames, 2048)
        first_image_idx = int(segment_id * self.video_fr * self.audio_len / self.sr) + 1
        image_dirname = self.dataset_dir / (piece_name + '/features/')

        for idx, source_pair in enumerate(sorted([(source_map[a], a) for a in piece_name.split('_')[2:]])):
            source_name = source_pair[1]
            for i in range(self.n_visual_frames):
                filename = image_dirname / (source_name + f'_{(first_image_idx+i*self.frames_load_fr):04d}.jpg.npy')
                if filename.exists():
                    frames[idx][i] = torch.from_numpy(np.load(filename)).float()
                else:
                    print(filename)

        return frames

    def __getitem__(self, item):
        # get piece_name from the index
        piece_name, segment_idx = self.data_idx[item]
        # prepare indexes
        aux_output = np.zeros(self.n_instruments)

        # get segments of the spectrograms according to the index
        data_slice = np.zeros((len(source_map), self.channels_n, self.orig_freq_n, self.segment_len))
        segment_boundaries = slice(segment_idx*self.segment_len, (segment_idx+1)*self.segment_len)
        for key in self.amp_data[piece_name].keys():
            aux_output[int(key)-1] = 1
            data_slice[int(key), 0, :, :] = self.amp_data[piece_name][key][:, segment_boundaries]
            data_slice[int(key), 1, :, :] = self.phase_data[piece_name][key][:, segment_boundaries]

        frames = self._load_visual_features(piece_name, segment_idx)

        return [data_slice, frames], [], aux_output

