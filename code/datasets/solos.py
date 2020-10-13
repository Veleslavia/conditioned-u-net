import json
from pathlib import Path
import random

import numpy as np

from PIL import Image
import soundfile as sf
import torch
from .data_utils import magphase
import torch.utils.data
import torchvision.transforms.functional as TF


class Solos(torch.utils.data.Dataset):
    pass


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SolosSpec(Solos):
    def __init__(self, data_type, data_dir,
                 load_to_ram=True,
                 multitask=False, n_mix_max=7,
                 context=False):
        """

        :param data_type: 'train', 'validation'
        """

        self.in_ram = load_to_ram
        self.type = data_type
        self.data_dir = data_dir
        self.multimodal = False
        self.multitask = multitask
        self.context = context

        self.n_instruments = 13
        self.max_sources_in_mix = 7
        self.sources = ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
                        'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']
        # self.source_weights = [3, 11, 10, 3, 11, 5, 6, 9, 9, 12, 5, 12, 20]
        self.source_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.n_mix_min = 2
        self.n_mix_max = n_mix_max
        self.sr = 11025
        self.audio_len = 65535
        self.stft_frame = 1022
        self.stft_hop = 256
        self.log_sample_n = 256
        self.segment_len = 256
        self.eps = 1e-4
        self.mock_spec_size = (14, 2, 512, 256)

        self.window = torch.hann_window(self.stft_frame)
        self.meta = self._load_meta()
        if self.in_ram:
            self.data = dict()
            self._load_data()

    def increase_n_mix(self):
        if self.n_mix_max < self.max_sources_in_mix:
            self.n_mix_max += 1
            return True
        return False

    def _load_meta(self):
        suffix = '*.wav'
        meta = dict([(source, sorted(list((Path(self.data_dir) / source).glob(suffix)))) for source in self.sources])

        for source in meta:
            source_len = len(meta[source])
            meta_slice = slice(int(0.8*source_len)) if self.type == 'train' else slice(int(0.8*source_len), source_len)
            meta[source] = meta[source][meta_slice]

        for source in meta:
            for path_idx, path in enumerate(meta[source]):
                meta[source][path_idx] = (path, sf.info(path.as_posix()).frames)

        return meta

    def _load_data(self):
        for source in self.meta:
            self.data[source] = list()
            for filename, filelen in self.meta[source]:
                self.data[source].append(torch.FloatTensor(
                    sf.read(filename.as_posix())[0]))
            print(f'Loaded source: {source}')

    def __len__(self):
        return 8000 if self.type == 'train' else 2000

    def __getitem__(self, item):
        k = random.randint(self.n_mix_min, self.n_mix_max)
        mock_size = (self.n_instruments+1, self.audio_len)
        sources = torch.zeros(mock_size)

        source_indices = np.random.choice(list(range(self.n_instruments)), size=k, replace=False,
                                          p=np.array(self.source_weights) / sum(self.source_weights))

        aux_output = np.zeros(self.n_instruments)
        aux_source_indices = [0]

        for source_idx in source_indices:
            aux_output[source_idx] = 1
            aux_source_indices.append(source_idx+1)
            source_name = self.sources[source_idx]
            sample_idx = random.randint(0, len(self.data[source_name])-1)
            offset = random.randint(0, len(self.data[source_name][sample_idx]) - self.audio_len - 1)
            sources[source_idx+1] = self.data[source_name][sample_idx][offset: offset+self.audio_len]
            smax, smin = sources[source_idx+1].max(), sources[source_idx+1].min()
            if not np.isclose((smax - smin), [0.0]):
                sources[source_idx+1] = (sources[source_idx+1] - smin) / (smax - smin) * 2 - 1

        sources[0] = sum(sources) / k

        model_input = [sources, aux_output] if self.context else sources
        return model_input, [], aux_output, aux_source_indices

    def save_predicted(self, *args, **kwargs):
        pass


class SolosMM(SolosSpec):
    def __init__(self, data_type, data_dir, n_mix_max=7,
                 context=True, n_visual_frames=15, n_mix_min=2):
        """

        :param data_type: 'train', 'validation'
        """

        self.type = data_type
        self.data_dir = data_dir
        self.multimodal = True
        self.context = True

        self.n_instruments = 13
        self.max_sources_in_mix = 7
        self.sources = ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
                        'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']
        self.source_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.n_mix_min = n_mix_min
        self.n_mix_max = n_mix_max

        self.sr = 11025
        self.audio_len = 65535
        self.stft_frame = 1022
        self.stft_hop = 256
        self.log_sample_n = 256
        self.segment_len = 256
        self.eps = 1e-4
        self.mock_spec_size = (14, 2, 512, 256)
        self.visual_shape = (3, 224, 224)
        self.n_visual_frames = n_visual_frames
        self.video_fr = 25
        self.frames_load_fr = int(self.video_fr*6/self.n_visual_frames)

        self.window = torch.hann_window(self.stft_frame)
        self.meta = self._load_meta()

    def increase_n_mix(self):
        if self.n_mix_max < self.max_sources_in_mix:
            self.n_mix_max += 1
            return True
        return False

    def __len__(self):
        return 8000 if self.type == 'train' else 2000

    def __getitem__(self, item):
        k = random.randint(self.n_mix_min, self.n_mix_max)
        mock_size = (self.n_instruments+1, self.audio_len)
        sources = torch.zeros(mock_size)

        source_indices = np.random.choice(list(range(self.n_instruments)), size=k, replace=False,
                                          p=np.array(self.source_weights) / sum(self.source_weights))

        frames = torch.zeros(self.max_sources_in_mix, self.n_visual_frames, *self.visual_shape)
        aux_output = np.zeros(self.n_instruments)

        for idx, source_idx in enumerate(source_indices):
            aux_output[source_idx] = 1
            source_name = self.sources[source_idx]
            filename, file_len = random.choice(self.meta[source_name])
            offset = random.randint(0, file_len - self.audio_len - 1)
            sources[source_idx+1] = torch.FloatTensor(
                sf.read(filename.as_posix(),
                        frames=self.audio_len,
                        start=offset)[0])
            smax, smin = sources[source_idx+1].max(), sources[source_idx+1].min()
            if not np.isclose((smax - smin), [0.0]):
                sources[source_idx+1] = (sources[source_idx+1] - smin) / (smax - smin) * 2 - 1

            # LOAD SOURCE FRAMES HERE

            image_dirname = str(filename).replace('wav11k', 'frames').replace('.wav', '')
            divider = self.sr / self.video_fr
            first_image_idx = int(offset / divider) + 1
            for i in range(self.n_visual_frames):
                filename = Path(image_dirname) / f'{(first_image_idx+i*self.frames_load_fr):04d}.jpg'
                if filename.exists():
                    image = Image.open(filename)
                    frames[idx][i] = TF.to_tensor(image)
                else:
                    print(filename)

        sources[0] = sum(sources) / k

        return [sources, frames], [], aux_output

    def save_predicted(self, *args, **kwargs):
        pass


class SolosMMFeatures(SolosMM):
    def __getitem__(self, item):
        k = random.randint(self.n_mix_min, self.n_mix_max)
        mock_size = (self.n_instruments+1, self.audio_len)
        sources = torch.zeros(mock_size)

        source_indices = np.random.choice(list(range(self.n_instruments)), size=k, replace=False,
                                          p=np.array(self.source_weights) / sum(self.source_weights))

        frames = torch.zeros(self.max_sources_in_mix, self.n_visual_frames, 2048)
        aux_output = np.zeros(self.n_instruments)

        for idx, source_idx in enumerate(source_indices):
            aux_output[source_idx] = 1
            source_name = self.sources[source_idx]
            filename, file_len = random.choice(self.meta[source_name])
            offset = random.randint(0, file_len - self.audio_len - 1)
            sources[source_idx+1] = torch.FloatTensor(
                sf.read(filename.as_posix(),
                        frames=self.audio_len,
                        start=offset)[0])
            smax, smin = sources[source_idx+1].max(), sources[source_idx+1].min()
            if not np.isclose((smax - smin), [0.0]):
                sources[source_idx+1] = (sources[source_idx+1] - smin) / (smax - smin) * 2 - 1

            # LOAD SOURCE FRAMES FEATURES HERE
            # Doesn't affect cluster runs

            image_dirname = str(filename).replace('wav11k', 'features').replace('.wav', '')
            divider = self.sr / self.video_fr
            first_image_idx = int(offset / divider) + 1
            for i in range(self.n_visual_frames):
                filename = Path(image_dirname) / f'{(first_image_idx+i*self.frames_load_fr):04d}.jpg.npy'
                if filename.exists():
                    frames[idx][i] = torch.from_numpy(np.load(filename)).float()
                else:
                    print(filename)

        sources[0] = sum(sources) / k

        return [sources, frames], [], aux_output



def make_solos_specs(data_dir, spec_dir):
    """

    :param data_dir: '/storage/Datasets/Solos/wav11k'
    :param spec_dir: '/storage/Datasets/Solos/spec11k'
    :return:
    """

    sources = ['Bassoon', 'Cello', 'Clarinet', 'DoubleBass', 'Flute',
               'Horn', 'Oboe', 'Saxophone', 'Trombone', 'Trumpet', 'Tuba', 'Viola', 'Violin']

    stft_frame = 1022
    stft_hop = 256
    eps = 1e-4
    window = torch.hann_window(stft_frame).to(device)

    for source in sources:
        print(f"processing source {source}")
        spec_out_dir = Path(spec_dir) / source
        spec_out_dir.mkdir(exist_ok=True)
        for audiofile in (Path(data_dir) / source).glob('*.wav'):
            source = torch.FloatTensor(sf.read(audiofile.as_posix())[0]).to(device)
            source_stft = torch.stft(source, n_fft=stft_frame, hop_length=stft_hop, window=window)
            mag, phase = magphase(source_stft)
            np.save(spec_out_dir / audiofile.with_suffix('.mag.npy').name, mag.detach().data.cpu() + eps)
            np.save(spec_out_dir / audiofile.with_suffix('.phase.npy').name, phase.detach().data.cpu())
