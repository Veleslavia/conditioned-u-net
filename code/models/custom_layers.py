"""
This module defines different conditioning layers, i.e. concatenation, multiplicative, FiLM
"""

import math
import torch
from torch import nn

"""
from https://github.com/ap229997/Neural-Toolbox-PyTorch/blob/master/dynamic_fc.py
dynamic FC - for varying dimensions on the go
Input - embedding in (batch_size(N), * , channels(C)) [* represent extra dimensions]
"""


class DynamicFC(nn.Module):

    def __init__(self):
        super(DynamicFC, self).__init__()

        self.in_planes = None
        self.out_planes = None
        self.activation = None
        self.use_bias = None

        self.activation = None
        self.linear = None
        self.initialized = False

    def forward(self, embedding, out_planes=1, activation=None, use_bias=True):
        """
        Arguments:
            embedding : input to the MLP (N,*,C)
            out_planes : total channels in the output
            activation : 'relu' or 'tanh'
            use_bias : True / False
        Returns:
            out : output of the MLP (N,*,out_planes)
        """

        self.in_planes = embedding.data.shape[-1]
        self.out_planes = out_planes
        self.use_bias = use_bias

        if not self.initialized:
            self.linear = nn.Linear(self.in_planes, self.out_planes, bias=use_bias).cuda()
            if activation == 'relu':
                self.activation = nn.ReLU(inplace=True).cuda()
            elif activation == 'tanh':
                self.activation = nn.Tanh().cuda()

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    print("initialize conditioning")
                    if self.use_bias:
                        nn.init.constant_(m.bias, 0.1)
            self.initialized = True

        out = self.linear(embedding)
        if self.activation is not None:
            out = self.activation(out)

        return out


"""
from https://github.com/ap229997/Neural-Toolbox-PyTorch/blob/master/film_layer.py
FiLM layer with a linear transformation from context to FiLM parameters
"""


class FilmLayer(nn.Module):

    def __init__(self):
        super(FilmLayer, self).__init__()

        self.batch_size = None
        self.channels = None
        self.height = None
        self.width = None
        self.feature_size = None

        self.fc = DynamicFC()

    def film3d(self, feature_maps, film_params):
        self.batch_size, self.channels, self.height = feature_maps.data.shape

        # stack the FiLM parameters across the temporal dimension
        film_params = torch.stack([film_params] * self.height, dim=2)

        # slice the film_params to get betas and gammas
        gammas = film_params[:, :self.feature_size, :]
        betas = film_params[:, self.feature_size:, :]

        return gammas, betas

    def film4d(self, feature_maps, film_params):
        self.batch_size, self.channels, self.height, self.width = feature_maps.data.shape

        # stack the FiLM parameters across the spatial dimension
        film_params = torch.stack([film_params] * self.height, dim=2)
        film_params = torch.stack([film_params] * self.width, dim=3)

        # slice the film_params to get betas and gammas
        gammas = film_params[:, :self.feature_size, :, :]
        betas = film_params[:, self.feature_size:, :, :]

        return gammas, betas

    def forward(self, feature_maps, context):
        """
        Arguments:
            feature_maps : input feature maps (N, C, H, W) or (N, C, W)
            context : context embedding (N, L)
        Return:
            output : feature maps modulated with betas and gammas (FiLM parameters)
        """

        # FiLM parameters needed for each channel in the feature map
        # hence, feature_size defined to be same as no. of channels
        self.feature_size = feature_maps.data.shape[1]

        # linear transformation of context to FiLM parameters
        film_params = self.fc(context, out_planes=2 * self.feature_size, activation=None)

        gammas, betas = self.film4d(feature_maps, film_params) if len(feature_maps.data.shape) == 4 else self.film3d(feature_maps, film_params)
        # modulate the feature map with FiLM parameters
        output = (1 + gammas) * feature_maps + betas

        return output


def crop_and_concat(x1, x2):
    # input is [Channels, Width], crop and concat functionality
    diff = x2.size()[2] - x1.size()[2]
    if diff > 0:
        x2 = x2[:, :, math.floor(diff/2): -(diff - math.floor(diff/2))]
    x = torch.cat((x2, x1), dim=1)
    return x


class DownsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=0):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.features = None

    def forward(self, x):
        x = self.conv(x)
        self.features = x
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='linear')
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=1, padding=0):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='linear')
        x = crop_and_concat(x1, x2)
        x = self.conv(x)
        return x


# open-unmix layers from https://github.com/sigsep/open-unmix-pytorch/blob/master/model.py

class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)
