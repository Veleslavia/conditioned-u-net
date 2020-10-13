from __future__ import print_function, division

import itertools
import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch.nn.functional as F


from models.custom_layers import FilmLayer, crop_and_concat, DownsampleBlock, UpsampleBlock, NoOp, STFT, Spectrogram


class WaveUNet(nn.Module):

    def __init__(self, n_sources, n_blocks=12,
                 n_filters=24, filter_size=15, merge_filter_size=5,
                 conditioning=None, context=True, output_padding=None):
        super(WaveUNet, self).__init__()
        self.n_sources = n_sources
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.merge_filter_size = merge_filter_size
        self.conditioning = conditioning    # None, 'multi', 'concat', 'film'
        self.conditioned = bool(conditioning)
        self.context = context

        if self.context:
            self.encoding_padding = 0
            self.decoding_padding = 0
            self.output_padding = output_padding
        else:
            self.encoding_padding = self.filter_size // 2
            self.decoding_padding = self.merge_filter_size // 2

        # number of input/output channels for every layer in encoder and decoder
        channels = [1] + [(i + 1) * n_filters for i in range(n_blocks+1)]

        self.encoder = self._make_encoder(channels)
        self.bottleneck = nn.Sequential(nn.Conv1d(channels[-2], channels[-1], self.filter_size,
                                                  padding=self.encoding_padding, dilation=1),
                                        nn.BatchNorm1d(channels[-1]),
                                        nn.LeakyReLU(inplace=True)
                                        )
        if self.conditioning:
            self.scale_conditioning = nn.Linear(n_sources, channels[-1])
        self.decoder = self._make_decoder(channels[::-1])
        self.output = self.output_layer()

    def forward(self, x, labels=None):
        original_audio = x

        for i_block in range(len(self.encoder)):
            if i_block % 2 == 0:
                x = self.encoder[i_block](x)
            else:
                x = self.encoder[i_block](x, labels)

        x = self.bottleneck(x)

        if self.conditioning:
            # Apply multiplicative conditioning
            scaled_labels = self.scale_conditioning(labels)
            x = torch.mul(x, scaled_labels.view(scaled_labels.shape + (1, )))

        for i_block in range(self.n_blocks):
            x = self.decoder[i_block](x, self.encoder[(-i_block-1)*2].features)

        x = crop_and_concat(x, original_audio)

        outputs = list()
        for i in range(self.n_sources):
            outputs.append(self.output(x))
        outputs = torch.stack(outputs, dim=1)
        if self.context:
            outputs = outputs[:, :, :, self.output_padding[0]: -self.output_padding[1]]
        return outputs

    def _make_encoder(self, channels):
        layers = list()
        for i in range(self.n_blocks):
            layers.append(DownsampleBlock(channels[i], channels[i+1],
                                          kernel_size=self.filter_size,
                                          padding=self.encoding_padding
                                          ))
            if self.conditioning:
                layers.append(FilmLayer())
        return nn.Sequential(*layers)

    def _make_decoder(self, channels):
        layers = list()
        for i in range(self.n_blocks):
            layers.append(UpsampleBlock(channels[i]+channels[i+1], channels[i+1],
                                        kernel_size=self.merge_filter_size,
                                        padding=self.decoding_padding,
                                        ))

        return nn.Sequential(*layers)

    def output_layer(self):
        # return an output for one source individually
        return nn.Sequential(
                nn.Conv1d(self.n_filters + 1, 1, 1),
                nn.Tanh())

# from https://github.com/hangzhaomit/Sound-of-Pixels

class Unet(nn.Module):
    def __init__(self, fc_dim=64, num_downs=5, ngf=64, use_dropout=False,
                 n_sources=13, complementary=False, conditioning=None):
        super(Unet, self).__init__()

        self.conditioning = conditioning
        self.conditioned = bool(conditioning)
        self.complementary = complementary
        # construct unet structure
        unet_block = UnetBlock(
            ngf * 8, ngf * 8, input_nc=None,
            submodule=None, innermost=True, conditioning=self.conditioned)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block, use_dropout=use_dropout)
        unet_block = UnetBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block)
        unet_block = UnetBlock(
            fc_dim, ngf, input_nc=1,
            submodule=unet_block, outermost=True)

        self.bn0 = nn.BatchNorm2d(1)
        self.unet_block = unet_block
        self.output = nn.Conv2d(ngf, n_sources, kernel_size=1)
        self.output_softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.bn0(x)
        x = self.unet_block(x)
        x = self.output(x)
        if self.complementary:
            x = self.output_softmax(x)
        return x


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, outer_nc, inner_input_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False, conditioning=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if innermost:
            inner_output_nc = inner_input_nc
        elif inner_output_nc is None:
            inner_output_nc = 2 * inner_input_nc

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_input_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if outermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3, padding=1)

            down = [downconv]
            up = [uprelu, upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)

            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            downconv = nn.Conv2d(
                input_nc, inner_input_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(
                inner_output_nc, outer_nc, kernel_size=3,
                padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost or self.noskip:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class IdentityContext(nn.Module):
    """A placeholder which admits and skips the context"""
    def __init__(self, *args, **kwargs):
        super(IdentityContext, self).__init__()

    def forward(self, x, context):
        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bias, conditioning):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conditioning = FilmLayer() if conditioning == 'film_encoder' else IdentityContext()
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, context=None):
        x = self.conv(x)
        x = self.bn(x)
        x = self.conditioning(x, context)
        x = self.relu(x)
        return x


class MHUnet(nn.Module):
    def __init__(self, num_downs=5, n_sources=13, n_decoders=13,
                 use_dropout=True, complementary=True,
                 pre_softmax_conv=False, multitask=False,
                 conditioning=None, use_bias=False):
        super(MHUnet, self).__init__()

        in_channels = [1, 16, 32, 64, 128, 256, 512]
        out_channels = [16, 32, 64, 128, 256, 512]

        # in_channels = [1, 32, 64, 128, 256, 512] + [512] * (num_downs-4)
        # out_channels = [32, 64, 128, 256, 512] + [512] * (num_downs-4)

        self.complementary = complementary
        self.conditioning = conditioning
        self.conditioned = bool(conditioning)
        self.pre_softmax_conv = pre_softmax_conv
        self.multitask = multitask

        self.encoder = nn.ModuleList([
            EncoderBlock(in_channels[i], out_channels[i], bias=use_bias, conditioning=self.conditioning)
            for i in range(num_downs)])
        self.encoder.append(nn.Conv2d(
            in_channels[-2], in_channels[-1], kernel_size=4,
            stride=2, padding=1, bias=use_bias))

        self.film = FilmLayer()

        self.bottleneck = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)) for i in range(n_decoders)])

        self.decoder = nn.ModuleList()
        for i in range(num_downs+1):
            decoder_layer = nn.ModuleList()
            decoder_output_channels = n_sources if n_decoders != n_sources and i == num_downs else in_channels[-(i + 2)]
            for j in range(n_decoders):
                if use_dropout:
                    inner_block = [nn.BatchNorm2d(decoder_output_channels), nn.ReLU(True), nn.Dropout(0.2)]
                else:
                    inner_block = [nn.BatchNorm2d(decoder_output_channels), nn.ReLU(True)]

                if i != num_downs:
                    basic_block = (
                        nn.Conv2d(out_channels[-(i+1)]+in_channels[-(i+2)], decoder_output_channels,
                                  kernel_size=3, padding=1, bias=use_bias),
                        *inner_block,
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                else:
                    basic_block = (
                        nn.Conv2d(out_channels[-(i+1)]+in_channels[-(i+2)], decoder_output_channels,
                                  kernel_size=3, padding=1, bias=use_bias),
                        *inner_block)
                decoder_layer.append(nn.Sequential(*basic_block))
            self.decoder.append(decoder_layer)

        self.output = nn.ModuleList([
            nn.Conv2d(in_channels[0], in_channels[0], kernel_size=1, padding=0, bias=use_bias)
                for i in range(n_decoders)])
        self.softmax = nn.Softmax2d()
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, n_sources)
        )

    def forward_classify(self, x):
        if self.multitask:
            classification_output = self.avgpool(x)
            classification_output = torch.flatten(classification_output, 1)
            classification_output = self.classifier(classification_output)
            return classification_output
        else:
            return None

    def forward_final_conditioning(self, x, context):
        if self.conditioning == 'film_final':
            x = self.film(x, context)
        if self.conditioning == 'mult_final':
            for i, j in itertools.product(range(x.shape[0]), range(x.shape[1])):
                x[i, j] *= context[i, j]
        return x

    def forward_bn_conditioning(self, x, context):
        if self.conditioning == 'film_bn':
            x = self.film(x, context)
        if self.conditioning == 'mult_bn':
            pass
        return x

    def forward(self, x, context=None):

        partial_enc_features = [x]
        for enc_layer in self.encoder[:-1]:
            x = enc_layer(x, context)
            partial_enc_features.append(x)

        x = self.encoder[-1](x)
        partial_enc_features.append(x)

        classification_output = self.forward_classify(x)

        if self.conditioned and 'bn' in self.conditioning:
            x = self.forward_bn_conditioning(x, context)

        partial_output = [self.bottleneck[i](x) for i in range(len(self.bottleneck))]
        for i, dec_layer in enumerate(self.decoder):
            partial_output = [dec_layer[j](
                torch.cat((
                    partial_enc_features[-(i+2)],
                    partial_output[j]), 1)) for j in range(len(dec_layer))]

        if self.pre_softmax_conv:
            partial_output = [output(partial_output[i]) for i, output in enumerate(self.output)]

        partial_output = torch.cat(partial_output, dim=1)

        if self.conditioning and 'final' in self.conditioning:
            partial_output = self.forward_final_conditioning(partial_output, context)

        partial_output = self.softmax(partial_output) if self.complementary else partial_output

        return (partial_output, classification_output) if classification_output is not None else partial_output
