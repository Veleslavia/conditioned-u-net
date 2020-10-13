import torch.nn as nn
from models.visual import ResNetLSTMFeatures
from models.audio import MHUnet


class FeatureConditioned(nn.Module):
    """
    This class defines a u-net conditioned on visual features from frames.

    """
    def __init__(self, exp_config):
        super(FeatureConditioned, self).__init__()
        self.conditioned = True
        self.multitask = False
        # Import pre-trained visual model and let it finetune if needed
        self.visual = ResNetLSTMFeatures(with_resnet=exp_config.with_resnet,
                                         with_lstm=exp_config.with_lstm,
                                         max_pooling=exp_config.max_pooling,
                                         visual_finetune=exp_config.visual_finetune,
                                         get_probabilities=(exp_config.conditioning == 'mult_final'))

        # Construct needed audio source separation model
        self.audio = MHUnet(use_dropout=True, complementary=False,
                            multitask=False, conditioning=exp_config.conditioning,
                            use_bias=False, n_decoders=exp_config.num_decoders)

    def forward(self, x_audio, x_visual):
        visual_context = self.visual(x_visual)
        return self.audio(x_audio, visual_context)

    def train(self, mode=True):
        super(FeatureConditioned, self).train(mode=mode)
