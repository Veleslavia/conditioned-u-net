import torch
import torch.utils.data

from configs import ex, logger
from utils.attr_dict import AttrDict

from models.multimodal import FeatureConditioned
from datasets.solos import SolosMM, SolosMMFeatures
from action_pipeline import ModelActionPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
This module trains a multimodal U-Net conditioned on image features.
Multimodal networks are defined in models.multimodal

"""


@ex.automain
def main(exp_config):
    # for convenient attribute access
    exp_config = AttrDict(exp_config)

    # Select Multimodal Model Conditioned on Labels of Features
    model = FeatureConditioned(exp_config)

    # Select Dataset
    if not exp_config.with_resnet:
        loaders = [torch.utils.data.DataLoader(SolosMMFeatures(data_type, data_dir=exp_config.dataset_dir,
                                                               n_mix_max=2 if exp_config.curriculum_training else 7,
                                                               context=bool(exp_config.conditioning),
                                                               n_visual_frames=exp_config.n_visual_frames),
                                               batch_size=exp_config.batch_size,
                                               shuffle=shuffle, num_workers=8)
                   for data_type, shuffle in zip(['train', 'test'], [True, False])]
    else:
        loaders = [torch.utils.data.DataLoader(SolosMM(data_type, data_dir=exp_config.dataset_dir,
                                                       n_mix_max=2 if exp_config.curriculum_training else 7,
                                                       context=bool(exp_config.conditioning),
                                                       n_visual_frames=exp_config.n_visual_frames),
                                               batch_size=exp_config.batch_size,
                                               shuffle=shuffle, num_workers=8)
                   for data_type, shuffle in zip(['train', 'test'], [True, False])]

    pipeline = ModelActionPipeline(model=model,
                                   train_loader=loaders[0],
                                   val_loader=loaders[1],
                                   exp_config=exp_config)

    pipeline.train_model()
