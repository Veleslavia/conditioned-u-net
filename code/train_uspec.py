import os
import shutil

from configs import ex, logger
import torch.utils.data

from action_pipeline import ModelActionPipeline
from datasets.urmp import URMPSpec
from datasets.solos import SolosSpec
from models.audio import Unet, MHUnet
from utils.attr_dict import AttrDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ex.automain
def main(exp_config):

    exp_config = AttrDict(exp_config)
    num_workers = 8

    if exp_config.model == 'specunet':
        model = Unet(use_dropout=exp_config.use_dropout, complementary=exp_config.complementary,
                     conditioning=exp_config.conditioning)
    elif exp_config.model == 'mhunet':
        model = MHUnet(use_dropout=exp_config.use_dropout, complementary=exp_config.complementary,
                       multitask=exp_config.multitask, conditioning=exp_config.conditioning,
                       use_bias=exp_config.use_bias, n_decoders=exp_config.num_decoders,
                       num_downs=exp_config.num_downblocks)

    loaders = [torch.utils.data.DataLoader(SolosSpec(data_type, data_dir=exp_config.dataset_dir,
                                                     load_to_ram=exp_config.load_to_ram,
                                                     multitask=exp_config.multitask,
                                                     n_mix_max=2 if exp_config.curriculum_training else 7,
                                                     context=bool(exp_config.conditioning)),
                                           batch_size=exp_config.batch_size,
                                           shuffle=shuffle, num_workers=num_workers)
               for data_type, shuffle in zip(['train', 'test'], [True, False])]

    pipeline = ModelActionPipeline(model=model,
                                   train_loader=loaders[0],
                                   val_loader=loaders[1],
                                   exp_config=exp_config)

    pipeline.train_model()
