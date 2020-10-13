import os

from configs import ex, logger
import torch.utils.data

from action_pipeline import ModelActionPipeline
from datasets.solos import SolosSpec
from datasets.urmp import URMPSpec
from models.audio import Unet, MHUnet
from utils.attr_dict import AttrDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ex.automain
def main(exp_config):

    exp_config = AttrDict(exp_config)

    model = MHUnet(use_dropout=exp_config.use_dropout, complementary=exp_config.complementary,
                   multitask=exp_config.multitask, conditioning=exp_config.conditioning,
                   use_bias=exp_config.use_bias, n_decoders=exp_config.num_decoders,
                   num_downs=exp_config.num_downblocks)

    if exp_config.dataset == 'urmp':
        ds = URMPSpec(dataset_dir=exp_config.dataset_dir, context=bool(exp_config.conditioning))
    elif exp_config.dataset == 'solos':
        ds = SolosSpec('test', data_dir=exp_config.dataset_dir, load_specs=(exp_config.input_type == 'spec_load'),
                       context=bool(exp_config.conditioning))

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    pipeline = ModelActionPipeline(model=model,
                                   train_loader=loader,
                                   val_loader=loader,
                                   exp_config=exp_config)

    checkpoint_path = os.path.join(exp_config.dir_checkpoint, exp_config.model_checkpoint)
    pipeline.test_model(checkpoint_path, exp_config.output_dir)
