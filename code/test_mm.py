import os

from configs import ex, logger
import torch.utils.data

from action_pipeline import ModelActionPipeline
from models.multimodal import FeatureConditioned
from datasets.solos import SolosMM, SolosMMFeatures
from datasets.urmp import URMPMM, URMPMMFeatures
from utils.attr_dict import AttrDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ex.automain
def main(exp_config):

    exp_config = AttrDict(exp_config)

    # Select Dataset
    if exp_config.dataset == 'urmp':
        if exp_config.with_resnet:
            ds = URMPMM(dataset_dir=exp_config.dataset_dir, context=bool(exp_config.conditioning),
                        n_visual_frames=exp_config.n_visual_frames)
        else:
            ds = URMPMMFeatures(dataset_dir=exp_config.dataset_dir, context=bool(exp_config.conditioning),
                                n_visual_frames=exp_config.n_visual_frames)
    elif exp_config.dataset == 'solos':
        if not exp_config.with_resnet:
            ds = SolosMMFeatures('test', data_dir=exp_config.dataset_dir, n_mix_max=7,
                                 context=bool(exp_config.conditioning), n_visual_frames=exp_config.n_visual_frames)
        else:
            ds = SolosMM('test', data_dir=exp_config.dataset_dir, n_mix_max=7,
                         context=bool(exp_config.conditioning), n_visual_frames=exp_config.n_visual_frames)

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Select Multimodal Model Conditioned on Labels of Features
    model = FeatureConditioned(exp_config)

    pipeline = ModelActionPipeline(model=model,
                                   train_loader=loader,
                                   val_loader=loader,
                                   exp_config=exp_config)

    checkpoint_path = os.path.join(exp_config.dir_checkpoint, exp_config.model_checkpoint)
    pipeline.test_model(checkpoint_path, exp_config.output_dir)
