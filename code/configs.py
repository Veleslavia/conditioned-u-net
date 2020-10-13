import logging

from sacred import Experiment
from sacred.observers import FileStorageObserver

logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)

ex = Experiment('unet')
ex.observers.append(FileStorageObserver.create('experiments_meta'))

@ex.config
def cfg():
    # Base configuration
    exp_config = dict(mode='train',
                      dataset_dir='/Datasets/Solos/wav11k',
                      model='mhunet',
                      num_sources=13,
                      num_decoders=1,
                      num_channels=1,
                      num_downblocks=5,
                      # None, multi, concat, sum, film_bn, film_final, film_encoder, mult_bn, mult_final
                      conditioning=None,
                      use_dropout=True,
                      use_bias=False,
                      complementary=True,   # softmax in u-net model

                      batch_size=32,
                      init_lr=1e-4,
                      num_epochs=1000,
                      evaluation_steps=10000,
                      patience=50,  # epochs before decreasing learning rate
                      curriculum_training=False,
                      curriculum_patience=10,
                      loss='mse',   # log, mse, mpe, sisdr, msesdr, bce, l1, smoothl1, cce
                      loss_pos_weight=5,
                      ratio_masks=True,  # use complementary ratio masks

                      # Visual conditioning settings
                      dir_checkpoint='./checkpoints',
                      save_cp=True,
                      save_cp_frequency=10,
                      visual_checkpoint='/pretrained/visual/resnet_0.844595.pt',
                      with_resnet=False,
                      with_lstm=False,
                      max_pooling=False,
                      n_visual_frames=1,
                      visual_finetune=False,

                      # Testing time parameters
                      output_dir='./eval',
                      norbert=False,

                      # Testing time parameters and resume training
                      model_checkpoint='00001/CP0200.pth',
                      load_model=True,
                      resume_training=False,

                      # Data load and processing settings
                      load_to_ram=True,
                      dataset='solos',   # 'solos' or 'urmp'
                      input_type='spec_compute',
                      spec_predict=False,

                      # preprocessing parameters
                      amp_to_db=True,
                      spec_scale=True,
                      add_noise=False,

                      expected_sr=11025,
                      stft_frame=1022,
                      mock_spec_size=(14, 2, 512, 256),
                      log_resample=True,
                      log_sample_n=256)


@ex.named_config
def baseline():
    logger.info("Training Spec-U-Net with Softmax")
    exp_config = dict(
        model='mhunet',
        num_decoders=1,
        input_type='spec_compute',
        ratio_masks=True,
        loss='mse',
        complementary=True
    )


@ex.named_config
def spec_multihead():
    logger.info("Training Multihead-Spec-U-Net")
    exp_config = dict(
        model='mhunet',
        num_decoders=13,
        input_type='spec_compute',
        ratio_masks=True,
        loss='mse',
        complementary=True
    )

@ex.named_config
def film_bottleneck():
    logger.info("Training Spec-U-Net with FiLM-bottleneck Conditioning with Labels")
    exp_config = dict(
        model='mhunet',
        num_decoders=1,
        input_type='spec_compute',
        conditioning='film_bn'
    )

@ex.named_config
def film_final_visual():
    logger.info("Training Spec-U-Net with FiLM-final Visual conditioning")
    exp_config = dict(
        model='mhunet',
        num_decoders=1,
        input_type='spec_compute',
        conditioning='film_final',

        n_visual_frames=1,
        with_resnet=True,
        max_pooling=True,
        visual_finetune=True
    )


@ex.named_config
def final_multiply_visual():
    logger.info("Training Spec-U-Net with Final-multiply Visual Conditioning")
    exp_config = dict(
        model='mhunet',
        num_decoders=1,
        input_type='spec_compute',
        conditioning='mult_final',

        n_visual_frames=1,
        with_resnet=True,
        max_pooling=True,
        visual_finetune=True
    )

@ex.named_config
def film_final_visual_multiframe():
    logger.info("Training Spec-U-Net with FiLM-Final Visual Conditioning on multiframe settings")
    exp_config = dict(
        model='mhunet',
        num_decoders=1,
        input_type='spec_compute',
        conditioning='mult_final',

        n_visual_frames=5,
        with_resnet=True,
        with_lstm=True,
        max_pooling=False,
        visual_finetune=False
    )
