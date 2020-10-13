## Supplementary code for experiments listed in "Conditioned Source Separation for Musical Instrument Performances" model

## Requirements
In order to use this code, clone the repo and install the requirements listed in [Pipfile](Pipfile)

## Datasets

This repo requires two datasets which should be downloaded and pre-processed separately

* Solos: A Dataset for Audio-Visual Music Source Separation and Localization [Solos](https://www.juanmontesinos.com/Solos/)
* University of Rochester Multi-modal Music Performance (URMP) Dataset [URMP](http://www2.ece.rochester.edu/projects/air/projects/URMP.html)

## Training new models

For training new models, please, check the desired configuration in [config.py](config.py) file.
You can run a new training job for a model without conditioning or with label-conditioning using:

```bash
. ./train_uspec.sh
# Or specify all parameters explicitly
CUDA_VISIBLE_DEVICES=0 python train_uspec.py with exp_config.dataset_dir='path/to/dataset/here'
```
For training a model with visual conditioning, please, use:

```bash
. ./train_mm.sh
# Or specify all parameters explicitly
CUDA_VISIBLE_DEVICES=0 python train_mm.py with exp_config.dataset_dir='path/to/dataset/here'
```

## Evaluating a model

The running configuration for a model is saved in `experiments_meta`
directory under a specific experiments id.

While model is training, you will find logs and tensorboard
stats in `runs` directory under a specific experiments id.

The checkpoints are saved in `runs` directory under a specific experiments id.

In order to evaluate a model, you have to create an evaluation folder first and run:

```bash
mkdir eval/exp_id
mkdir eval/exp_id/dataset

CUDA_VISIBLE_DEVICES=0 python test_uspec.py with experiments_meta/exp_id/config.json exp_config.model_checkpoint='exp_id/CP0990.pth' exp_config.dataset_dir='path/to/dataset/here'
# Or change parameters in the evaluation script and run
. ./train_uspec.sh
```

## Pretrained models

We are providing checkpoints and training configurations for several models
with their corresponding IDs (as listed in the paper).

* Exp. 1 - ([Baseline](https://drive.google.com/drive/folders/1lsrF1Rrg2HvGDGaK_sf8jc27JiM21FIq?usp=sharing))
* Exp. 6 - ([Linear-Scale STFT](https://drive.google.com/drive/folders/1tNXW26ucLrPpH5JL1qYPIAaP5yjdS2TC?usp=sharing))
* Exp. 8 - ([FiLM-bottleneck Label Conditioning](https://drive.google.com/drive/folders/1M_ZoOulzBnp2Llaj8PUT4i8ezqghB-hT?usp=sharing))
* Exp. 9 - ([FiLM-encoder Label Conditioning](https://drive.google.com/drive/folders/18dZsj2AeY4MIiHQ3VA6gXGlcdKHWRnPE?usp=sharing))
* Exp. 10 - ([FiLM-final Label Conditioning](https://drive.google.com/drive/folders/1-8isxVQU-hI15xAxMf2dr6FB8Tmx8zmL?usp=sharing))
* Exp. 11 - ([Label-multiply Conditioning](https://drive.google.com/drive/folders/1lRYhBWuiUjDDoeFA8vqrXU9Fy6wTvxzm?usp=sharing))
* Exp. 13 - ([FiLM-final LSTM Visual Conditioning](https://drive.google.com/drive/folders/1XxcLgXP5ibzyjnM1IReWbdZVgIfoKmE3?usp=sharing))
* Exp. 16 - ([Final-multiply Visual Conditioning](https://drive.google.com/drive/folders/18AwI1i8RWX1xXbseEGfwUU636fwcsHAR?usp=sharing))
* Exp. 18 - ([FiLM-final Visual Conditioning](https://drive.google.com/drive/folders/1ELpTDaal3M8Slu3uNM2VVs8-r99DgfSZ?usp=sharing))

The configuration for each model can be found [here](https://drive.google.com/drive/folders/1Hv9JwPuFIZ4SMlp0jc2OB3PKctcBCdUw?usp=sharing).