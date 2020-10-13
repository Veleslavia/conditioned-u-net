#!/bin/bash

python3 test_mm.py with experiments_meta/1/config.json \
              exp_config.dataset_dir='/Datasets/urmpv2/all' \
              exp_config.dataset='urmp' \
              exp_config.model_checkpoint='00001/CP0990.pth' \
              exp_config.output_dir='./eval/00001/urmp_norbert' \
              exp_config.norbert=True