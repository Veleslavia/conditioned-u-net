#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python test_uspec.py with experiments_meta/1/config.json \
        exp_config.model_checkpoint='00001/CP0990.pth' \
        exp_config.output_dir='./eval/00001/urmp' \
        exp_config.dataset='urmp' \
        exp_config.dataset_dir='/Datasets/urmpv2/all'