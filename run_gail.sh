#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export CUDA_VISIBLE_DEVICES=1

set -x
#python -u -m habitat_baselines.run \
#    --exp-config configs/gail_objectnav.yaml \
#    --run-type train \
#    TASK_CONFIG.DATASET.SPLIT 'train'

python3 -u -m run \
    --exp-config configs/gail_objectnav.yaml \
    --run-type train \
    TASK_CONFIG.DATASET.SPLIT 'train'
    #    TASK_CONFIG.DATASET.SPLIT 'val' # small dataset for debug


#python3 run.py --exp-config configs/gail_objectnav.yaml --run-type train TASK_CONFIG.DATASET.SPLIT 'train'
