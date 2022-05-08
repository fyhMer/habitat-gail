#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export CUDA_VISIBLE_DEVICES=1

set -x
#python -u -m habitat_baselines.run \
#    --exp-config configs/gail_objectnav.yaml \
#    --run-type train \
#    TASK_CONFIG.DATASET.SPLIT 'train'

exp_config_path=$1 # configs/gail_objectnav.yaml

python3 -u -m run \
    --exp-config $exp_config_path \
    --run-type train
#    TASK_CONFIG.DATASET.SPLIT 'train_1x1_seed1'
    #    TASK_CONFIG.DATASET.SPLIT 'val' # small dataset for debug


#python3 run.py --exp-config configs/gail_objectnav.yaml --run-type train TASK_CONFIG.DATASET.SPLIT 'train'

# bash run_gail.sh configs/gail_objectnav_v2.yaml
# bash run_gail.sh configs/gail_objectnav_single_episode_v0_seed1.yaml
# bash run_gail.sh configs/gail_objectnav_single_episode_v1_seed7.yaml
