#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
#export MAGNUM_GPU_VALIDATION=ON

export CUDA_VISIBLE_DEVICES=1

set -x

exp_config_path=$1 # configs/gail_objectnav.yaml
ckpt_path=$2

python3 -u -m run \
    --exp-config $exp_config_path \
    --run-type eval \
    TASK_CONFIG.DATASET.DATA_PATH "habitat-challenge-data/data/datasets/objectnav_mp3d_v1/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
    EVAL_CKPT_PATH_DIR $ckpt_path \
    BASE_TASK_CONFIG_PATH "configs/objectnav_mp3d.yaml"

#bash run_gail_eval_hm3d_val_mini.sh configs/gail_objectnav_v7ir_7_1_1_mp3d_70k.yaml ~/Data/habitat/checkpoints/gail_v7ir_7_1_1_mp3d_70k/train/ckpt.2.512768.pth
