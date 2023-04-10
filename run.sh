#!/usr/bin/env bash

if [ $# -lt 1 ];then
    echo "please select [sft|rw|rl]"
    exit 0
fi

if [[ $1 = "sft" ]]; then
    echo "start training" $1
    python train_sft.py --model_name_or_path bigscience/bloomz-560m \
                    --data_path data/train.json \
                    --output_dir model/sft/
    exit 0
fi


if [[ $1 = "rw" ]]; then
    echo "start training" $1
    python train_reward.py --model_name_or_path bigscience/bloomz-560m \
                    --data_path data/train.json \
                    --output_dir model/sft/   \
                    --per_device_train_batch_size 4  \
                    --gradient_accumulation_steps 4  \
                    --num_train_epochs 3  \
                    --optim adafactor
    exit 0
fi


if [[ $1 = "rl" ]]; then
    echo "start training" $1
    python train_rl.py --model_name bigscience/bloomz-560m \
                    --logging_dir data/log/  \
                    --learning_rate 1.41e-5  \
                    --dataset_name data/reward_train.json  \
                    --seed 307  \
                    --reward_model_name model/reward_model  \
                    --output_dir model/rl_model
    exit 0
fi

echo "please select [sft|rw|rl]"
exit 0