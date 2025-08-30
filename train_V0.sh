# nohup bash ./train_V0.sh > train_v0.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
config=experiment_T5_V0

python train.py --config-name=$config

