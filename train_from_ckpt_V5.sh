# nohup bash ./train_V5.sh > train_V5.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4,5,6
config=experiment_T5_V5_BeatBarContext

python train.py --config-name=$config\
  accelerator=gpu\
  devices=3\
  training.training_steps=33333\
  training.online_testing=false