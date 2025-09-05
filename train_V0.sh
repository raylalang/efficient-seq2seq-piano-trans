# nohup bash ./train_V0.sh > train_V0.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
config=experiment_T5_V0

python train.py --config-name=$config\
  accelerator=gpu\
  devices=1\
  model.context_mode=null\
  model.context_fuse=null\
  training.training_steps=100000\
  training.online_testing=false\
  training.notes="NoContext"

