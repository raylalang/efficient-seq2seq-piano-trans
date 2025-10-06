# nohup bash ./train_V0.sh > train_V0.log 2>&1 &

export CUDA_VISIBLE_DEVICES=6,7
config=experiment_T5_V0

python train.py --config-name=$config\
  accelerator=gpu\
  devices=2\
  +model.context_mode=null\
  +model.context_fuse=null\
  training.training_steps=100000\
  training.online_testing=false\
  training.notes="NoContext"

