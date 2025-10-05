# nohup bash ./train_V0_old.sh > train_V0_old.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3,4
config=experiment_T5_V0

python train_old.py --config-name=$config\
  accelerator=gpu\
  devices=2\
  +model.context_mode=null\
  +model.context_fuse=null\
  training.training_steps=100000\
  training.online_testing=false\
  training.notes="NoContext"

