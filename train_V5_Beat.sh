# nohup bash ./train_V5_Beat.sh > train_V5_Beat.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
config=experiment_T5_V5_BeatBarContext

python train.py --config-name=$config\
  accelerator=gpu\
  devices=1\
  model.use_context=true\
  model.context_mode=beat\
  model.context_fuse=concat\
  training.training_steps=100000\
  training.online_testing=false\
  training.notes="BeatContext"