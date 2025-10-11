# nohup bash ./train_V5_BeatBarAttnDouble.sh > train_V5_BeatBarAttnDouble.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4,5
config=experiment_T5_V5_BeatBarContext

python train.py --config-name=$config\
  accelerator=gpu\
  devices=2\
  model.use_context=true\
  model.context_mode=beat_bar\
  model.context_fuse=attn\
  training.training_steps=100000\
  training.online_testing=false\
  training.notes="BeatBarAttnDoubleContext"\
  data.n_frames=1024\
  data.max_token_length=2048