# nohup bash ./eval_ckpt_V5_BeatBarAttn_Double.sh > eval_ckpt_V5_BeatBarAttn_Double.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
# CKPT_V5=runs/Transformer-T5/251007-185321_BeatBarAttnDoubleContext/cpt/latest.ckpt
CKPT_V5=runs/Transformer-T5/251007-185321_BeatBarAttnDoubleContext/cpt/steps_50000.ckpt

python evaluate.py --config-name=experiment_T5_V5_BeatBarContext \
  hydra.job.chdir=false \
  accelerator=gpu devices=1 \
  model.checkpoint_path=$CKPT_V5 \
  data.n_frames=1024 \
  data.max_token_length=2048 \
  training.log_dir="/home/ray/Research/efficient-seq2seq-piano-trans/runs/Transformer-T5/251007-185321_BeatBarAttnDoubleContext/eval"