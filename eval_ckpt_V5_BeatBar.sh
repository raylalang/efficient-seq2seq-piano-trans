# nohup bash ./eval_ckpt_V5_BeatBar.sh > eval_ckpt_V5_BeatBar.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
CKPT_V5=runs/Transformer-T5/250901-175648_BeatBarContext/cpt/latest.ckpt

python evaluate.py --config-name=experiment_T5_V5_BeatBarContext \
  hydra.job.chdir=false \
  accelerator=gpu devices=1 \
  model.checkpoint_path=$CKPT_V5 \
  training.log_dir="/home/ray/Research/efficient-seq2seq-piano-trans/runs/Transformer-T5/250901-175648_BeatBarContext/eval/BeatBarContext_100000_$(date +%y%m%d-%H%M%S)"