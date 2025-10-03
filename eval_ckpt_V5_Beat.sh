# nohup bash ./eval_ckpt_V5_Beat.sh > eval_ckpt_V5_Beat.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
CKPT_V5=runs/Transformer-T5/250901-175659_BeatContext/cpt/latest.ckpt

python evaluate.py --config-name=experiment_T5_V5_BeatBarContext \
  hydra.job.chdir=false \
  accelerator=gpu devices=1 \
  model.checkpoint_path=$CKPT_V5 \
  training.log_dir="/home/ray/Research/efficient-seq2seq-piano-trans/runs/Transformer-T5/250901-175659_BeatContext/eval/BeatContext_100000_$(date +%y%m%d-%H%M%S)"