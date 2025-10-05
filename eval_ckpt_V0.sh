# nohup bash ./eval_ckpt_V0.sh > eval_ckpt_V0.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
CKPT_V0=/home/ray/Research/efficient-seq2seq-piano-trans/runs/Transformer-T5/251004-171944_NoContext/cpt/last.ckpt

python evaluate.py --config-name=experiment_T5_V0 \
  accelerator=gpu devices=1 \
  model.checkpoint_path=$CKPT_V0 \
  training.log_dir="/home/ray/Research/efficient-seq2seq-piano-trans/runs/Transformer-T5/251004-171944_NoContext/eval"