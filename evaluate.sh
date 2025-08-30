checkpoint_path="checkpoints/T5_V4_steps_200000.ckpt"
echo $checkpoint_path
python evaluate.py \
  model.checkpoint_path="'$checkpoint_path'" \
  accelerator=gpu \
  devices=[0,1,2,3]

