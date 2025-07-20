# Sequence-to-Sequence Piano Transcription

This is a repository for Transformer based sequence-to-sequence piano transcription.

## Environment Requirement

- PyTorch>=2.2
- Python>=3.10

Install PyTorch
```bash
# CUDA 12.4
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.6
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

Install FlashAttention
```bash
pip install flash-attn==2.7.4.post1
```

- To use [FlashAttention](https://github.com/Dao-AILab/flash-attention), PyTorch>=2.2 is required. 

Install requirements.
```
$ pip install -r requirements.txt
```
## Dataset Preparation

Download MAESTRO v3.0.0 dataset, and unzip to ./dataset/maestro-v3.0.0
```bash
mkdir dataset
cd dataset
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip
cd ../
```

or use soft symbolic link:

```bash
ln -s path/to/maestro-v3.0.0 ./dataset/maestro-v3.0.0
```


Convert audios into h5py file and save to "dataset/maestro-v3.0.0/audio.h5".
```bash
python data/audio_to_h5_MAESTRO.py
```


Convert midi files to tsv and save to "dataset/maestro-v3.0.0/cache/".
```bash
python data/midi_2_performance_tsv_MAESTRO.py
```

### Training



```bash
# use the default training config in config/main_config.yaml
python train.py

# override
config=experiment_T5_V4_HierarchyPool
python train.py --config-name=$config

# customized config
python train.py --config-name=$config\
  training.batch=16\
  training.learning_rate=1e-4\
  training.training_steps=200000\
  model.checkpoint_path="path/to/checkpoint.ckpt"\
  accelerator=gpu\
  devices=[0,1]\

```

- Config files is managed with [hydra](http://hydra.cc/).


## Evaluation

```bash
checkpoint_path="path/to/checkpoint.ckpt"
echo $checkpoint_path
python evaluate.py \
  model.checkpoint_path="'$checkpoint_path'" \
  accelerator=gpu \
  devices=[0,1]


```


## Inference

Download the checkpoint from release page to ./checkpoints.

```bash
checkpoint_path=checkpoints/T5_V4_steps_200000.ckpt
python inference.py audio_path="'audio/Franz-Liszt_Liebestraum.mp3'" midi_path="outputs/Franz-Liszt_Liebestraum.mid"

```
## Acknowledgements

- rlax59us' [MT3-pytorch](https://github.com/rlax59us/MT3-pytorch)
- Official [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)






