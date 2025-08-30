import h5py
import sys
import pretty_midi
import os
work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.constants import *
import torchaudio
from torchaudio.transforms import Resample
import torch

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_audio_split_list(df, output_path):
    midi_paths = [x for x in df["midi_filename"]]
    audio_paths = [os.path.splitext(x)[0] + ".wav" for x in midi_paths]
    indexs = list(df.index)
    with open(output_path, "w") as f:
        for i, path in zip(indexs, audio_paths):
            f.write(str(i) + "\t" + path + "\n")


if __name__ == "__main__":

    root_dir='dataset/maestro-v3.0.0/'
    csv_path = root_dir + '/maestro-v3.0.0.csv'
    df = pd.read_csv(csv_path, header=0)


    for split in ["train", "validation", "test", "all"]:
        split_df = df[df["split"] == split]
        if split == "all":
            split_df = df
        save_audio_split_list(split_df, os.path.join(root_dir, f"subset_{split}.tsv"))



    midi_paths = [os.path.join(root_dir, x) for x in df["midi_filename"]]
    # audio_paths = [os.path.splitext(x)[0] + ".flac" for x in midi_paths]
    audio_paths = [os.path.splitext(x)[0] + ".wav" for x in midi_paths]
    indexs = list(df.index)
    with h5py.File(root_dir + f"/audio.h5", "w") as h5:
        for idx, path in tqdm(zip(indexs, audio_paths), total=len(audio_paths)):
            
            # audio, sr = librosa.load(path, sr=None)
            # torchaudio.set_audio_backend("soundfile")
            audio, sr = torchaudio.load(path)
            audio = audio.to(DEFAULT_DEVICE).mean(dim=0)
            if sr != DEFAULT_SAMPLE_RATE:
                print(" Resample from", sr, "to", DEFAULT_SAMPLE_RATE)
                # audio = librosa.resample(audio, orig_sr=sr, target_sr=DEFAULT_SAMPLE_RATE)
                # Use torchaudio for resampling
                resampler = Resample(orig_freq=sr, new_freq=DEFAULT_SAMPLE_RATE).to(device=DEFAULT_DEVICE)
                audio = resampler(torch.tensor(audio).to(DEFAULT_DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
            
            
            idx = str(idx)
            group = h5.create_group(idx)
            group["audio"] = audio
            group["path"] = path
            group["sample_rate"] = DEFAULT_SAMPLE_RATE
            group["duration"] = len(audio) / DEFAULT_SAMPLE_RATE
