
from pathlib import Path
import os
import sys


work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)
import torch
from torch.utils.data import IterableDataset, Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as Functional
import librosa
import librosa.display
import music21
import torchaudio

import hashlib

import numpy as np 
from tqdm import tqdm
from copy import deepcopy
from omegaconf import OmegaConf
import hydra
import random
import h5py
import matplotlib.pyplot as plt
from glob import glob
import json
import pandas as pd

from data.constants import *

from data.symbolic_music_tokenizer import SymbolicMusicTokenizer, TokenOnset, ONSET_SEC_UP_SAMPLING
import data.symbolic_music_tokenizer as Tokenizer
from collections import defaultdict



# Hash string to int 0~9.
def stable_hash_to_digit(s: str) -> int:
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h, 16) % 10


class SingleWavDataset(Dataset):
    def __init__(self,config, dataset_dir:str, dataset_index:int, audio_idx, midi_path, audio_h5_path, random_clip = True) -> None:
        self.config = config
        self.dataset_dir = dataset_dir
        self.dataset_index = dataset_index
        self.dataset_sequence_type = config.data.dataset_sequence_types[dataset_index]
        self.dataset_name = os.path.split(dataset_dir)[-1]
        self.midi_path = midi_path
        self.data_loaded = False
        n_frames = config.data.n_frames
        max_token_length = config.data.max_token_length
        hop_length = config.data.hop_length


        frames_per_second=DEFAULT_SAMPLE_RATE / hop_length
        
        # load cache
        # root_dir = config.data.dataset_dir
        # token_cache_path = os.path.join(root_dir, "cache", os.path.relpath(midi_path, root_dir) + ".pth")
        # os.makedirs(os.path.dirname(token_cache_path), exist_ok=True)
        # if os.path.exists(token_cache_path):
        #     data_dict = torch.load(token_cache_path)
        #     tokens_ary = data_dict["tokens_ary"]
        #     seconds_ary = data_dict["seconds_ary"]
        # else:
        #     tokens_ary, seconds_ary = cp_tokenizer.tokenize_midi(midi_path)
        #     data_dict = {
        #         "tokens_ary": tokens_ary,
        #         "seconds_ary": seconds_ary
        #     }
        #     torch.save(data_dict, token_cache_path)
        
        # self.tokens_ary = torch.tensor(tokens_ary)
        # self.seconds_ary = torch.tensor(seconds_ary)
        
        # Lazy Loading
        self.tokens_ary = None
        self.seconds_ary = None
        self.data_loaded = False
        

        ###########
        # audio
        
        self.audio_idx = str(audio_idx)
        self.audio_h5 = h5py.File(audio_h5_path)
        duration = self.audio_h5[self.audio_idx]["duration"][()]
        assert duration <= 3000 # 1600 # <= 10 min
        # Check audio name and midi name.
        audio_path = self.audio_h5[self.audio_idx]["path"][()].decode()
        self.audio_name = os.path.split(audio_path)[1].replace(".flac", "")
        # assert midi_name == audio_name
        self.audio_size = self.audio_h5[self.audio_idx]["audio"].shape[0]
        
        self.total_frames = np.ceil( self.audio_size / hop_length ).astype(int)
        # self.total_frames = int(duration * frames_per_second)
        self.length = np.ceil( self.audio_size / (hop_length*n_frames) ).astype(int)
        # assert self.total_frames >= n_frames

        self.n_frames = n_frames
        self.hop_length = hop_length
        self.max_token_length = max_token_length

        self.random_clip = random_clip
        self.random = np.random.RandomState() #Don't use constand seed here!!! (Overfitting problems).

    def __len__(self):
        # if self.random_clip == True:
        #     return 1 # self.length
        # return 100 # 
        return self.length
        
    def __load_cache(self):
        # load cache
        
        if True:
            tsv_path = os.path.splitext(self.midi_path)[0] + ".midi-notes.tsv"
            dataframe_midi = pd.read_csv(tsv_path, sep="\t")
            dataframe_midi = sm_tokenizer.notes_to_midi_events(dataframe_midi)
        else:
            dataframe_midi = sm_tokenizer.midi_to_dataframe(self.midi_path)
            dataframe_midi = sm_tokenizer.notes_to_midi_events(dataframe_midi)

        self.dataframe_midi = dataframe_midi.sort_values(by="onset_sec").reset_index(drop=True)
            
        self.data_loaded = True
        
    def __getitem__(self, index, begin=None, end=None):
        if self.data_loaded == False:
            self.__load_cache()
            
        row = {}
        
        if (begin is None) and (end is None):
            if self.random_clip == False:
                begin = index * self.n_frames # begin frame idx
                end = begin + self.n_frames # end frame idx
            else:
                # begin = index * self.n_frames + self.random.randint(-self.n_frames//2, self.n_frames//2)
                # begin = np.clip(0, self.total_frames - self.n_frames//4)
                random = np.random.RandomState()
                begin = random.randint(0, self.total_frames - self.n_frames//4)
                
                end = begin + self.n_frames
        else:
            assert self.random_clip == False
            assert (begin is not None) and (end is not None)
            
        audio_begin = begin * self.hop_length
        audio_end = end * self.hop_length
        # assert end <= self.total_frames
        # assert audio_end <= self.audio_size

        # audio = torch.tensor(self.data_dict["wav"][audio_begin:audio_end])
        audio = torch.tensor(self.audio_h5[self.audio_idx]["audio"][audio_begin:audio_end])
        
        # convert time resolution from frame to OUTPUT_TIME_STEP_PER_SECOND
        second_per_frame = self.config.data.hop_length/DEFAULT_SAMPLE_RATE 
        begin_sec = begin * second_per_frame # * OUTPUT_TIME_STEP_PER_SECOND
        end_sec = end * second_per_frame # * OUTPUT_TIME_STEP_PER_SECOND
        

        pad_len = self.n_frames*self.hop_length - len(audio)
        if pad_len > 0:
            audio = Functional.pad(audio, [0, pad_len])
    
        # max_token_len = self.n_frames // self.downsample_rate
        max_token_len = self.config.data.max_token_length # * cp_tokenizer.compound_word_size
        # assert max_token_len < TOKEN_END
        decoder_targets = torch.ones([max_token_len], dtype=torch.long) * TOKEN_PAD
        decoder_targets_frame_index = torch.ones([max_token_len], dtype=torch.long) * (self.n_frames - 1) # max frame index
        encoder_decoder_mask = torch.zeros([max_token_len, self.n_frames], dtype=torch.long)
        seq_len = 0
        sel_tokens = torch.tensor([], dtype=torch.long)
        
        seconds = []
        
        # Add EOS
        cp_eos = torch.ones([sm_tokenizer.compound_word_size], dtype=sel_tokens.dtype) * sm_tokenizer.PAD
        # cp_eos[Tokenizer.TokenFamily.cp_index] = cp_tokenizer.EOS
        cp_eos[0] = sm_tokenizer.EOS #Tokenizer.TokenPitch.cp_index
        
        # MIDI Sequence
        df_midi = self.dataframe_midi[(self.dataframe_midi["onset_sec"] >= begin_sec) & (self.dataframe_midi["onset_sec"] <= end_sec)].copy(deep=True)
        if len(df_midi) > 0:
            # make a offset for onset_sec
            df_midi["onset_sec"] = df_midi["onset_sec"] - begin_sec # Fix offset when using clip for training.
            sel_cp_tokens, seconds = sm_tokenizer.tokenize_dataframe(df_midi, sequence_type="performance")
            sel_cp_tokens = torch.tensor(sel_cp_tokens, dtype=torch.long)
            sel_tokens = sel_cp_tokens.flatten()
            
        # Eos
        sel_tokens = torch.concat([sel_tokens, cp_eos], dim=0)
        seconds.extend([second_per_frame * self.n_frames] * sm_tokenizer.compound_word_size) # Add EOS seconds
        
        # Limit the token len to max_len
        if sel_tokens.size()[0] > max_token_len:
            print("Decoder target length(%d) > max(%d)."%(sel_tokens.size()[0], max_token_len))
            sel_tokens = sel_tokens[:max_token_len]
            
        
        seq_len = sel_tokens.size()[0]
        
        decoder_targets[:seq_len] = sel_tokens
        
        for i, sec in enumerate(seconds[:seq_len-1]):
            frame_index = int(  np.round(sec / second_per_frame ) )
            frame_index = np.clip(frame_index, 0, self.n_frames - 1) # Clip to valid frame index
            decoder_targets_frame_index[i] = frame_index
        
        # Encoder Decoder Mask
        seconds = seconds[:seq_len-1]
        if hasattr(self.config.model, "encoder_decoder_slide_window_size") and self.config.model.encoder_decoder_slide_window_size > 0:
            for i, sec in enumerate(seconds):
                if i % 3 == 0:
                    encoder_decoder_mask[i, :] = 1 # Set the mask to 1 for onset tokens
                    assert Tokenizer.TokenOnset.is_instance(sel_tokens[i]) or sel_tokens[i] == sm_tokenizer.EOS or sel_tokens[i] == sm_tokenizer.PAD, "Onset token expected, but got %s"%(sel_tokens[i])
                    continue
                frame_index = int(  np.round(sec / second_per_frame ) )
                frame_index = np.clip(frame_index, 0, self.n_frames - 1) # Clip to valid frame index
                # decoder_targets_frame_index[i] = frame_index
                # begin_index = max(0, frame_index - int(self.config.model.encoder_decoder_slide_window_size//2))
                begin_index = frame_index - (frame_index % self.config.model.encoder_decoder_slide_window_size)
                end_index = begin_index + self.config.model.encoder_decoder_slide_window_size
                encoder_decoder_mask[i, begin_index:end_index] = 1
        else:
            encoder_decoder_mask[:,:] = 1
            
    
        # remove invalid vocab
        invalid_vocab_mask = decoder_targets >= sm_tokenizer.vocab_size
        if invalid_vocab_mask.int().sum() > 0:
            print("decoder_targets >= VOCAB_SIZE")
            decoder_targets[invalid_vocab_mask] = sm_tokenizer.PAD
            
        decoder_targets_mask = (decoder_targets != sm_tokenizer.PAD).long()

        
        audio_name = [ord(x) for x in self.audio_name][:256]
        audio_name = torch.tensor(audio_name, dtype=torch.long)
        audio_name = torch.nn.functional.pad(audio_name, pad=[0, 256 - audio_name.size()[0]], value=ord(" "))
        
        midi_path = [ord(x) for x in self.midi_path][:1024]
        midi_path = torch.tensor(midi_path, dtype=torch.long)
        midi_path = torch.nn.functional.pad(midi_path, pad=[0, 1024 - midi_path.size()[0]], value=ord(" "))
        
        dataset_name = [ord(x) for x in self.dataset_name][:32]
        dataset_name = torch.tensor(dataset_name, dtype=torch.long)
        dataset_name = torch.nn.functional.pad(dataset_name, pad=[0, 32 - dataset_name.size()[0]], value=ord(" "))
        
        row.update({
            "inputs": audio,
            "decoder_targets": decoder_targets,
            "decoder_targets_mask": decoder_targets_mask,
            "decoder_targets_len": torch.tensor(seq_len, dtype=torch.long),
            "decoder_targets_frame_index": decoder_targets_frame_index,
            
            "encoder_decoder_mask": encoder_decoder_mask[None],
            
            "audio_ids": torch.tensor(int(self.audio_idx), dtype=torch.long),
            "audio_name": audio_name,
            "frame_offsets":torch.tensor(begin, dtype=torch.long),
            "midi_path": midi_path,
            "dataset_name": dataset_name,
            "dataset_index": torch.tensor(self.dataset_index, dtype=torch.long),
        })

        return  row
    
    
def get_subset(subset_list_path):
    paths = []
    with open(subset_list_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            assert len(line) == 2
            idx, path = line
            paths.append([idx, path])
        
    return paths

class Audio2Midi_Dataset(Dataset):
    def __init__(self, config, dataset_dir:str, dataset_index:int, subset='train', dataset_size = -1, random_clip = True) -> None:
        root_dir = dataset_dir
        
        subset_train = get_subset(root_dir + '/subset_train.tsv')
        subset_validation = get_subset(root_dir + '/subset_validation.tsv')
        subset_test = get_subset(root_dir + '/subset_test.tsv')
        
        if subset == "train":
            subset_data = subset_train
        elif subset == "validation":
            subset_data = subset_validation
        elif subset == "test":
            subset_data = subset_test
        elif subset == "all":
            subset_data = subset_train + subset_validation + subset_test
        else:
            raise "Unknown subset: " + subset
        
        
        self.audio_relative_paths = []
        self.mid_relative_paths = []
        self.dataset_dir = root_dir
        
        idx_list = []
        mid_paths = []
        
        for idx, path in subset_data:
            
            if( len(idx_list) >= 5) and subset != "test": # set mini set for debug
                # break
                pass
            
            mid_path = os.path.join(root_dir, config.data.cache_dir_name, os.path.splitext(path)[0].replace("/", ">") + ".mid")
            
            if os.path.exists(mid_path):
                idx_list.append(idx)
                mid_paths.append(mid_path)
                
                self.audio_relative_paths.append(path)
                self.mid_relative_paths.append(mid_path)
                
        audio_h5_path = root_dir + "/audio.h5"
        
        if subset == "test":
            pass
            if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
                pass
                idx_list = idx_list[::10]
                mid_paths = mid_paths[::10]   
            else:
                pass
                idx_list = idx_list[:] # Test set, only use 1/5 of the data.
                mid_paths = mid_paths[:]
        else:
            if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
                idx_list = idx_list[::5] # Test set, only use 1/5 of the data.
                mid_paths = mid_paths[::5]
                
        

        self.dataset_list = [ SingleWavDataset(config, dataset_dir, dataset_index, i, path, audio_h5_path, random_clip=random_clip) 
            for i, path in tqdm(zip(idx_list, mid_paths), total=len(mid_paths))
        ]
        self.datasets = ConcatDataset(self.dataset_list)
        if dataset_size > 0:
            self.length = dataset_size
        else:
            self.length = len(self.datasets)
        # self.random = np.random.RandomState(42)
    def __len__(self):
        # length = np.sum([dataset.length for dataset in self.dataset_list])
        return self.length
        # return 100 * len(self.dataset_list)
    def __getitem__(self, index):
        return self.datasets[index]
    

    
    
    
def visualize(data_dict, output_fig_path):
    sr = DEFAULT_SAMPLE_RATE
    y = data_dict["inputs"].cpu().numpy()
    tokens = data_dict["decoder_targets"]
    cp_tokens = tokens.reshape([-1, sm_tokenizer.compound_word_size])
    cp_tokens = cp_tokens.cpu().numpy()
    cp_n = cp_tokens.shape[0]
    
    C = librosa.cqt(y, sr=sr)
    C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    dur_mins = len(y)/sr/60
    fig_width = 12 * min(dur_mins, 5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Draw Spectrogram
    librosa.display.specshow(C_dB, sr=sr, x_axis="time", y_axis="cqt_note", ax=ax)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    
    # Draw  Notes
    notes = []
    def get_val(cp_word, TokenType):
        return cp_word[TokenType.cp_index] - TokenType.begin_index

    beat_times = []
    downbeat_times = []
    tempo = 120 # 
    for idx in range(cp_n):
        cp_word = cp_tokens[idx]
        family = cp_word[0]
        start = get_val(cp_word, Tokenizer.TokenOnset) / Tokenizer.ONSET_SEC_UP_SAMPLING
        if family == Tokenizer.FamilyType.NOTE.value:
            pitch = get_val(cp_word, Tokenizer.TokenPitch)
            
            sec_per_quarter = 60 / tempo
            
            staff = get_val(cp_word, Tokenizer.TokenStaff)
            if Tokenizer.TokenBeat in sm_tokenizer.token_tpye_set:
                beat = get_val(cp_word, Tokenizer.TokenBeat)
                if beat % Tokenizer.BEAT_UP_SAMPLING == 0:
                    beat_times.append(start)
            
            end = start + get_val(cp_word, Tokenizer.TokenDur) / Tokenizer.DUR_TATUM_UP_SAMPLING * sec_per_quarter
            
            notes.append({
                "pitch": librosa.midi_to_note(pitch),
                "freq": librosa.midi_to_hz(pitch),
                "start": start,
                "end": end,
                "staff":staff
            })
        elif family == Tokenizer.FamilyType.MEASURE.value:
            if Tokenizer.TokenTempo in sm_tokenizer.token_tpye_set:
                tempo = get_val(cp_word, Tokenizer.TokenTempo) * Tokenizer.TEMPO_DOWN_SAMPLING
            downbeat_times.append(start)
        elif family == Tokenizer.FamilyType.EOS.value:
            break
        
    for n in notes:
        if n["staff"] == 0:
            color = "lime"
        else:
            color = "cyan"
        ax.hlines(
            y=n["freq"], xmin=n["start"], xmax=n["end"],
            colors=color, linewidth=2, label=n["pitch"]
        )
    
    # 避免重复 legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    # ax.legend(unique.values(), unique.keys(), loc="upper right")
    ax.legend(list(unique.values())[:2], ["Staff 1", "Staff 2"], loc="upper right")
    
    
    
    # Draw Beat & Down Beat
    height = (ylim[1] - ylim[0])/4
    ymin = ylim[0]
    for t in beat_times:
        ax.vlines(t, ymin=ymin, ymax=ymin+height * 0.25, color='cyan', linestyle='dashed', linewidth=1)

    for t in downbeat_times:
        db_color = "magenta"
        db_color = "white"
        ax.vlines(t, ymin=ymin, ymax=ymin+height * 0.5, color=db_color, linestyle='solid', linewidth=1.5)
    
    
    
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    fig.savefig(output_fig_path, format="pdf", dpi=300)
    

    
## Test
@hydra.main(config_path="../config", config_name="main_config", version_base = None)
def my_main(config: OmegaConf):
    
    datasets = []
    for dataset_dir in config.data.dataset_dirs:
        if not os.path.exists(dataset_dir):
            raise ValueError("Dataset dir %s not exists!"%dataset_dir)
        datasets.append( Audio2Midi_Dataset(config, dataset_dir, dataset_index=0, subset="test", random_clip=True) )
        print(dataset_dir, " Len:", len(datasets[-1]))
    dataset = ConcatDataset(datasets)
    
    prev_audio_name = None
    ratios = defaultdict(list)
    for i, data in enumerate(tqdm(dataset)):
        continue
        
        
        if i % 5 != 0:
            pass
            # continue
        # print(i, data)
        # score_str = token_to_score(data["decoder_targets_onset"].cpu().numpy())
        # sy_score = dur_token_to_midi(data["decoder_targets_onset"].cpu().numpy())
        audio_name = data["audio_name"]
        frame_offsets_ratio = data["frame_offsets_ratio"]
        
        audio_name = "".join([chr(x) for x in audio_name.numpy()]).strip()
        onset_pitch_dict = defaultdict(list)
        targets = data["decoder_targets"].cpu().numpy()
        for i in range(0, data["decoder_targets"].size()[0], 2):
            onset = targets[i]
            pitch = targets[i+1]
            onset_pitch_dict[onset].append(pitch)
        if prev_audio_name == audio_name:
            continue
        if i % 1000 == 0:
            # print("Audio name:", audio_name)
            # print("Frame offset ratio:", frame_offsets_ratio)
            # print("Ratio mean:", np.mean(ratios))
            # ratios = []
            print(audio_name)
        
        continue
        
        audio_path = "preview/%05d_%s.wav"%(i, audio_name)
        audio = data["inputs"].cpu().numpy()
        torchaudio.save(audio_path, torch.tensor(audio)[None,], DEFAULT_SAMPLE_RATE)
        
        
        if i >  50000:
            break
        prev_audio_name = audio_name
        

if __name__ == "__main__":
    my_main()