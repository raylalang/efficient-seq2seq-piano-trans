import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
from data.constants import *
import os
from train import MT3Trainer
import numpy as np
from tqdm import tqdm
import gc
import argparse


from omegaconf import OmegaConf
import hydra

from utils import sequence_processing
from pytorch_memlab import MemReporter
from torch.profiler import profile, record_function, ProfilerActivity

device = "cuda" if torch.cuda.is_available() else "cpu"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

trainer = None

@hydra.main(config_path="config", config_name="main_config", version_base = None)
def my_main(config: OmegaConf):
    torch.cuda.empty_cache()
    checkpoint_path = config.model.checkpoint_path
    
    ####################################################
    # Create model.
    global trainer
    if trainer is None:
        trainer = MT3Trainer(config)
        print("Loading model from checkpoint:", checkpoint_path)
        state_dict = torch.load(checkpoint_path)
        
        # Remove keys that are in the ignore list.
        for key in list(state_dict.keys()):
            if key in config.model.checkpoint_ignore_layres:
                print(f"Removing key {key} from state_dict")
                del state_dict[key]

        trainer.model.load_state_dict(state_dict, strict=True) # config.model.strict_checkpoint
        trainer.model = trainer.model.to(device).eval()

    audio_path = config.audio_path
    
    wav, sr = torchaudio.load(audio_path)
    # Merge channels if stereo.
    if wav.shape[0] > 1:
        wav = wav.to(device).mean(dim=0, keepdim=True)
    if sr != DEFAULT_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, DEFAULT_SAMPLE_RATE).to(device)(wav)

    clip_len = config.data.n_frames * config.data.hop_length
    clip_len_in_second = clip_len/DEFAULT_SAMPLE_RATE

    ## Input Features
    with torch.no_grad():
        # => [B, T, F]
        encoder_inputs = trainer.features_extracter.to(device)(wav[:, :-1]).transpose(-1, -2)
        encoder_inputs = encoder_inputs.detach()
        if trainer.config.data.amplitude_to_db:
            # assert self.config.data.features != "mel"
            if trainer.config.data.features != "mel":
                encoder_inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(encoder_inputs)
                
    if encoder_inputs.shape[1] % config.data.n_frames != 0:
        # pad to the next multiple of n_frames
        pad_len = config.data.n_frames - (encoder_inputs.shape[1] % config.data.n_frames)
        encoder_inputs = F.pad(encoder_inputs, (0, 0, 0, pad_len), value=0.0)   

    clip_num = encoder_inputs.shape[1] // config.data.n_frames
    # reshape to [B, T, F]
    encoder_inputs = encoder_inputs.view(clip_num, config.data.n_frames, -1)
                
    batch_size = config.training.batch_inference
    num_batches = np.ceil( encoder_inputs.shape[0] / batch_size ).astype(int)
    notes_list = []
    max_seq_len = config.data.max_token_length
    torch.cuda.synchronize("cuda")
    torch.cuda.empty_cache()
    gc.collect()

    for i in tqdm(range(num_batches), desc="Inference"):
        start = i * batch_size
        end = min((i + 1) * batch_size, encoder_inputs.shape[0])
        encoder_inputs_i = encoder_inputs[start:end].to(device)
        
        
        # torch.cuda.memory._record_memory_history(max_entries=100000)

        with torch.no_grad():
            decoder_output_tokens = trainer.model.generate(encoder_inputs_i, target_seq_length=max_seq_len, berak_on_eos=True)
        
        # try:
        #     # https://docs.pytorch.org/memory_viz
        #     # https://pytorch.org/blog/understanding-gpu-memory-1/
        #     torch.cuda.memory._dump_snapshot(f"gpu_memory_dump_snapshot.pickle")
        # except Exception as e:
        #     print(f"Failed to capture memory snapshot {e}")
        
        
        # get_output_seq_lens
        output_eos_tokens_flag = decoder_output_tokens
        output_eos_tokens_flag = (output_eos_tokens_flag == sm_tokenizer.EOS).int() * sm_tokenizer.EOS
        output_seq_lens = sequence_processing.get_sequence_lengths(output_eos_tokens_flag, sm_tokenizer.EOS)
        for b in range(encoder_inputs_i.shape[0]):
            output_seq_len = output_seq_lens[b].item()
            if output_seq_len == 0:
                continue
            output_seq_b = decoder_output_tokens[b, :output_seq_len]
            offsets_sec_b = (start + b) * clip_len_in_second
            events_list_b = sm_tokenizer.detokenize(output_seq_b.cpu().numpy(), offsets_sec=offsets_sec_b)
            notes_list_b = sm_tokenizer.midi_events_to_notes(events_list_b)
            notes_list.extend(notes_list_b)
    
    # save midi
    if hasattr(config, "midi_path"):
        midi_path = config.midi_path
    else:
        basename = os.path.basename(audio_path)
        midi_path = os.path.splitext(basename)[0] + ".output.mid"
        midi_path = os.path.join("outputs", midi_path)
    os.makedirs(os.path.dirname(midi_path), exist_ok=True)
    sm_tokenizer.save_midi(notes_list, midi_path)
    

if __name__ == "__main__":
    # assert len(sys.argv) == 3
    # audio_path = sys.argv[1]
    # midi_path = sys.argv[2]
    
    # argparse.ArgumentParser(description="Inference script for AMT-MIDI")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--audio_path", type=str, default=None, help="Path to the audio file")
    # # parser.add_argument("--midi_path", type=str, default=midi_path, help="Path to save the output MIDI file")
    # args = parser.parse_args()
    # audio_path = args.audio_path
    # midi_path = args.midi_path
    
    my_main()