"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as Functional

from model.Encoder import Encoder
from model.Decoder import Decoder, CompoundDecoder
from model.Layers import *
from model.Mask import *
from model.HPPNet import HPPNet
from data.constants import *

from torch.nn.utils.rnn import pad_sequence
from config.utils import DictToObject

from tqdm import tqdm
from utils.log_memory_usage import profile_cuda_memory

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        config =DictToObject(dict(config))
        config.dtype = eval(config.dtype)
        # config.dtype = torch.float32 # 
        self.config = config
        # select encoder
        if config.encoder_name == "TransformerEncoder":
            self.encoder = Encoder(config)
        elif config.encoder_name == "CNNEncoder":
            self.encoder = CNNEncoder(config)
        elif config.encoder_name == "HPPNetEncoder":
            self.encoder = HPPNet(config)
        elif config.encoder_name == "hFT_Transformer_Encoder":
            self.encoder = hFT_Transformer_Encoder(config)
        else:
            raise "Unknown encoder: " + config.encoder_name
        # froze encoder
        if config.froze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        

        # select decoder
        # self.decoder = Decoder(config=config)
        sub_token_names = [ t.get_class_name() for t in sm_tokenizer.token_type_list]
        
        if config.decoder_name == "TransformerDecoder":
            self.decoder = Decoder(config=config)
        elif config.decoder_name == "CompoundTransformerDecoder":
            self.decoder = CompoundDecoder(config=config, sub_token_names=sub_token_names)

        self.pad_token = TOKEN_PAD
        self.eos_token = TOKEN_END
        
    def encode(
            self, 
            encoder_input_tokens, 
            encoder_segment_ids=None, 
            enable_dropout=True
            ):
        assert encoder_input_tokens.ndim == 3  # (batch, length, depth)

        encoder_mask = make_attention_mask(
            torch.ones(encoder_input_tokens.shape[:-1]),
            torch.ones(encoder_input_tokens.shape[:-1]),
            dtype=self.config.dtype
        )

        if encoder_segment_ids is not None:
            encoder_mask = combine_masks(
                encoder_mask,
                make_attention_mask(
                    encoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype
                )
            )
        
        encoder_mask = encoder_mask.to(encoder_input_tokens.device)
            
        if self.config.froze_encoder:
            with torch.no_grad():
                encoder_outputs = self.encoder(encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)
                encoder_outputs = encoder_outputs.detach()
        else:
            encoder_outputs = self.encoder(encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)
        return encoder_outputs
    
    def decode(
            self, 
            encoded, 
            encoder_input_tokens, 
            decoder_input_tokens, 
            decoder_target_tokens, 
            encoder_segment_ids=None, 
            decoder_segment_ids=None, 
            decoder_positions=None, 
            enable_dropout=True, 
            decode=False, #decode: Whether to prepare and use an autoregressive cache
            max_decode_length=None,
            use_preframe_tokens=True,
            decoder_targets_frame_index=None,
            encoder_decoder_mask=None
            ):
        
        encoder_decoder_mask_0 = encoder_decoder_mask
            
        # We use pooling to reduce decoder sequence length.
        # decoder_target_tokens here just use for generating decoder_mask and attention_mask.
        B, T = decoder_target_tokens.size()
        decoder_target_tokens = decoder_target_tokens #[:, :T//SEQUENCE_POOLING_SIZE]


        if decode:
            decoder_mask = None
            encoder_decoder_mask = make_attention_mask(
                torch.ones_like(decoder_target_tokens).to(encoded.device),
                torch.ones(encoder_input_tokens.shape[:-1]).to(encoded.device),
                dtype=self.config.dtype
            )
        else:
            decoder_mask = make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens,
                dtype=self.config.dtype,
                decoder_segment_ids=decoder_segment_ids
            )
            encoder_decoder_mask = make_attention_mask(
                decoder_target_tokens > 0,
                torch.ones(encoder_input_tokens.shape[:-1]).to(encoded.device),
                dtype=self.config.dtype
            )

        if encoder_segment_ids is not None:
            if decode:
                raise ValueError('During decoding, packing should not be used but `encoder_segment_ids` was passed to `Transformer.decode`.')

            encoder_decoder_mask = combine_masks(
                encoder_decoder_mask,
                make_attention_mask(
                    decoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype
                )
            )
            
        
        # If the encoder_decoder_mask is provided, we use the default mask.
        if encoder_decoder_mask_0 is not None:
            if hasattr(self.config, "encoder_decoder_slide_window_size") and self.config.encoder_decoder_slide_window_size > 0:
                encoder_decoder_mask = encoder_decoder_mask_0
                
                
        # Hierarchical pooling
        encoded_pooling_dict = None
        if hasattr(self.config, "cross_attention_hierarchy_pooling") and self.config.cross_attention_hierarchy_pooling:
            encoded_pooling_dict = {}
            for pooling in set(self.config.pooling_sizes):
                encoded_i = encoded
                if pooling > 1:
                    encoded_i = encoded_i.reshape(encoded_i.size(0), encoded_i.size(1)//pooling, pooling, encoded_i.size(2)).mean(dim=2)  # [batch, 1, encoded_length, emb_dim]
                encoded_pooling_dict[pooling] = encoded_i  # Save the encoded_i for later use
            encoded = None

        decoder_output_dict = self.decoder(
            encoded,
            encoded_pooling_dict=encoded_pooling_dict,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            decoder_targets_frame_index=decoder_targets_frame_index,
            )
        return decoder_output_dict
    
    def _shift_right(self, input_ids, shift_step=1):
        BOS = TOKEN_START

        shifted_input_ids = torch.zeros_like(input_ids)
        if shift_step is None:
            shift_step = SEQUENCE_POOLING_SIZE
        shifted_input_ids[..., shift_step:] = input_ids[..., :-shift_step].clone()
        shifted_input_ids[..., :shift_step] = BOS

        return shifted_input_ids
    
    def forward(self, 
                encoder_input_tokens=None, 
                decoder_target_tokens=None, 
                decoder_input_tokens=None, 
                encoder_segment_ids=None, 
                decoder_segment_ids=None, 
                encoder_positions=None, 
                decoder_positions=None, 
                enable_dropout=True, 
                decode=False,
                dur_inputs = None,
                dur_targets = None,
                decoder_targets_frame_index=None,
                encoder_decoder_mask=None
    ):
        res_dict = {}
            
        if decoder_input_tokens == None:
            decoder_input_tokens = self._shift_right(decoder_target_tokens, shift_step=1)
            
        encoder_outputs = self.encode(encoder_input_tokens, encoder_segment_ids=encoder_segment_ids, enable_dropout=enable_dropout)
        
        decoder_output_dict = self.decode(encoder_outputs, encoder_outputs, decoder_input_tokens, decoder_target_tokens, encoder_segment_ids=encoder_segment_ids, decoder_segment_ids=decoder_segment_ids, decoder_positions=decoder_positions, enable_dropout=enable_dropout, decode=decode, 
            decoder_targets_frame_index=decoder_targets_frame_index,
            encoder_decoder_mask=encoder_decoder_mask)


        res_dict.update(decoder_output_dict)

        return res_dict

    def generate_0(self, primer=None, target_seq_length=1024):
        num_primer = len(primer)
        len_primer = len(primer[0])
        # -> [num_frame x vec_size]
        gen_tokens = torch.LongTensor([self.pad_token for i in range(target_seq_length-len_primer)]).expand(num_primer, target_seq_length-len_primer)
        gen_tokens = torch.concat((primer.type(torch.long), gen_tokens.device(primer.device)), dim=-1)
        

        i = num_primer
        while (i < target_seq_length):
            logits, _ = self.forward(gen_tokens[..., :i], decode=True)
            probs = self.softmax(logits)[..., :self.eos_token]
            token_probs = probs[:, i - 1, :]

            next_token = torch.argmax(token_probs)
            gen_tokens[:, i] = next_token

            if next_token == self.eos_token:
                break
            i += 1

        return gen_tokens[:, :i]
    
    # @profile_cuda_memory
    def generate(self, encoder_inputs, target_seq_length = 1024, berak_on_eos=False, global_rank=0):
        self.decoder.initialize_decoder_cache()
        
        batch_size, T, n_mel = encoder_inputs.size()
        gen_tokens = torch.ones([batch_size, target_seq_length]).to(encoder_inputs) * TOKEN_PAD
        eos_flags = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)
        curr_frame_index = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)
        max_num_tokens = 0
        encoded = self.encode(encoder_inputs, enable_dropout=False)
        decoder_mask = torch.tril(torch.ones((target_seq_length, target_seq_length), dtype=self.config.dtype), diagonal=0).to(encoder_inputs.device)
        decoder_mask = decoder_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.config.num_heads, -1, -1)  # [batch_size, 1, target_seq_length, target_seq_length]
        encoder_decoder_mask = torch.ones((batch_size, self.config.num_heads, target_seq_length, T), dtype=self.config.dtype).to(encoder_inputs.device)  # [batch_size, 1, target_seq_length, T]
        
        pred_step = 1
        if hasattr(self.config, "predict_onset_by_AttnMap") and self.config.predict_onset_by_AttnMap:
            pred_step = 2
            
        use_kv_cache = True
        curr_token = torch.ones([batch_size, pred_step]).to(encoder_inputs) * TOKEN_START  # [batch_size, pred_step]
        for i in tqdm(range(0, target_seq_length, pred_step), desc="Generating tokens (rank %d)" % global_rank):
            encoded_i = encoded[:, :T, :]  # [batch_size, T, n_mel]
            
            if use_kv_cache:
                decoder_input_tokens_i = curr_token # 
                encoder_decoder_mask_i = encoder_decoder_mask[:, :, i:i+pred_step, :T]
                decoder_mask_i = decoder_mask[:, :, i:i+pred_step, :i+pred_step]
                decoder_positions_i = torch.arange(i, i+pred_step, device=encoder_inputs.device).unsqueeze(0).expand([batch_size, -1])  # [1, 1, pred_step]
            else:
                decoder_input_tokens_i = self._shift_right(gen_tokens[:, :i+pred_step], shift_step=1)  # [batch_size, i+pred_step]
                encoder_decoder_mask_i = encoder_decoder_mask[:, :, :i+pred_step, :T]  # [batch_size, 1, i+pred_step, T]
                decoder_mask_i = decoder_mask[:, :, :i+pred_step, :i+pred_step]
                decoder_positions_i = None
            

            # Slide window for encoder-decoder attention
            if i % 3 != 0 and hasattr(self.config, "encoder_decoder_slide_window_size") and self.config.encoder_decoder_slide_window_size > 0:
                enc_len, emb_dim = encoded.size(1), encoded.size(2)
                encoded_fold = encoded.view(batch_size, -1, self.config.encoder_decoder_slide_window_size, emb_dim)  # [batch_size, num_windows, window_size, emb_dim]
                # => [batch_size, 1]
                curr_window_index = torch.floor(curr_frame_index // self.config.encoder_decoder_slide_window_size).int()#.unsqueeze(0)  # [batch_size, 1]
                batch_indices = torch.arange(batch_size, device=encoded.device) #.unsqueeze(1)  # [batch_size, 1]
                encoded_i = encoded_fold[batch_indices, curr_window_index, :, :]  # [batch_size, 1, window_size, emb_dim]
                # print(encoded_i.size())
                encoded_i = encoded_i.view(batch_size, self.config.encoder_decoder_slide_window_size, emb_dim)  # [batch_size, window_size, emb_dim]
                    
                # encoder_decoder_mask_i = torch.stack(encoder_decoder_mask_i_list, dim=0)
                encoder_decoder_mask_i = encoder_decoder_mask_i[:, :, :, :self.config.encoder_decoder_slide_window_size]  # [batch_size, 1, i+pred_step, T]
            
            
            # Hierarchical pooling
            encoded_pooling_dict = None
            if hasattr(self.config, "cross_attention_hierarchy_pooling") and self.config.cross_attention_hierarchy_pooling:
                encoded_pooling_dict = {}
                for pooling in set(self.config.pooling_sizes):
                    encoded_pool = encoded_i
                    if pooling > 1:
                        encoded_pool = encoded_pool.reshape(batch_size, encoded_pool.size(1)//pooling, pooling, encoded_pool.size(2)).mean(dim=2)  # [batch, 1, encoded_length, emb_dim]
                    encoded_pooling_dict[pooling] = encoded_pool  # Save the encoded_i for later use
                encoded_i = None
            
            decoder_output_dict = self.decoder(
                encoded_i,
                encoded_pooling_dict=encoded_pooling_dict,
                decoder_input_tokens=decoder_input_tokens_i,
                decoder_positions=decoder_positions_i,
                decoder_mask=decoder_mask_i,
                encoder_decoder_mask=encoder_decoder_mask_i,
                deterministic=True,
                decode=use_kv_cache,
                )
            
            logits = decoder_output_dict["decoder_outputs"]
            probs = torch.softmax(logits[:, -pred_step:, :], dim=-1)[..., :self.pad_token]
            
            # Check if the output sequence is in the expected order: onset, pitch/noteOff, velocity
            if hasattr(self.config, "hybrid_global_local_cross_attn") and self.config.hybrid_global_local_cross_attn:
                tokens_ori = torch.argmax(probs, dim=-1)
                eos_flags_int = eos_flags.int()
                for b in range(batch_size):
                    token_i = tokens_ori[b, 0].item()
                    if eos_flags_int[b] == 1:
                        continue
                    if Tokenizer.TokenOnset.is_instance(token_i) or token_i == self.eos_token:
                        token_type_i = Tokenizer.TokenOnset
                    elif Tokenizer.TokenPitch.is_instance(token_i):
                        token_type_i = Tokenizer.TokenPitch
                    elif Tokenizer.TokenVel.is_instance(token_i) or token_i == TOKEN_BLANK:
                        token_type_i = Tokenizer.TokenVel
                    elif Tokenizer.TokenNoteOff.is_instance(token_i):
                        token_type_i = Tokenizer.TokenNoteOff
                    else:
                        token_type_i = "Other"
                    if i % 3 == 0:
                        dest_token_type_i = (Tokenizer.TokenOnset,)
                    elif i % 3 == 1:
                        dest_token_type_i = (Tokenizer.TokenPitch, Tokenizer.TokenNoteOff)
                    elif i % 3 == 2:
                        prev_token_i = gen_tokens[b, i-1].item()
                        if Tokenizer.TokenNoteOff.is_instance(prev_token_i) and token_i != TOKEN_BLANK:
                            # If the previous token is a NoteOff, then the current velocity should be TOKEN_BLANK
                            print(f"Warning: The generated token {token_i} at batch idx {b} position {i} is a velocity token, but the previous token is a NoteOff. The velocity should be {TOKEN_BLANK}, but got {token_i}.")
                        dest_token_type_i = (Tokenizer.TokenVel, )  
                    if not token_type_i in dest_token_type_i:
                        print(f"Warning: The generated token {token_i} at batch idx {b} position {i} is not in the expected format. Expected: {dest_token_type_i}, but got: {token_type_i}.")

                

            if hasattr(self.config, "hybrid_global_local_cross_attn") and self.config.hybrid_global_local_cross_attn:
                # Force the output sequence to in the format of onset, pitch, and velocity
                if i % 3 == 0: # Onset or EOS token
                    begin,end = Tokenizer.TokenOnset.get_bound()
                    probs[:, :, begin:end] += 1
                    probs[:, :, self.eos_token] += 1
                elif i % 3 == 1: # Pitch token, NoteOn or NoteOff
                    begin, end = Tokenizer.TokenPitch.get_bound()
                    probs[:, :, begin:end] += 1
                    begin, end = Tokenizer.TokenNoteOff.get_bound()
                    probs[:, :, begin:end] += 1
                elif i % 3 == 2: # Velocity token
                    begin, end = Tokenizer.TokenVel.get_bound()
                    probs[:, :, begin:end] += 1
                    probs[:, :, TOKEN_BLANK] += 1  # Force the velocity token to be a valid token
            
            
            token_probs = probs
            curr_token = torch.argmax(token_probs, dim=-1) # [batch_size, pred_step]
            gen_tokens[:, i:i+pred_step] = curr_token
            max_num_tokens += pred_step
            # Check  EOS
            eos_flags = eos_flags | (curr_token[:, 0] == self.eos_token)
            if berak_on_eos and eos_flags.int().sum() == batch_size:
                break

            # Update current frame index
            gen_onsets = curr_token[:, 0]
            frames_per_second = DEFAULT_SAMPLE_RATE  / DEFAULT_HOP_WIDTH
            eos_flags_int = eos_flags.int()
            for b in range(batch_size):
                onset_i = gen_onsets[b].item()
                if eos_flags_int[b] == 1:
                    curr_frame_index[b] = T - 1
                    continue
                if Tokenizer.TokenOnset.is_instance(onset_i): # Check if it is a onset token
                    onset_val = Tokenizer.TokenOnset.get_value(onset_i)
                    onset_sec = onset_val / Tokenizer.ONSET_SEC_UP_SAMPLING
                    onset_frame_index = int(onset_sec * frames_per_second)
                    curr_frame_index[b] = min(T-1, onset_frame_index)

        return gen_tokens #[:, :max_num_tokens]

    
    



