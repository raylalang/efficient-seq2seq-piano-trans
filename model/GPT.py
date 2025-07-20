import sys
import os
work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)

import torch
import torch.nn as nn
import torch.nn.functional as Functional

from model.Layers import *
from model.Mask import *
from model.HPPNet import HPPNet
from data.constants import *
from torch.nn.utils.rnn import pad_sequence
# from attrdict import AttrDict
from config.utils import DictToObject
from model.Decoder import DecoderLayer
from model.Attention import Multi_Head_Attention
import copy
import omegaconf
from omegaconf import OmegaConf
from model.QWen import QwenDecoder


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.config = config

        self.pre_self_attention_layer_norm = LayerNorm(config.emb_dim)
        self.self_attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        # self.pre_cross_attention_layer_norm = LayerNorm(config.emb_dim)
        # self.encoder_decoder_attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.pre_mlp_layer_norm = LayerNorm(config.emb_dim)
        self.mlp = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        
        if config.use_MoE:
            self.mlp2 = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        
        self.dropout3 = nn.Dropout(config.dropout_rate)

    def forward(self, inputs, encoded, decoder_mask=None):
        x = self.pre_self_attention_layer_norm(inputs)
        x= self.self_attention(x, x, mask=decoder_mask)
        x = self.dropout1(x)
        y = x + inputs

        z = self.pre_mlp_layer_norm(y)
        
        if self.config.use_MoE == True:
            # => [batch, length/2, 2, emb_dim]
            z = torch.stack([self.mlp(z[:, 0::2, :]), self.mlp2(z[:, 1::2, :])], dim=2)  # Two MLPs in parallel
            batch_size, seq_length, cp_size_midi, emb_dim = z.size()
            z = z.reshape(batch_size, seq_length * cp_size_midi, emb_dim)  # [batch, length/2 * 2, emb_dim]
        else:
            z = self.mlp(z)
        
        z = self.dropout3(z)
        z = z + y

        return z
    
class FrequencyTransformerEncoder(nn.Module):
    def __init__(self, config):
        super(FrequencyTransformerEncoder, self).__init__()
        
        emb_dim_out = config.emb_dim
        
        config = copy.deepcopy(config)
        
        config.num_heads = 1
        config.head_dim = config.slide_window_size
        config.emb_dim = config.slide_window_size
        config.mlp_dim = config.slide_window_size * 4
        
        self.config = config
        
        self.pos_emb = FixedEmbed(config.emb_dim)
        
        self.freq_attention_layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.num_freq_attention_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = LayerNorm(emb_dim_out)
        self.input_linear = nn.Linear(config.num_mel_bins, emb_dim_out)
        self.linear_linear2 = nn.Linear(config.num_mel_bins//4 * config.slide_window_size, emb_dim_out)
        
    def forward(self, x):
        """_summary_

        Args:
            inputs (x): [batch_size, seq_len, n_freq]

        Returns:
            tensor: [batch_size, seq_len, emb_dim]
        """
        out_1 = self.input_linear(x)  # [batch_size, input_len, emb_dim]
        
        # [batch_size, input_len, n_mel] => [batch_size, input_len, n_mel, win_size]
        y = self.center_window(x, win_size=self.config.slide_window_size)  # Center-aligned sliding window
        batch_size, input_len, n_mel, win_size = y.size()
        # => [batch_size*input_len, n_mel, win_size]
        y = y.reshape(-1, y.size(2), y.size(3))
        
        position_ids = torch.arange(n_mel)[None, :].to(x.device)  # [batch_size, input_len]
        y = y + self.pos_emb(position_ids)  # Add positional embeddings
        y = self.dropout(y)
        
        for i, layer in enumerate( self.freq_attention_layers ):
            y = layer(y, y)
            
            
        # => [batch_size * input_len, n_mel//4, win_size]
        y = y.reshape(-1, n_mel//4, 4, win_size).mean(dim=2)  
        y = y.reshape(-1, n_mel//4 * win_size)  # [batch_size * input_len, n_mel//4 * win_size]
        y = self.linear_linear2(y)  # [batch_size * input_len, emb_dim]
        y = y.reshape(batch_size, input_len, -1)  # [batch_size, input_len, emb_dim]
        y = self.layer_norm(y)  # [batch_size, input_len, emb_dim]
        out_2 = self.dropout(y)  # [batch_size, input_len, emb_dim]
        
        
        
        out = out_1 + out_2  # [batch_size, input_len, emb_dim]
        
        return out
    
    def center_window(self, x: torch.Tensor, win_size: int) -> torch.Tensor:
        """
        Apply center-aligned sliding window on time dimension (dim=1), keeping time dimension unchanged.

        Args:
            x: Tensor of shape [B, T, H]
            k: Window size (must be odd)

        Returns:
            Tensor of shape [B, T, H, win_size]
        """
        assert win_size % 2 == 1, "Window size must be odd for center alignment"
        
        B, T, H = x.shape
        pad = win_size // 2  # padding size on each side
        
        # Pad the time dimension (dim=1) with replication (or you could use zero-padding)
        # Pad format: (before_last_dim, after_last_dim, ..., before_dim1, after_dim1)
        x_padded = F.pad(x, pad=(0, 0, pad, pad), mode='replicate')  # shape: [B, T + 2*pad, H]

        # Apply sliding window
        x_unfolded = x_padded.unfold(dimension=1, size=win_size, step=1)  # shape: [B, T, k, H]

        return x_unfolded

class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        config =DictToObject(dict(config))
        config.dtype = torch.float32 # 
        self.config = config
        
        self.input_linear = nn.Linear(config.num_mel_bins, config.emb_dim)
        
        self.pos_emb = FixedEmbed(self.config.emb_dim)
        
        
        # Frequency Transformer Encoder
        # self.freq_encoder = FrequencyTransformerEncoder(config)
        self.encoder = HPPNet(config)
        self.encoder_outputs_mlp = MlpBlock(emb_dim=config.encoder_size * config.num_mel_bins // 4, intermediate_dim=512, emb_dim_out=config.emb_dim)
        
        
        sub_token_names=[x.get_class_name() for x in sm_tokenizer.token_type_list]
        self.projector = CompoundProjector(config=config, sub_token_names=sub_token_names)

        # self.decoder_layers = nn.ModuleList([
        #     DecoderLayer(config) for _ in range(config.num_decoder_layers)
        # ])
        self.decoder = QwenDecoder()

        self.pad_token = TOKEN_PAD
        self.eos_token = TOKEN_END
        
        self.layer_norm = LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dense = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
        
    def forward(self, 
                encoder_input_tokens=None, 
                decoder_target_tokens=None, 
                decoder_target_tokens_score=None,
                decoder_input_tokens=None, 
                encoder_segment_ids=None, 
                decoder_segment_ids=None, 
                encoder_positions=None, 
                decoder_positions=None, 
                enable_dropout=True, 
                decode=False,
                dur_inputs = None,
                dur_targets = None,
    ):
        res_dict = {}
        decoder_input_tokens = self._shift_right(decoder_target_tokens, shift_step=1)
        
        # encoder_inputs = self.freq_encoder(encoder_input_tokens)
        
        encoder_outputs = self.encoder(encoder_input_tokens)
        encoder_outputs = torch.relu(encoder_outputs)
        B, T, F, H = encoder_outputs.size()
        encoder_outputs = encoder_outputs.reshape([B, T, F*H])
        # => [B, T, emb_dim]
        encoder_outputs = self.encoder_outputs_mlp(encoder_outputs)
        
        # Add pos_emb to encoded_seq
        seq_length = encoder_outputs.size()[1]
        # [batch, length, depth] -> [batch, length, emb_dim]
        encoder_output_positions = torch.arange(seq_length)[None, :].to(encoder_outputs.device)
        encoder_outputs = encoder_outputs + self.pos_emb(encoder_output_positions)
        
        
        
        if False:
            seq_length = encoder_inputs.size()[1]
            encoder_output_positions = torch.arange(seq_length)[None, :].to(encoder_inputs.device)
            encoder_inputs = encoder_inputs + self.pos_emb(encoder_output_positions, decode=decode)
        
        batch_size, input_len = encoder_outputs.size()[:2]
        
        decoder_inputs = self.projector(decoder_input_tokens, is_compound=False, add_pos_emb=True)
        output_len = decoder_inputs.size()[1]

        
        # => [batch_size, 1, input_len + output_len, input_len + output_len]
        # decoder_mask = make_decoder_only_mask(
        #     prefix_len=input_len, 
        #     output_seq_len=output_len, 
        #     batch_size=batch_size, 
        #     device=decoder_target_tokens.device,
        #     )
        # decoder_mask = make_causal_mask_k_predict(batch_size=batch_size, seq_len=input_len + output_len, k=1).to(decoder_target_tokens.device)
        
        # decoder_mask = torch.tril(torch.ones((input_len + output_len, input_len +  output_len), device=decoder_target_tokens.device))
        decoder_mask = torch.ones((input_len + output_len, input_len +  output_len), device=decoder_target_tokens.device)
        decoder_mask=decoder_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # => [batch_size, input_len + output_len, emb_dim]
        y = torch.concat([encoder_outputs, decoder_inputs], dim=1)
        
        if enable_dropout:
            y = self.dropout(y)
            
        if False:
            for i, layer in enumerate(self.decoder_layers):
                y = layer(y, y, decoder_mask=decoder_mask)
                
            y = self.layer_norm(y)
            if enable_dropout:
                y = self.dropout(y)
            y = self.dense(y)
        
        # Mask
        prefix_mask = torch.ones((B, encoder_outputs.size()[1]), dtype=encoder_outputs.dtype).to(encoder_outputs.device)
        attention_mask = (decoder_target_tokens != sm_tokenizer.PAD).float()
        attention_mask = torch.concat([prefix_mask, attention_mask], dim=1)  # [batch_size, input_len + output_len]
        
        
        psedo_input_labels = torch.ones([batch_size, input_len], dtype=torch.long, device=decoder_target_tokens.device) * TOKEN_PAD
        labels = torch.concat( [psedo_input_labels, decoder_input_tokens], dim=1)  # [batch_size, input_len + output_len]), 模型会自动shift labels，所以这里不需要 shift，直接 concat
        
        labels[labels == TOKEN_PAD] = -100  # Mask out padding tokens in the loss calculation, Hugging Face 默认的 CrossEntropyLoss(ignore_index=-100)
            
        outputs_dict = self.decoder(
            inputs_embeds=y, 
            # position_ids=None,
            attention_mask=attention_mask, 
            labels=labels,
            ) 
        # , labels=decoder_target_tokens,
        
        y = outputs_dict["logits"]  # [batch_size, input_len + output_len, vocab_size]
        loss = outputs_dict["loss"]
        
        decoder_outputs = y[:, input_len:, :]  # Only keep the decoder part
        
        # decoder_outputs[:, 1:output_len, :] = decoder_outputs[:, 0:output_len-1, :]  # Shift right by one to match the decoder input
        
        res_dict["decoder_outputs"] = decoder_outputs
        res_dict["loss"] = loss
        return res_dict
        
        
    def _shift_right(self, input_ids, shift_step=None):
        BOS = TOKEN_START

        shifted_input_ids = torch.zeros_like(input_ids)
        if shift_step is None:
            shift_step = SEQUENCE_POOLING_SIZE
        shifted_input_ids[..., shift_step:] = input_ids[..., :-shift_step].clone()
        shifted_input_ids[..., :shift_step] = BOS


        return shifted_input_ids
    

if __name__ == "__main__":
    config = OmegaConf.load("config/model/T5_DecoderOnly.yaml")
    gpt = GPT(config)
    print(gpt)
    t = 128
    x1 = torch.randn([2, t, config.num_mel_bins])  # [batch_size, seq_len, n_mel_bins]
    x2 = torch.ones([2, t], dtype=torch.long)  # [batch_size, seq_len, n_mel_bins]
    
    y = gpt(encoder_input_tokens=x1, decoder_target_tokens=x2)
    