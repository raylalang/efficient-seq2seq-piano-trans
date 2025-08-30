"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as Functional
from model.Layers import *
from model.Attention import Multi_Head_Attention, RelativeGlobalAttention


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.config = config

        self.pre_attention_layer_norm = LayerNorm(config.emb_dim)
        if config.encoder_attention == "Multi_Head_Attention":
            window_size = None
            if hasattr(config, 'encoder_window_size'):
                window_size = config.encoder_window_size
            self.attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate, window_size=window_size)
        elif config.encoder_attention == "RelativeGlobalAttention":
            self.attention = RelativeGlobalAttention(d_model=config.emb_dim, num_heads=config.num_heads, dropout=config.dropout_rate)
        else:
            raise "Unknown attention type: " + config.encoder_attention
        self.dropout1 = nn.Dropout(config.dropout_rate)
        
        self.pre_mlp_layer_norm = LayerNorm(config.emb_dim)
        self.mlp = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, inputs, encoder_mask=None, deterministic=False):
        x = self.pre_attention_layer_norm(inputs)
        x = self.attention(x, x, mask=encoder_mask, deterministic=deterministic)
        x = self.dropout1(x)
        x = x + inputs

        y = self.pre_mlp_layer_norm(x)
        y = self.mlp(y)
        y = self.dropout2(y)
        y = y + x

        return y

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        
        # self.dense = nn.Linear(in_features=config.emb_dim, out_features=config.emb_dim)
        self.dense = nn.Linear(in_features=config.encoder_input_dim, out_features=config.emb_dim)
        self.embed = FixedEmbed(features=config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.layer_norm = LayerNorm(config.emb_dim)
        
    def forward(self, encoder_input_tokens, encoder_mask=None, deterministic=False):
        """_summary_

        Args:
            encoder_input_tokens (_type_): shape [batch, length, depth]
            encoder_mask (_type_, optional): _description_. Defaults to None.
            deterministic (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: shape [batch, length, emb_dim]
        """
        cfg = self.config
        assert encoder_input_tokens.ndim == 3  # [batch, length, depth]

        seq_length = encoder_input_tokens.shape[-2]
        batch_size = encoder_input_tokens.shape[0]

        # [batch, length, depth] -> [batch, length, emb_dim]
        x = self.dense(encoder_input_tokens)

        inputs_positions = torch.arange(seq_length)[None, :].to(encoder_input_tokens.device)
        x = x + self.embed(inputs_positions)
        
        x = self.dropout(x)
        x = x.type(cfg.dtype)

        for layer in self.encoder_layers:
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = layer(x, encoder_mask, deterministic)
            
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x
    
