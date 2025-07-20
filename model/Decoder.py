"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
from model.Layers import *
from model.Attention import Multi_Head_Attention
from data.constants import *
from model.Mask import make_sliding_window_mask

class DecoderLayer(nn.Module):
    def __init__(self, config,  layer_idx=0):
        super(DecoderLayer, self).__init__()
        self.config = config

        window_size = None
        if hasattr(config, 'decoder_window_size'):
            window_size = config.decoder_window_size
        self.pre_self_attention_layer_norm = LayerNorm(config.emb_dim)
        self.self_attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate, window_size=window_size, is_causal=True)
        self.dropout1 = nn.Dropout(config.dropout_rate)

        self.pre_cross_attention_layer_norm = LayerNorm(config.emb_dim)
        self.encoder_decoder_attention = Multi_Head_Attention(num_heads=config.num_heads, head_dim=config.head_dim, dropout_rate=config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

        self.pre_mlp_layer_norm = LayerNorm(config.emb_dim)
        self.mlp = MlpBlock(emb_dim=config.emb_dim, intermediate_dim=config.mlp_dim, activations=config.mlp_activations, intermediate_dropout_rate=config.dropout_rate)
        
        self.dropout3 = nn.Dropout(config.dropout_rate)
        
    def initialize_decoder_cache(self,):
        self.self_attention.initialize_decoder_cache()

    def forward(self, inputs, encoded, decoder_mask=None, encoder_decoder_mask=None, decode=False, return_attn_weights=False, sliding_window_size=None, decoder_targets_frame_index=None):
        # self-attention
        x = self.pre_self_attention_layer_norm(inputs)  
        x= self.self_attention(x, x, mask=decoder_mask, decode=decode, sliding_window_size=sliding_window_size)
        x = self.dropout1(x)
        x = x + inputs

        # Cross-attention
        y = self.pre_cross_attention_layer_norm(x)
        if return_attn_weights:
            y, cross_attn_weights = self.encoder_decoder_attention(y, encoded, mask=encoder_decoder_mask, return_attn_weights=return_attn_weights)
        else:
            if hasattr(self.config, "hybrid_global_local_cross_attn",) and self.config.hybrid_global_local_cross_attn:
                y = self.hybrid_global_local_cross_attention(y, encoded, mask=encoder_decoder_mask, decode=decode, decoder_targets_frame_index=decoder_targets_frame_index)
            else:
                y = self.encoder_decoder_attention(y, encoded, mask=encoder_decoder_mask)
        y = self.dropout2(y)
        y = y + x

        # MLP block
        z = self.pre_mlp_layer_norm(y)
        z = self.mlp(z)
        z = self.dropout3(z)
        z = z + y

        if return_attn_weights:
            return z, cross_attn_weights
        else:
            return z
        
    def hybrid_global_local_cross_attention(self, y, encoded, mask=None, decode=None, decoder_targets_frame_index=None):
        """
        Selective cross attention with global-local attention mechanism.
        Args:
            y: Input tensor.
            encoded: Encoded tensor.
            decoder_mask: Decoder mask.
            encoder_decoder_mask: Encoder-decoder mask.
            decode: Whether to decode.
        Returns:
            Output tensor after applying selective cross attention.
        """
        if decode:
            assert y.size(1) == 1, "y should have batch size of 1 during decoding"
            # if encoded.size(1) == 1:
            #     return y
            return  self.encoder_decoder_attention(y, encoded, mask=mask)
        
        
        # Implement the hybrid cross global-local attention logic here
        batch_size, seq_length, emb_dim = y.size()
        assert seq_length % 3 == 0, "seq_length should be divisible by 3 for hybrid global-local attention: onset, pitch, and velocity"
        
        y_onset = y[:, 0::3, :]  # [batch, n, emb_dim] for onset
        y_pitch = y[:, 1::3, :]  # [batch, n, emb_dim] for pitch
        y_vel = y[:, 2::3, :] # [batch, n, emb_dim] for velocity
        
        mask_onset = None
        mask_pitch = None
        mask_vel = None
        if mask is not None:
            assert mask.ndim == 4, "mask should be 4D tensor"
            mask_onset = mask[:, :, ::3, :]
            mask_pitch = mask[:, :, 1::3, :]
            mask_vel = mask[:, :, 2::3, :]
        
        encoded_onset = encoded  # global attention for onset
        # Onset
        y_onset = self.encoder_decoder_attention(y_onset, encoded_onset, mask=mask_onset)
        # Pitch
        y_pitch = self.encoder_decoder_attention(y_pitch, encoded_onset, mask=mask_pitch)
        # Velocity
        y_vel = self.encoder_decoder_attention(y_vel, encoded_onset, mask=mask_vel)
        
        if False: # When window size = 1
            indices = decoder_targets_frame_index[:, 0::3]  # [batch, n] for onset
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n)  # [B, n]
            encoded_pit_vel = encoded[batch_indices, indices]  # => [batch, n, emb_dim] for pitch and velocity
            encoded_pit_vel = encoded_pit_vel.reshape(batch_size * n, 1, emb_dim)  # [batch * n, 1, emb_dim] for pitch and velocity

            # Pitch
            y_pitch = y_pitch.reshape([batch_size * n, 1, emb_dim])  # [batch * n, 1, emb_dim] for pitch
            y_pitch = self.encoder_decoder_attention(y_pitch, encoded_pit_vel, mask=None)
            y_pitch = y_pitch.reshape([batch_size, n, emb_dim])  # [batch, n, emb_dim] for pitch
            # Velocity
            y_vel = y_vel.reshape([batch_size * n, 1, emb_dim])  # [batch * n, 1, emb_dim] for velocity
            y_vel = self.encoder_decoder_attention(y_vel, encoded_pit_vel, mask=None)
            y_vel = y_vel.reshape([batch_size, n, emb_dim])  # [batch, n, emb_dim] for velocity
        
        y = torch.stack([y_onset, y_pitch, y_vel], dim=2)  # [batch, n, 3, emb_dim]
        y = y.reshape(batch_size, seq_length, emb_dim)  # [batch,
        
        return y
    
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        
        self.token_embed = Embed(config.vocab_size, config.emb_dim, one_hot=True)
        self.fixed_embed = FixedEmbed(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_decoder_layers)])
        self.layer_norm = LayerNorm(config.emb_dim)
        self.dense = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
        self.dense_pitch = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
        self.dense_velocity = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
            
            
    def initialize_decoder_cache(self):
        for layer in self.decoder_layers:
            layer.initialize_decoder_cache()

        
    def forward(self, encoded, decoder_input_tokens, encoded_pooling_dict=None, decoder_positions=None, decoder_mask=None, encoder_decoder_mask=None, deterministic=False, decode=False, max_decode_length=None, decoder_targets_frame_index=None):
        cfg = self.config
        assert decoder_input_tokens.ndim == 2  # [batch, len]
        res_dict = {}
        
        batch_size, seq_length = decoder_input_tokens.size()

        if decoder_positions is None:
            decoder_positions = torch.arange(seq_length)[None, :].to(decoder_input_tokens.device)

        # [batch, length] -> [batch, length, emb_dim]
        y = self.token_embed(decoder_input_tokens)
        y = y + self.fixed_embed(decoder_positions, decode=decode)
        y = self.dropout(y)
        y = y.type(cfg.dtype)
            

        for layer_idx ,layer in enumerate(self.decoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            encoded_i = encoded 
            encoder_decoder_mask_i = encoder_decoder_mask

            # Hierarchical pooling.
            if encoded_pooling_dict is not None:
                assert encoded is None, "encoded should be None when using encoded_pooling_dict"
                pooling_size = self.config.pooling_sizes[layer_idx]
                encoded_i = encoded_pooling_dict[pooling_size]  # [batch, encoded_length//pooling, emb_dim]
                if encoder_decoder_mask_i is not None:
                    encoder_decoder_mask_i = encoder_decoder_mask_i[:, :, :seq_length, ::pooling_size]  # [batch, num_heads, length, encoded_length//pooling]
            
            if encoder_decoder_mask_i is not None:
                encoder_decoder_mask_i = encoder_decoder_mask_i[:, :, :seq_length, :encoded_i.size(1)]  # [batch, num_heads, length, encoded_length]

            y = layer(y, encoded_i, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask_i, decode=decode, decoder_targets_frame_index=decoder_targets_frame_index)
            
            if hasattr(self.config, "get_encder_decoder_attention_weights") and self.config.get_encder_decoder_attention_weights:
                res_dict[f"decoder_layer_cross_attention_weights_{layer_idx}"] = None  # Save attention weights for each layer
        
        y = self.layer_norm(y)
        y = self.dropout(y)

        if hasattr(self.config, "hybrid_global_local_cross_attn") and self.config.hybrid_global_local_cross_attn:
            # Hybrid global-local cross attention
            # Different classification heads for onset, pitch, and velocity
            if decode:
                assert decoder_positions.size(1) == 1
                assert y.size(1) == 1, "y should have sequence length of 1 during decoding"
                if decoder_positions[0, 0] %3 == 0:
                    logits = self.dense(y)
                elif decoder_positions[0, 0] %3 == 1:
                    logits = self.dense_pitch(y)
                elif decoder_positions[0, 0] %3 == 2:
                    logits = self.dense_velocity(y)
            else:
                logits_onset = self.dense(y[:, 0::3, :])  # [batch, n, emb_dim] for onset
                logits_pitch = self.dense_pitch(y[:, 1::3, :])  # [batch, n, emb_dim] for pitch
                logits_velocity = self.dense_velocity(y[:, 2::3, :])  # [batch, n, emb_dim] for velocity
                logits = torch.stack([logits_onset, logits_pitch, logits_velocity], dim=2)  # [batch, n, 3, vocab_size]
                logits = logits.reshape(batch_size, seq_length, cfg.vocab_size)  # [batch, length, vocab_size]
        else:
            # [batch, length, emb_dim] -> [batch, length, vocab_size]
            logits = self.dense(y)

        res_dict.update( {
            "decoder_outputs":logits, 
            # "cross_attention_weights":cross_attention_weights, # [batch_size, num_heads, decoder_length, encoder_length]
        })
        return res_dict
    

    
    
class CompoundDecoder(nn.Module):
    def __init__(self, config, sub_token_names=[], decoder_type="performance", decoder_layer_num = None): # decoder_type can be "performance" or "score"
        super(CompoundDecoder, self).__init__()
        self.config = config
        self.sub_token_names = sub_token_names
        self.compound_size = len(sub_token_names)
        self.decoder_type = decoder_type
        
        self.token_embed = Embed(config.vocab_size, config.emb_dim, one_hot=True)
        self.fixed_embed = FixedEmbed(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        if decoder_layer_num is None:
            decoder_layer_num = config.num_decoder_layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(decoder_layer_num)])
        # self.encoder_layers_score = nn.ModuleList([EncoderLayer(config) for _ in range(3)])
        for name in sub_token_names:
            self.__setattr__(f"head_{name}", nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size))
        self.layer_norm = LayerNorm(config.emb_dim)
        
        self.mlp_attnMap_to_onset = MlpBlock(emb_dim=config.input_seq_length, intermediate_dim=512, emb_dim_out=config.vocab_size)
        
        self.dense = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
        if config.use_MoE:
            self.dense2 = nn.Linear(in_features=config.emb_dim, out_features=config.vocab_size)
        
    def forward(self, encoded, decoder_input_tokens, decoder_positions=None, decoder_mask=None, encoder_decoder_mask=None, deterministic=False, decode=False, enable_dropout=True):
        cfg = self.config
        
        y = self.token_embed(decoder_input_tokens)
            
        if decoder_positions is None:
            decoder_positions = torch.arange(y.size()[1])[None, :].to(y.device)

        # Add fixed positional embeddings
        y = y + self.fixed_embed(decoder_positions, decode=decode)
        
        if enable_dropout:
            y = self.dropout(y)
        y = y.type(cfg.dtype)

        # Decoder layers
        cross_attention_weights = None
        for layer_idx, layer in enumerate(self.decoder_layers):
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            if layer_idx < len(self.decoder_layers) - 1:
                y = layer(y, encoded, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask, decode=decode)
            else:
                # Get attention weights of the last layer
                y, cross_attention_weights = layer(y, encoded, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask, decode=decode, return_attn_weights=True)
        
        y = self.layer_norm(y)
        y = self.dropout(y)
        
        y = self.dense(y)  # [batch, length, emb_dim] -> [batch, length, vocab_size]
        
        return y, cross_attention_weights  # logits, cross_attention_weights, hidden_states
