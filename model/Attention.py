"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
import torch.nn as nn

from model.Layers import *
from model.Mask import *
import math
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


from xformers.ops import memory_efficient_attention
import xformers.ops.fmha.attn_bias as xformer_attn_bias


# from torch.nn.attention.flex_attention import flex_attention, create_block_mask
# from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks

class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, head_dim, dtype=torch.float32, dropout_rate=0.0, kernel_init=None, float32_logits=False, window_size=None, is_causal=False):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.projection = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.kernel_init = kernel_init if kernel_init is not None else nn.init.xavier_uniform_
        self.float32_logits = float32_logits
        self.output = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim)
        
        self.is_causal = is_causal
        self.window_size = window_size
            

    def dot_product_attention(self, query, key, value, bias=None, deterministic=False):
        assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], ('q, k, v batch dims must match.')
        assert query.shape[-2] == key.shape[-2] == value.shape[-2], ('q, k, v num_heads must match.')
        assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
        assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

        # Casting logits and softmax computation for float32 for model stability.
        if self.float32_logits:
            query = query.float()
            key = key.float()

        # `attn_weights`: [batch, num_heads, q_length, kv_length]
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key)

        # Apply attention bias: masking, dropout, proximity bias, etc.
        if bias is not None:
            attn_weights = attn_weights + bias.to(attn_weights.dtype).to(query.device)

        # Normalize the attention weights across `kv_length` dimension.
        attn_weights = F.softmax(attn_weights, dim=-1).to(self.dtype)

        attn_weights = self.dropout(attn_weights) #edited from original code

        # Take the linear combination of `value`.
        y = torch.einsum('bhqv,bvhd->bqhd', attn_weights, value)
        return y, attn_weights
    
    def flash_attn_sliding_window_attention(self, query, key, value,decode=False):
        assert self.window_size is not None, 'Sliding window attention requires a window size.'
        if self.is_causal:
            window_size = (self.window_size - (self.window_size%2), -1)
        else:
            w = int(self.window_size // 2)
            window_size = (w, w)
        query = query.to(torch.float16)
        key = key.to(torch.float16)
        value = value.to(torch.float16)
        x = flash_attn_func(query, key, value, dropout_p=self.dropout_rate, window_size=window_size, causal=self.is_causal, deterministic=decode)
        x = x.float()  # Convert back to float32
        return x
        
    def xformers_sliding_window_attention(self, query, key, value, dropout_p=0.0):
        assert self.window_size is not None, 'Sliding window attention requires a window size.'
        
        window_left = window_right = self.window_size // 2
        if self.is_causal:
            window_left = self.window_size
            window_right = 0
        bias = xformer_attn_bias.LocalAttentionFromBottomRightMask(window_left=window_left, window_right=window_right, dtype=query.dtype)

        x = memory_efficient_attention(
            query, # (batch_size, seq_len, num_heads, head_dim)
            key,
            value,
            attn_bias=bias,
            p=dropout_p,
        )
        
        return x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
        
        
    def initialize_decoder_cache(self):
        """Initializes the cache for autoregressive decoding."""
        if hasattr(self, 'cached_key'):
            delattr(self, 'cached_key')
        if hasattr(self, 'cached_value'):
            delattr(self, 'cached_value')
        if hasattr(self, 'cache_index'):
            delattr(self, 'cache_index')

    def forward(self, inputs_q, inputs_kv, mask=None, bias=None, decode=False, deterministic=False, return_attn_weights=False, sliding_window_size=None):
        #In Original MT3, they initialize the parameter with query_init, using customized Dense Layer
        query = self.projection(inputs_q).view(inputs_q.size(0), inputs_q.size(1), self.num_heads, self.head_dim)
        key = self.projection(inputs_kv).view(inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim)
        value = self.projection(inputs_kv).view(inputs_kv.size(0), inputs_kv.size(1), self.num_heads, self.head_dim)

        if decode:
            batch, length, num_heads, head_dim,  = key.size()
            expected_shape = (batch, 1, num_heads, head_dim)
            if expected_shape != query.size():
                raise ValueError('Autoregressive cache shape error, '
                                'expected query shape %s instead got %s.' %
                                (expected_shape, query.size()))
            
            if hasattr(self, 'cached_key'):
                cached_key = getattr(self, 'cached_key')
                cached_value = getattr(self, 'cached_value')
                cached_index = getattr(self, 'cache_index')
                key = torch.concat([cached_key, key], dim=1)  # Append new key
                value = torch.concat([cached_value, value], dim=1)  # Append
            else:
                cached_index = 0
            
            # Sliding window handling
            if sliding_window_size is not None:
                key = key[:, -sliding_window_size:, :, :]  # Use only the last `sliding_window_size` keys for decoding
                value = value[:, -sliding_window_size:, :, :]  # Use only the last `sliding_window_size` values for decoding

            setattr(self, 'cached_key', key)
            setattr(self, 'cached_value', value)
            setattr(self, 'cache_index', cached_index + 1)

        if mask is not None:
            attention_bias = torch.where(mask > 0,
                                        torch.zeros_like(mask).to(self.dtype),
                                         float("-inf") * torch.ones_like(mask).to(self.dtype))
        else:
            attention_bias = None

        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)
        
        if return_attn_weights:
            x, attn_weights = self.dot_product_attention(
                    query,
                    key,
                    value,
                    bias=attention_bias,
                    deterministic=deterministic)
        else:
            dropout_p = self.dropout_rate if not decode else 0.0
            if self.window_size is None:
                # Faster implementation using PyTorch's built-in attention
                query = query.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)

                x = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_bias, dropout_p=dropout_p)
                x = x.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, head_dim)
            else:
                x = self.flash_attn_sliding_window_attention(query, key, value, decode=decode)
                # x = self.xformers_sliding_window_attention(query, key, value, dropout_p=dropout_p)
        
        out = self.output(x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3)))
        
        if return_attn_weights:
            return out, None
        
        return out

class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, q, kv, mask = None, deterministic = None):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = kv.shape
        assert q.size()[1] == seq_len
        
        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )
        
        k_t = self.key(kv).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(kv).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(q).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel