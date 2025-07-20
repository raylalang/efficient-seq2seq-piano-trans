"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/layers.py#L489 to pytorch
"""

import torch
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

# device = "cuda" if torch.cuda.is_available() else "cpu"

def make_attention_mask(query_input, key_input, pairwise_fn: Callable=torch.mul, extra_batch_dims=0, dtype=torch.float32):
    mask = pairwise_fn(
        query_input.unsqueeze(-1),
        key_input.unsqueeze(-2)
    )
    mask = mask.unsqueeze(-3)
    
    for i in range(extra_batch_dims):
        mask = mask.unsqueeze(i)
    
    return mask.type(dtype)

def make_causal_mask(x, extra_batch_dims=0, dtype=torch.float32):
    idxs = torch.arange(x.shape[-1], dtype=torch.int32).expand(x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        torch.greater_equal,
        extra_batch_dims=extra_batch_dims,
        dtype=dtype)

def combine_masks(*masks, device = "cpu", dtype=torch.float32):
    masks = [m.to(device) for m in masks if m is not None]
    
    if not masks:
        return None
    
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    
    mask, *other_masks = masks
    
    for other_mask in other_masks:
        mask = torch.logical_and(mask, other_mask)

    return mask.type(dtype)

def combine_biases(*masks):
    masks = [m for m in masks if m is not None]
    
    if not masks:
        return None
    
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    
    mask, *other_masks = masks
    
    for other_mask in other_masks:
        mask = mask + other_mask
    
    return mask

def make_decoder_mask(decoder_target_tokens,
                      dtype,
                      decoder_causal_attention=None,
                      decoder_segment_ids=None):
    masks = []
    
    causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

    if decoder_causal_attention is not None:
        inputs_mask = make_attention_mask(
            decoder_causal_attention,
            decoder_causal_attention,
            torch.logical_and,
            dtype=dtype)
        masks.append(torch.logical_or(causal_mask, inputs_mask).astype(dtype))
    else:
        masks.append(causal_mask)

    masks.append(
        make_attention_mask(
            decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

    # Packing mask
    if decoder_segment_ids is not None:
        masks.append(
            make_attention_mask(
                decoder_segment_ids, decoder_segment_ids, torch.equal, dtype=dtype))

    return combine_masks(*masks, dtype=dtype, device=decoder_target_tokens.device)


def make_causal_mask_k_predict(batch_size, seq_len, k=2):
    """
    Creates a causal mask for sequence prediction tasks, allowing the model to predict the next k tokens.
    Args:
        seq_len (int): Length of the sequence.
        k (int): Number of future tokens that can be predicted.
    # Output: [batch_size, 1, seq_len, seq_len]
        tensor([[ 1, 1, 0, 0, 0, 0...],
                [ 1, 1, 0, 0, 0, 0...],
                [ 1, 1, 1, 1, 0, 0, ...],
                [ 1, 1, 1, 1, 0, 0, ...],
                [ 1, 1, 1, 1, 1, 1, 0, 0, ...],
                [ 1, 1, 1, 1, 1, 1, 0, 0, ...],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
                ...])
    """
    assert seq_len % k == 0, "seq_len must be divisible by k"
    mask = torch.zeros([batch_size, 1, seq_len, seq_len])
    for i in range(seq_len//k):
        for j in range(k):
            mask[:,:,  i*k + j, :(i+1)*k] = 1
    return mask  # 取反：True 表示可见
    
def make_causal_chunk_mask_k_predict(batch_size, seq_len, k=2, chunk_size=4):
    """
    Creates a causal mask for sequence prediction tasks, allowing the model to predict the next k tokens.
    Args:
        seq_len (int): Length of the sequence.
        k (int): Number of future tokens that can be predicted.
    # Output:
        tensor([[ 1, 1, 0, 0, 0, 0...],
                [ 1, 1, 0, 0, 0, 0...],
                [ 1, 1, 1, 1, 0, 0, ...],
                [ 1, 1, 1, 1, 0, 0, ...],
                [ 0, 0, 1, 1, 1, 1, 0, 0, ...],
                [ 0, 0, 1, 1, 1, 1, 0, 0, ...],
                [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...],
                [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, ...],
                ...])
    """
    assert seq_len % k == 0, "seq_len must be divisible by k"
    mask = torch.zeros([batch_size, 1, seq_len, seq_len])
    for i in range(seq_len//k):
        for j in range(k):
            row = i * k + j
            end = (i+1)*k
            begin = max(0, end - chunk_size)
            mask[:,:,  row, begin:end] = 1
    return mask  # 取反：True 表示可见

def make_decoder_only_mask(prefix_len, output_seq_len, batch_size, device, chunk_size = 10e9):
    """
    Creates a mask for decoder-only models, allowing the model to attend to the prefix and the output sequence.
    Args:
        prefix_len (int): Length of the prefix.
        output_seq_len (int): Length of the output sequence.
        batch_size (int): Batch size.
    Returns:
        torch.Tensor: Mask tensor of shape [batch_size, 1, prefix_len + output_seq_len, prefix_len + output_seq_len].
        
    Example:
        when prefix_len = 4 and output_seq_len = 6, the output will be:
             [[[[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]
                
        chunk_size = 2:
        [[[[ 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]
    """
    mask = torch.zeros([batch_size, 1, prefix_len + output_seq_len, prefix_len + output_seq_len], device=device)
    for row in range(prefix_len):
        begin = max(0, row - chunk_size)
        end = row + 1 + chunk_size
        end = min(end, prefix_len)
        mask[:, :, row, begin:end] = 1
    for col in range(prefix_len):
        mask[:, :, prefix_len:, col] = 1
    for row in range(prefix_len, prefix_len + output_seq_len):
        # mask[:, :, i, :prefix_len + i + 1] = 1
        mask[:, :, row, :row + 1] = 1
        
        
    return mask  # 取反：True 表示可见
    
def make_harmonic_awared_mask(batch_size, seq_len, bins_per_octave = 12, n_har = 12, har_width = 2):
    """
    Creates a harmonic-aware mask for sequence prediction tasks.
    Args:
        batch_size (int): Batch size.
        seq_len (int): Length of the sequence.
        bins_per_octave (int): Number of bins per octave.
        n_har (int): Number of harmonics to consider.
    Returns:
        torch.Tensor: Mask tensor of shape [batch_size, 1, seq_len, seq_len].
    """
    
    steps = [0]
    n = 2
    while(1):
        step = bins_per_octave * np.log(n) / np.log(2)
        step = int( np.round(step) )
        if step - steps[-1] <= 1: # Overlap: if the step is the same as the last one, we stop
            break
        steps.append(step)
        n += 1
        if n > n_har:
            break
        
    for i in range(1, 12):
        steps.append(bins_per_octave * i)
    
    # steps = [bins_per_octave * np.log(i) / np.log(2) for i in range(1, n_har + 1)]
    
    steps += [-s for s in steps]
    # steps = [int(np.round(s)) for s in steps]
    
    mask = torch.zeros([seq_len, seq_len])
    
    for i in range(seq_len):
        for s in steps:
            j = i + s
            if j >=0 and j < seq_len:
                scaled_har_width = int( har_width / ( abs(s)/bins_per_octave + 1 ) )
                j_begin = max(0, j - scaled_har_width)
                j_end = min(seq_len, j + scaled_har_width + 1)
                mask[i, j_begin:j_end] = 1
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
                
    return mask

def make_sliding_window_mask(seq_len: int, window_size:int, causal:bool) -> torch.Tensor:
    """
    创建sliding window attention mask
    """
    # 创建基础的因果mask (下三角矩阵)
    mask = torch.zeros(seq_len, seq_len)
    
    if not causal:
        window_size = window_size // 2
    
    # 应用sliding window限制
    for i in range(seq_len):
        # 只保留窗口内的attention
        start = max(0, i - window_size)
        if causal:
            end = i + 1
        else:
            end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = 1
        
    return mask.unsqueeze(0).unsqueeze(0)  # 添加batch和head维度


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    seq_len = 8
    k = 1
    mask = make_causal_mask_k_predict(batch_size, seq_len, k)
    print(mask)
    mask = make_causal_chunk_mask_k_predict(batch_size, seq_len, k, chunk_size=4)
    print(mask)
    
    mask = make_decoder_mask(torch.ones([1, 8], dtype=torch.float32), dtype=torch.float32)
    print(mask)
    
    print("Decoder-only mask:")
    prefix_len = 8
    output_seq_len = 6
    chuni_size = 2000
    decoder_only_mask = make_decoder_only_mask(prefix_len, output_seq_len, batch_size, device="cpu", chunk_size=chuni_size)
    print(decoder_only_mask)
    
    print("Harmonic-aware mask:")
    seq_len  = 360
    bins_per_octave = 48
    n_har = 16
    har_width = 0
    harmonic_mask = make_harmonic_awared_mask(batch_size, seq_len, bins_per_octave=bins_per_octave, n_har=n_har, har_width=har_width)
    print(harmonic_mask.numpy())
    plt.imsave('img/harmonic_mask_seq-len=%d_bins-per-o=%d_n-har=%d_har-width=%d.png'%(seq_len, bins_per_octave, n_har, har_width), harmonic_mask[0, 0].numpy())
    
    print("Sliding window mask:")
    window_size = 5
    sliding_window_mask = make_sliding_window_mask(seq_len, window_size, causal=False, device=torch.device("cpu"))
    print(sliding_window_mask)  
    plt.imsave('img/sliding_window_mask_seq-len=%d_window-size=%d.png'%(seq_len, window_size), sliding_window_mask[0, 0].numpy())
    
    # Output:
    # tensor([[ 1, 1, 0, 0, 0, 0...],
    #         [ 1, 1, 1, 1, 0, 0, ...],
    #         [ 1, 1, 1, 1, 1, 1, 0, 0, ...],
    #         [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...],
    #         ...])