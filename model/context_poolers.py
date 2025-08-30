
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_weights(frame_times_sec: torch.Tensor, centers_sec: torch.Tensor, sigma: float):
    """
    frame_times_sec: [T] or [B, T]
    centers_sec: [K] (K beats)
    returns: weights [K, T] normalized along T
    """
    if frame_times_sec.dim() == 1:
        frame_times_sec = frame_times_sec.unsqueeze(0)  # [1, T]
    # [K, 1] - [1, T] -> [K, T]
    diff = centers_sec.unsqueeze(1) - frame_times_sec
    w = torch.exp(-0.5 * (diff / (sigma + 1e-8)) ** 2)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    return w

class BeatPooler(nn.Module):
    def __init__(self, d_model: int, sigma_frames: float = 6.0, hop_length: int = 512, sample_rate: int = 16000, add_tempo_features: bool = False):
        super().__init__()
        self.d_model = d_model
        self.sigma_frames = float(sigma_frames)
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.add_tempo_features = add_tempo_features
        if add_tempo_features:
            self.tempo_proj = nn.Linear(1, d_model)

    def forward(self, enc_frames: torch.Tensor, beat_times_sec: torch.Tensor):
        """
        enc_frames: [B, T, d]
        beat_times_sec: list/tuple length B of 1D tensors with K_b beat times (sec) per sample, or a padded Tensor [B, K]
        Returns: beat_emb [B, K, d]
        """
        B, T, d = enc_frames.shape
        device = enc_frames.device
        # Frame times in seconds
        frame_times_sec = torch.arange(T, device=device, dtype=torch.float32) * (self.hop_length / float(self.sample_rate))
        # Normalize for each sample independently
        beat_embs = []
        max_K = 0
        beats_list = []
        for b in range(B):
            bt = beat_times_sec[b]
            if isinstance(bt, torch.Tensor):
                bt = bt.to(device=device, dtype=torch.float32)
            else:
                bt = torch.tensor(bt, device=device, dtype=torch.float32)
            beats_list.append(bt)
            max_K = max(max_K, bt.numel())
        # pad beat times
        padded = torch.zeros((B, max_K), device=device, dtype=torch.float32)
        mask = torch.zeros((B, max_K), device=device, dtype=torch.bool)
        for b, bt in enumerate(beats_list):
            k = bt.numel()
            if k > 0:
                padded[b, :k] = bt
                mask[b, :k] = True

        # Compute weights [B, K, T]
        sigma_sec = self.sigma_frames * (self.hop_length / float(self.sample_rate))
        W = []
        for b in range(B):
            w = gaussian_weights(frame_times_sec, padded[b, mask[b]], sigma=sigma_sec)  # [K_b, T]
            W.append(w)
        # Pool
        beat_emb = []
        for b in range(B):
            Kb = W[b].shape[0]
            if Kb == 0:
                beat_emb.append(torch.zeros((1, d), device=device, dtype=enc_frames.dtype))
                continue
            # [Kb, T] @ [T, d] -> [Kb, d]
            pooled = W[b] @ enc_frames[b]
            beat_emb.append(pooled)
        # Pad to [B, K_max, d]
        out = torch.zeros((B, max_K, d), device=device, dtype=enc_frames.dtype)
        for b, pooled in enumerate(beat_emb):
            Kb = pooled.shape[0]
            out[b, :Kb] = pooled
        return out, mask  # mask shows valid beats per sample

class BarPooler(nn.Module):
    def __init__(self, d_model: int, add_bar_pos: bool = True):
        super().__init__()
        self.d_model = d_model
        self.add_bar_pos = add_bar_pos
        if add_bar_pos:
            self.pos_embed = nn.Embedding(128, d_model)

    def forward(self, beat_emb: torch.Tensor, beat_mask: torch.Tensor, beats_per_bar: int = 4):
        """
        beat_emb: [B, K, d]; beat_mask: [B, K] boolean
        Returns bar_emb: [B, M, d], bar_mask: [B, M] boolean
        """
        B, K, d = beat_emb.shape
        M = int((K + beats_per_bar - 1) // beats_per_bar)
        device = beat_emb.device
        # Group by simple folding
        pad_len = M * beats_per_bar - K
        if pad_len > 0:
            beat_emb = F.pad(beat_emb, (0, 0, 0, pad_len))   # pad K dimension
            beat_mask = F.pad(beat_mask, (0, pad_len), value=False)
        beat_emb = beat_emb.view(B, M, beats_per_bar, d)
        beat_mask = beat_mask.view(B, M, beats_per_bar)

        # mean over valid beats in the bar
        masked = beat_emb * beat_mask.unsqueeze(-1).to(beat_emb.dtype)
        denom = beat_mask.sum(dim=2, keepdim=True).clamp_min(1).to(beat_emb.dtype)
        bar_emb = masked.sum(dim=2) / denom  # [B, M, d]
        bar_mask = (denom.squeeze(-1) > 0)

        if self.add_bar_pos:
            # add absolute bar index (0..M-1) embedding
            pos = torch.arange(M, device=device).unsqueeze(0).expand(B, -1)
            bar_emb = bar_emb + self.pos_embed(pos)

        return bar_emb, bar_mask
