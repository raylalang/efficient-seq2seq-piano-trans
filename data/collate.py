from torch.utils.data._utils.collate import default_collate
import torch


def _pad_1d(batch_list, pad_value=0.0, dtype=None):
    """Pad a list of 1D tensors to [B, T_max], return (padded, lengths)."""
    assert len(batch_list) > 0
    if dtype is None:
        dtype = batch_list[0].dtype
    lengths = torch.tensor([int(x.numel()) for x in batch_list], dtype=torch.long)
    T = int(lengths.max())
    out = torch.full((len(batch_list), T), pad_value, dtype=dtype)
    for i, x in enumerate(batch_list):
        n = int(x.numel())
        if n > 0:
            out[i, :n] = x
    return out, lengths


def collate_with_context(batch):
    """
    batch: list of dict samples
    Pads:
      - context_audio -> [B, T_wav], context_audio_len -> [B]
      - context_beat_times -> [B, T_beats] (pad=-1.0), context_num_beats -> [B]
    Everything else uses default_collate.
    """
    # fast path if no context in this batch (baseline run)
    if "context_audio" not in batch[0] and "context_beat_times" not in batch[0]:
        return default_collate(batch)

    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "context_audio":
            wavs = [b[k] for b in batch]  # list of 1D float tensors
            padded, lengths = _pad_1d(wavs, pad_value=0.0)
            out["context_audio"] = padded
            out["context_audio_len"] = lengths
        elif k == "context_beat_times":
            beats = [b[k] for b in batch]  # list of 1D float tensors
            padded, lengths = _pad_1d(beats, pad_value=-1.0)
            out["context_beat_times"] = padded
            out["context_num_beats"] = lengths
        else:
            out[k] = default_collate([b[k] for b in batch])
    return out
