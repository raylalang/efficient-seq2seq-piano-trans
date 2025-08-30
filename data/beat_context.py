
from __future__ import annotations
import numpy as np
import librosa

def compute_beats_full_audio(audio: np.ndarray, sr: int):
    """
    Run an off-the-shelf beat tracker on the full audio once (per piece).
    Returns (tempo, beat_times_sec).
    """
    # librosa expects float32 in [-1, 1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        maxv = np.max(np.abs(audio)) + 1e-8
        if maxv > 0:
            audio = audio / maxv
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, trim=False, units='frames')
    beat_times_sec = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times_sec.astype(np.float32)

def group_beats_into_bars_simple(beat_times_sec: np.ndarray, beats_per_bar: int = 4):
    """
    Simple bar grouping assuming constant meter (e.g., 4/4). Returns an array of shape [num_bars+1] with
    pointers into the beat array, so that beats for bar i are in [ptrs[i], ptrs[i+1]).
    """
    n_beats = len(beat_times_sec)
    if n_beats == 0:
        return np.array([0], dtype=np.int64)
    num_bars = int(np.ceil(n_beats / float(beats_per_bar)))
    ptrs = np.arange(0, (num_bars + 1) * beats_per_bar, beats_per_bar, dtype=np.int64)
    ptrs[-1] = n_beats  # clip last pointer to actual length
    return ptrs

def select_context_bar_window(ptrs: np.ndarray, center_bar_idx: int, num_context_bars_each_side: int):
    """
    Given bar pointers and a center bar index, select a symmetric window of bars.
    Returns (start_bar, end_bar_exclusive).
    """
    n_bars = len(ptrs) - 1
    start_bar = max(0, center_bar_idx - num_context_bars_each_side)
    end_bar = min(n_bars, center_bar_idx + num_context_bars_each_side + 1)
    return start_bar, end_bar

def slice_context_seconds(beat_times_sec: np.ndarray, ptrs: np.ndarray, start_bar: int, end_bar: int):
    """
    Convert the selected bar window into (begin_sec, end_sec) using beat boundaries.
    """
    if len(beat_times_sec) == 0:
        return 0.0, 0.0
    start_beat = ptrs[start_bar]
    end_beat = max(ptrs[end_bar] - 1, start_beat)
    # Context is from the beginning of the first beat to the *end* of the last beat (approx by next inter-beat gap)
    begin_sec = float(beat_times_sec[start_beat])
    if end_beat + 1 < len(beat_times_sec):
        gap = float(beat_times_sec[end_beat + 1] - beat_times_sec[end_beat])
    else:
        # fallback: average gap of last few beats
        gaps = np.diff(beat_times_sec[-min(8, len(beat_times_sec)) :])
        gap = float(np.mean(gaps)) if len(gaps) > 0 else 0.5
    end_sec = float(beat_times_sec[end_beat] + max(gap, 1e-3))
    return begin_sec, end_sec

def nearest_beat_index(beat_times_sec: np.ndarray, t_sec: float):
    if len(beat_times_sec) == 0:
        return 0
    idx = int(np.argmin(np.abs(beat_times_sec - t_sec)))
    return idx

def bar_index_for_time(beat_times_sec: np.ndarray, ptrs: np.ndarray, t_sec: float, beats_per_bar: int = 4):
    """
    Approximate: find nearest beat, then map to bar via ptrs // beats_per_bar.
    """
    b = nearest_beat_index(beat_times_sec, t_sec)
    bar = int(b // beats_per_bar)
    # Clip to valid bar range
    n_bars = len(ptrs) - 1
    bar = max(0, min(bar, max(0, n_bars - 1)))
    return bar
