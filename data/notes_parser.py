import torch
import numpy as np
import torch.nn.functional as Functional
import pandas as pd

def pianoroll_to_frame_tokens(pianoroll):
    """_summary_

    Args:
        pianoroll (torch.tensor.long): shape is [n_frames, 256]; 0~127: onset; 128~255: offset.
    Returns:
        frame_tokens (torch.tensor.long): shape is [n_frames, n_tokens]; n_tokens is the max token seq length.
    """
    raise "Not implemented yet!"

def frame_tokens_to_pianoroll(frame_tokens: torch.tensor):
    """
    frame tokens to onset+offset pianoroll.
    Args:
        frame_tokens(tensor.long):  [T, n_tokens], T indicates the frame number, n_token indictes the max token seq len of a frame.
    Outputs:
        pianoroll(tensor.long):  [T, 256], 0~127: onsets, 128~255:offsets.
    """
    T, n_tokens = frame_tokens.size()
    tokens = torch.clone(frame_tokens)
    tokens += 1
    tokens[tokens > 256] = 0
    indices = torch.nonzero(tokens)
    time_indices = torch.arange(0, T, dtype = torch.long, device=tokens.device)
    time_indices = time_indices[:, None]
    time_indices = time_indices.repeat([1, n_tokens])

    sel_tokens = tokens[indices[:, 0], indices[:, 1]].long()
    sel_times = time_indices[indices[:, 0], indices[:, 1]]

    sel_tokens -= 1

    pianoroll = torch.zeros([T, 256], dtype = torch.long, device=tokens.device)
    pianoroll[sel_times, sel_tokens] = 1

    return pianoroll


    

def pianoroll_to_notes_list(onset_pianoroll: torch.long, offset_pianoroll: torch.long):
    """_summary_

    Args:
        onset_pianoroll (torch.long): shape [T, F]
        offset_pianoroll (torch.long): shape [T, F]
    Returns:
        intervals (np.array): shape [n_notes, 2]
        pitches (np.array): shape [n_notes,]
    """
    # => [F, T+1]
    on_pianoroll = Functional.pad(onset_pianoroll.transpose(0, 1), [0, 1]) #  
    off_pianoroll = Functional.pad(offset_pianoroll.transpose(0, 1), [0, 1]) #  
    
    # on_pianoroll[:, 1:] = torch.clip(on_pianoroll[:, 1:] - on_pianoroll[:, :-1], 0, 1) # remove duplicated onsets (choose the first one).
    # Only remain the first one.
    on_pianoroll[:, 1:] = torch.clip(on_pianoroll[:, 1:] - on_pianoroll[:, :-1], 0, 1) # for onset
    off_pianoroll[:, 1:] = torch.clip(off_pianoroll[:, 1:] - off_pianoroll[:, :-1], 0, 1) # for offset
    off_pianoroll[on_pianoroll== 1] = 1 # For a pitch, add offset to the same frame if there is onset in this frame. Ensure there is alway an offset before an onset.
    off_pianoroll[:, -1] = 1 # Add off to the end, ensure there alway are offsets after onsets.

    # To avoid empty list.
    if on_pianoroll.nonzero().sum() == 0:
        intervals = np.zeros([0, 2],dtype=int)
        notes = np.zeros([0], dtype=int)
        return intervals, notes

    notes_lst = [] # [onset, pitch, offset]
    
    for pit in range(128):
        onsets = on_pianoroll[pit:pit+1, :]
        offsets = off_pianoroll[pit:pit+1, :]
        # => [2, T] => [T, 2]
        off_on = torch.cat([offsets, onsets], dim=0).T # offsets are ahead of onsets in the same frame.
        # => [n, 2], n is num of nonzero; [i, 0]==0:off; [i, 0]==1:on; [i, 1] are time indices.
        indices = torch.nonzero(off_on)
        off_on_list = indices[:, 1] # [n_nonzero], val=0:off, val=1:on.
        time_list = indices[:, 0] # [n_nonzero]
        if len(indices) <= 1: # offsets are added to the end for each pitch, so the len must be >=2.
            continue
        onset_mask = off_on_list == 1
        onset_times = time_list[onset_mask] #[n_nonzero]
        offset_times = torch.roll(time_list, shifts=-1, dims=0)[onset_mask] # There must be one offset after onset, so just need to shift left for 1 step.
        pitchs = torch.ones_like(onset_times) * pit
        notes_pit = torch.stack([onset_times, offset_times, pitchs], dim = 1)
        notes_lst.append(notes_pit)

    if len(notes_lst) == 0:
        return np.zeros([0, 2]), np.zeros([0])

    # >= [n_notes, 3]
    notes = torch.cat(notes_lst, dim=0).cpu().numpy()
    df = pd.DataFrame(notes, columns=["onset", "offset", "pitch"])
    df = df.sort_values(by=["onset", "pitch"])
    notes_sorted = df.to_numpy()
    # notes = np.sort(notes, axis=1, kind="stable") # sort compare by onset first, then pitch, last offset.

    intervals = notes_sorted[:, :2]
    pitches = notes_sorted[:, 2]

    # # for dur longer than 100 frames, reset to 1.
    # dur = intervals[:, 1] - intervals[:, 0]
    # mask = dur > 100 
    # intervals[:, 1][mask] = intervals[:, 0][mask] + 100 # set notes longer than 2 seconds to 2s.

    # mask = dur <= 100 # Ignore notes longer than 2 seconds.
    # intervals = intervals[mask]
    # pitches = pitches[mask]

    return intervals, pitches

def frame_tokens_to_interval_and_pitch(frame_tokens):
    """
    inputs:
        frame_tokens(torch.array.long): [T, n_tokens]
    outputs:
        intervals(ndarray.long): [n_notes, 2] onset and offset time in frame index.
        pitches(ndarray.long): [n_notes]
    """
    frame_tokens = torch.clone(frame_tokens)
    
    # => [BT, 256]
    on_off_pianoroll = frame_tokens_to_pianoroll(frame_tokens)
    # => [BT+1, 256]
    # on_off_pianoroll = Functional.pad(on_off_pianoroll, [0, 0, 0, 1]) #  
    on_pianoroll = on_off_pianoroll[:, :128]
    off_pianoroll = on_off_pianoroll[:, 128:256]
    
    return pianoroll_to_notes_list(on_pianoroll, off_pianoroll)


def notes_to_pianoroll(intervals, pitches, n_frames = None):
    """
        To note piano roll (with duration)
        args:
            intervals(np.long): [n_notes, 2] onsets and offsets time, frame indexs.
            pitches(np.long): 0~127.
        returns:
            pianoroll(np.array): [128, n_frames]
    """
    if n_frames is None:
        n_frames = intervals.max() + 1
    pianoroll = np.zeros([128, n_frames], dtype=np.float64)
    # torch.split
    n_notes = intervals.shape[0]
    for i in range(n_notes):
        pit = pitches[i]
        on = intervals[i, 0]
        off = intervals[i, 1]
        pianoroll[pit, on:off] = 1
    return pianoroll

    
if __name__=="__main__":
    # test frame_tokens_to_interval_and_pitch
    frame_tokens = torch.tensor([
        [2,3,0,127,128,129, 255, 256, 257, 562],
        [0,0,128, 3,4,131, 132, 444,555,666],
        [130, 128, 0, 0, 0, 1, 2, 257,257,257]
    ])
    intervals, pitches = frame_tokens_to_interval_and_pitch(frame_tokens)




    # test notes_to_pianoroll
    # intervals = np.array([
    #     [0, 3],
    #     [2,3],
    #     [4,5],
    # ])
    # pitches = np.array([
    #     8,
    #     12,
    #     32
    # ])
    pianoroll = notes_to_pianoroll(intervals, pitches)
    pianoroll