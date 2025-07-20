import torch
import pretty_midi
from data.constants import *
from data.notes_parser import frame_tokens_to_interval_and_pitch
import numpy as np
import partitura

from symusic import Score, TimeUnit

def get_notes_with_pedal(midi_path):
    """_summary_
    Args:
        midi_path (_type_): _description_
    Returns:
        notes_dict: 
            onset: np.array
            offset: 
            duration:
            velocity:
    """
    notes = {}
    score = Score(midi_path, ttype=TimeUnit.second)
    assert len(score.tracks) == 1
    track = score.tracks[0]
   
    notes = track.notes.numpy()
    

    notes["onset"] = notes["time"]
    notes["offset"] = notes["time"] + notes["duration"]
    notes["pitch"] = notes["pitch"]

    # # Remove notes longer than 3 seconds.
    # mask = notes["duration"] <= 4.0 # remove notes longer than 2 seconds.
    # notes["onset"] = notes["onset"][mask]
    # notes["offset"] = notes["offset"][mask]
    # notes["pitch"] = notes["pitch"][mask]

    # mid = pretty_midi.PrettyMIDI(midi_path)

    # Modify offsets according to pedal.
    pedals = track.pedals.numpy()
    pedals["end"] = pedals["time"] + pedals["duration"]

    # # set pedal threshold
    # pd_threshold = 65
    # controls = track.controls.numpy()
    # pd_mask = controls["number"] == 64 # pedal number is 64.
    # pedals["pedal_change_time"] = controls["time"][pd_mask]
    # pedals["pedal_change_value"] = controls["value"][pd_mask]
    # pd_time_prev = 0
    # new_pd_times = []
    # new_pd_ends = []
    # pedal_status = False
    # for time, val in  zip(pedals["pedal_change_time"], pedals["pedal_change_value"] ):
    #     if (val > pd_threshold) == pedal_status: # Pedal status not changed.
    #         continue
    #     pedal_status = val > pd_threshold
    #     if pedal_status == False: # Pedal off
    #         new_pd_times.append(pd_time_prev)
    #         new_pd_ends.append(time)
    #     pd_time_prev = time
    # pedals["time"] = np.array(new_pd_times)
    # pedals["end"] = np.array(new_pd_ends)



    notes["offset_pedal"] = np.copy(notes["offset"])
    for pd_start, pd_end in zip(pedals["time"], pedals["end"]):
        mask = (notes["offset"] >= pd_start) & (notes["offset"] <= pd_end)
        notes["offset_pedal"][mask] = pd_end # np.round(pd_end * 5) / 5 # 200ms # np.round(pd_end, decimals=1) # round to 1 decimals to avoid float precision issues.

        # note cross pedal off and dur > 0.3 when pedal off.
        # mask = (notes["onset"] < pd_end) & (notes["offset"] > pd_end) & (pd_end - notes["onset"] >= 0.4)
        # notes["offset_pedal"][mask] = pd_end

    
    notes["offset"] = notes["offset_pedal"]
    notes["duration"] = notes["offset"] - notes["onset"]

    return notes, pedals


def get_onset_pianoroll(pianoroll, onset_type="first"):
    """_summary_

    Args:
        pianoroll (torch.long): shape [..., T, F]

    """
    pianoroll = torch.clone(pianoroll)
    if onset_type == "first":
        pianoroll = torch.clip(pianoroll[..., 1:, :] - pianoroll[..., :-1, :], 0, 1)
    elif onset_type == "last":
        pianoroll = torch.clip(pianoroll[..., :-1, :] - pianoroll[..., 1:, :], 0, 1)
    elif onset_type == "peak":
        raise "Not implement yet!"
    else:
        raise "Unknow type:" + onset_type
    return pianoroll 


def onset_pianoroll_to_note_list(onset_pianoroll: torch.long):
    """_summary_

    Args:
        onset_pianoroll (_type_): shape [T, F] 
    Returns:
        onsets_list:
        times_list:
    """
    T, F = onset_pianoroll.size()
    indices = torch.nonzero(onset_pianoroll)

    time_indices = torch.arange(0, T, dtype = torch.long, device=onset_pianoroll.device)
    time_indices = time_indices[:, None]
    time_indices = time_indices.repeat([1, F])

    freq_indices = torch.arange(0, F, dtype = torch.long, device=onset_pianoroll.device)
    freq_indices = freq_indices[None, :]
    freq_indices = freq_indices.repeat([T, 1])

    mask = onset_pianoroll == 1
    times_list = torch.masked_select(time_indices, mask).cpu().numpy()
    notes_list = torch.masked_select(freq_indices, mask).cpu().numpy()

    return times_list, notes_list



def save_notes_to_midi(file_path, notes_dict: dict):
    """_summary_

    Args:
        notes_dict (dict): "onset": np.array, "offset": np.array, "note": np.array
    """
    note_seq = []
    control_change_seq = []
    onset_list = notes_dict["onset"]
    offset_list = notes_dict["offset"]
    note_list = notes_dict["note"]
    pedal_offs = notes_dict["pedal_off"]
    for on, off, note in zip(onset_list, offset_list, note_list):
        # if off - on > 5: # When duration > 0.5 seconds, set the offset to the nearest pedal off location.
        #     pedal_offs_sel = pedal_offs[pedal_offs > on]
        #     if len(pedal_offs_sel) > 0:
        #         off = min(off, pedal_offs_sel[0])


        note = pretty_midi.Note(velocity=80, pitch=note, start=on, end=off)
        note_seq.append(note)
    # Pedal
    prev_off = 0
    for pd_off in pedal_offs:
        # pedal on
        pd_on = pd_off - 0.005 # pedal on is hard to recognize, so only pedal offsets are useful.
        control = pretty_midi.ControlChange(number=64, value=127, time=pd_on)
        control_change_seq.append(control)
        # pedal off
        control = pretty_midi.ControlChange(number=64, value=0, time=pd_off)
        control_change_seq.append(control)
        prev_off = pd_off

    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instrument = pretty_midi.Instrument(1, False, "Piano")
    instrument.notes = note_seq
    instrument.control_changes = control_change_seq
    # instrument.control_changes = 

    mid.instruments.append(instrument)
    if file_path is not None:
        mid.write(file_path)



