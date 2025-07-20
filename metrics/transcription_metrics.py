import os
import sys
work_dir = os.path.split(__file__)[0] + "/../"
import torch
import numpy as np
import data.notes_parser as notes_parser
from data.constants import *
from symusic import Score, TimeUnit
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn import metrics


from mir_eval.util import midi_to_hz
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes

import utils.pianoroll_parser as pianoroll_parser

from data.notes_parser import frame_tokens_to_interval_and_pitch

def pianoroll_frame_metrics(name, output_pianoroll_frame, target_pianoroll_frame):
    if (target_pianoroll_frame == 1).sum() == 0:
        return {}
    assert target_pianoroll_frame.max() <= 1 and target_pianoroll_frame.min() >= 0
    assert output_pianoroll_frame.max() <= 1 and output_pianoroll_frame.min() <= 0
    P = (target_pianoroll_frame == 1).sum()
    output_pianoroll_frame = (output_pianoroll_frame == 1).long()
    target_pianoroll_frame = (target_pianoroll_frame == 1).long()
    TP = torch.mul(output_pianoroll_frame, target_pianoroll_frame).sum()
    FP = torch.mul(output_pianoroll_frame, 1 - target_pianoroll_frame).sum()
    p = 0
    if TP + FP > 0:
        p = TP / (FP + TP)
    r = TP/P
    f1 = 0
    if p+r > 0:
        f1 = 2 * p * r /(p + r)
    
    return {
        name + "_p": p,
        name + "_r": r,
        name + "_f1": f1,
    }


def pianoroll_onset_metrics(name, output_pianoroll_onset, target_pianoroll_onset, threshold = 0.5, times_interval = 0.02):
    """_summary_

    Args:
        output_pianoroll_onset (_type_): shape [T, F] or [B, T, F]
        target_pianoroll_onset (_type_): shape [T, F] or [B, T, F]
    """
    if output_pianoroll_onset.ndim == 3:
        B,T,F = output_pianoroll_onset.size()
        output_pianoroll_onset = output_pianoroll_onset.reshape([B*T, F])
        target_pianoroll_onset = target_pianoroll_onset.reshape([B*T, F])
    output_pianoroll_onset = (output_pianoroll_onset >= threshold).long()

    output_pianoroll_onset = pianoroll_parser.get_onset_pianoroll(output_pianoroll_onset)
    target_pianoroll_onset = pianoroll_parser.get_onset_pianoroll(target_pianoroll_onset)

    output_times, output_onsets = pianoroll_parser.onset_pianoroll_to_note_list(output_pianoroll_onset)
    target_times, target_onsets = pianoroll_parser.onset_pianoroll_to_note_list(target_pianoroll_onset)
    output_times = np.array(output_times) * times_interval
    target_times = np.array(target_times) * times_interval
    output_hz = midi_to_hz(output_onsets)
    target_hz = midi_to_hz(target_onsets)

    num = len(target_onsets)

    output_intervals = np.stack([output_times, output_times + 0.2], axis=1)
    target_intervals = np.stack([target_times, target_times + 0.2], axis=1)
    p, r, f, o = evaluate_notes(target_intervals, target_hz, output_intervals, output_hz, offset_ratio=None, onset_tolerance=0.05)
    print("%s: p=%.3f, r=%.3f, f1=%.3f, n=%d"%(name, p, r, f, num))
    return {
        name + "_p": p,
        name + "_r": r,
        name + "_f1": f,
    }


def cal_frames_token_metrics(output_tokens, target_tokens, decoder_targets_tokens):
    """_summary_

    Args:
        output_tokens (torch.tensor.long): [B, n_tokens], n_tokens is the max token seq length.
        target_tokens (torch.tensor.long): [B, n_tokens]
        targets_delta_tokens (torch.tensor.long): [B, n_tokens]
    Returns:
        metric_dict (Dict):
    """
    metrics_dict = {}
    
    output_tokens = torch.clone(output_tokens)
    # self.log_time_event("metrics_cal_begin")
    B, n_tokens = output_tokens.size()

    # # set tokens after eos to TOKEN_PAD
    # eos_mask = torch.zeros([B], device=output_tokens.device).bool()
    # for i in range(n_tokens):
    #     output_tokens[eos_mask, i] = TOKEN_PAD
    #     eos_mask = eos_mask | (output_tokens[:, i] == TOKEN_END)
    #     # eos_mask = eos_mask | (output_tokens[:, i] >= 256) 

    output_onset_tokens = torch.clone(output_tokens)
    # output_onset_tokens[(output_onset_tokens > MIDI_MAX) | (output_onset_tokens < MIDI_MIN)] = TOKEN_BLANK
    
    output_intervals, output_pitches = frame_tokens_to_interval_and_pitch(output_onset_tokens)
    # notes_dict["output_intervals"] = output_intervals
    # notes_dict["output_pitches"] = output_pitches
    target_intervals, target_pitches = frame_tokens_to_interval_and_pitch(target_tokens)
    # notes_dict["target_intervals"] = target_intervals
    # notes_dict["target_pitches"] = target_pitches
    # => [128, BT]

    # cal metrics
    # frame idx to second.
    output_intervals_second = output_intervals * DEFAULT_HOP_WIDTH/DEFAULT_SAMPLE_RATE
    target_intervals_second = target_intervals * DEFAULT_HOP_WIDTH/DEFAULT_SAMPLE_RATE
    output_pitches_hz = midi_to_hz(output_pitches)
    target_pitches_hz = midi_to_hz(target_pitches)
    if len(target_pitches) > 0 and len(output_pitches) > 0:
        p, r, f, o = evaluate_notes(target_intervals_second, target_pitches_hz, output_intervals_second, output_pitches_hz, offset_ratio=None, onset_tolerance=0.05)
        # self.log_time_event("metrics_cal_done")
        num = len(target_pitches)
        
        metrics_dict["onset_p"] = p
        metrics_dict["onset_r"] = r
        metrics_dict["onset_f1"] = f
        metrics_dict["onset_num"] = num
        print("Onset: p=%.3f, r=%.3f, f1=%.3f, n=%d, n_tokens=%d"%(p,r,f,num, n_tokens))

        p, r, f, o = evaluate_notes(target_intervals_second, target_pitches_hz, output_intervals_second, output_pitches_hz, onset_tolerance=0.05)
        metrics_dict["note_p"] = p
        metrics_dict["note_r"] = r
        metrics_dict["note_f1"] = f
        print("Note: p=%.3f, r=%.3f, f1=%.3f"%(p,r,f))
    else:
        if len(target_pitches) == 0:
            print("Warning: len(target_pitches)=0.")
        else:
            print("Warning: len(output_pitches)=0.")

    # cal acc of tokens.
    output_tokens = output_tokens.view(-1)#.cpu().numpy()
    target_tokens = target_tokens.view(-1)#.cpu().numpy()
    decoder_targets_tokens = decoder_targets_tokens.view(-1)#.cpu().numpy()
    def event_acc(token_type, begin, end, output_tokens, target_tokens):
        # sel_idx = (target_tokens >= begin) &  (target_tokens < end)
        # output_tokens_sel = output_tokens[sel_idx]
        # target_tokens_sel =target_tokens[sel_idx]
        # if len(target_tokens_sel) == 0:
        #     return 0
        # return accuracy_score(target_tokens_sel, output_tokens_sel)
        sel_targets = ((target_tokens >= begin) &  (target_tokens < end))
        sel_outputs = ((output_tokens >= begin) & (output_tokens < end))
        num = sel_targets.sum()
        if num == 0:
            return {}
        TP = (output_tokens[sel_targets & sel_outputs] == target_tokens[sel_targets & sel_outputs]).sum()

        FP = sel_outputs.sum() - TP
        if TP + FP == 0:
            precision= 0
        else:
            precision = TP/(TP+FP)
        recall = TP / num
        if (precision + recall) <= 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        d =  {
            "token_%s_p"%token_type:precision,
            "token_%s_r"%token_type:recall,
            "token_%s_f1"%token_type:f1,
        }
        if token_type == "on":
            print("Token_on: p=%.3f, r=%.3f, f1=%.3f"%(precision,recall,f1))
        return d

    metrics_dict.update( event_acc("all", 0, TOKEN_PAD, output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("on", 0, START_IDX["note_off"], output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("off", START_IDX["note_off"], START_IDX["time_shift"], output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("time", START_IDX["time_shift"], START_IDX["time_shift"]+128, output_tokens, decoder_targets_tokens) )
    metrics_dict.update( event_acc("eos", TOKEN_END, TOKEN_END+1, output_tokens, decoder_targets_tokens) )

    return metrics_dict

def cal_onsets_and_frames_metrics(name, output_onset_pianoroll: torch.tensor, output_frame_pianoroll, target_onset_pianoroll, target_frame_pianoroll):
    """_summary_

    Args:
        output_onset_pianoroll (_type_): shape [B, T, F] binary value: 0, 1
        output_frame_pianoroll (_type_): _description_
        target_onset_pianoroll (_type_): _description_
        target_frame_pianoroll (_type_): _description_
    """
    assert target_frame_pianoroll.max() <= 1 and target_frame_pianoroll.min() >= 0
    assert output_onset_pianoroll.ndim == 3
    def get_offset_pianoroll(frame_pianoroll):
        offset_pianoroll  = torch.clone(frame_pianoroll)
        offset_pianoroll[:, : -1, :] = torch.clip(frame_pianoroll[:, :-1, :] - frame_pianoroll[:, 1:, :], 0, 1)
        offset_pianoroll[:, -1, :] = 1 # set the last frame as offsets.
        return offset_pianoroll

    output_offset_pianoroll = get_offset_pianoroll(output_frame_pianoroll)
    target_offset_pianoroll = get_offset_pianoroll(target_frame_pianoroll)
    B, T, F = output_onset_pianoroll.size()
    output_onset_pianoroll = output_onset_pianoroll.reshape([B*T, F])
    output_offset_pianoroll = output_offset_pianoroll.reshape([B*T, F])
    target_onset_pianoroll = target_onset_pianoroll.reshape([B*T, F])
    target_offset_pianoroll = target_offset_pianoroll.reshape([B*T, F])

    output_intervals, output_pitches_midi = notes_parser.pianoroll_to_notes_list(output_onset_pianoroll, output_offset_pianoroll)
    target_intervals, target_pitches_midi = notes_parser.pianoroll_to_notes_list(target_onset_pianoroll, target_offset_pianoroll)
    output_intervals = output_intervals * DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
    target_intervals = target_intervals * DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
    output_pitches = midi_to_hz(output_pitches_midi)
    target_pitches = midi_to_hz(target_pitches_midi)
    metrics_dict = {}
    dur_mask = output_intervals[:, 1] < output_intervals[:, 0] + 0.05 # Ensure the min duration is 50ms.
    output_intervals[:, 1][dur_mask] = (output_intervals[:, 0] + 0.05)[dur_mask]
    p, r, f, o = evaluate_notes(target_intervals, target_pitches, output_intervals, output_pitches, onset_tolerance=0.05)
    metrics_dict["%s_p"%name] = p
    metrics_dict["%s_r"%name] = r
    metrics_dict["%s_f1"%name] = f
    


    # Duration Error Analysis
    acc = -1
    err_dur_longer = -1
    if len(output_pitches_midi) > 0 and\
            len(output_pitches_midi) == len(target_pitches_midi) and\
            (output_pitches_midi != target_pitches_midi).sum() == 0:
        output_durs = output_intervals[:, 1] - output_intervals[:, 0]
        target_durs = target_intervals[:, 1] - target_intervals[:, 0]
        errors = target_durs - output_durs
        thresholds = np.clip(target_durs * 0.2, a_min = 0.05, a_max=100000)
        error_num = (np.abs(errors) > thresholds).sum()
        acc = 1 - error_num / len(output_pitches_midi)
        error_durs = target_durs[np.abs(errors) > thresholds]
        error_durs_delta = errors[np.abs(errors) > thresholds]
        err_dur_longer = (errors < -thresholds).sum() / error_num
        metrics_dict["%s_error_durs_list"%name] = error_durs
        metrics_dict["%s_error_durs_list_delta"%name] = error_durs_delta


    print("%s: p=%.3f, r=%.3f, f1=%.3f, acc=%.3f, err_dur_longer=%.3f"%(name, p,r,f, acc, err_dur_longer))


    return metrics_dict

def cal_notes_sustain_metrics(output_tokens, output_onset_pianoroll, target_onset_pianoroll, target_frame_pianoroll):
    """_summary_

    Args:
        output_tokens (_type_): shape [B*T, n_tokens]
        output_onset_pianoroll: [B, T, F]
        target_onset_pianoroll (_type_): [B, T, F]
        target_frame_pianoroll (_type_): [B, T, F]
    """
    # => [B*T, 256]
    output_on_off_pianoroll = notes_parser.frame_tokens_to_pianoroll(output_tokens)
    output_frame_pianoroll = output_on_off_pianoroll[:, :128]
    # Add offset to frame pianoroll
    output_offset_pianoroll = output_on_off_pianoroll[:, 128:256]
    # output_frame_pianoroll[(output_frame_pianoroll == 1) & (output_offset_pianoroll == 1) ] = 0
    # output_frame_pianoroll = output_offset_pianoroll
    

    B, T, F = target_onset_pianoroll.size()
    output_frame_pianoroll = output_frame_pianoroll.reshape([B, T, F])

    return cal_onsets_and_frames_metrics("Sustain", output_onset_pianoroll, output_frame_pianoroll, target_onset_pianoroll, target_frame_pianoroll)
    
    






def cal_decoder_token_metrics(output_tokens, target_tokens, metric_prefix=""):
    """_summary_

    Args:
        output_tokens (torch.tensor.long): [B, n_tokens], n_tokens is the max token seq length.
        target_tokens (torch.tensor.long): [B, n_tokens]
        target_tokens_mask (torch.tensor.long): [B, n_tokens]
    Returns:
        metric_dict (Dict):
    """
    metrics_dict = {}
    
    output_tokens = torch.clone(output_tokens)
    # self.log_time_event("metrics_cal_begin")
    # B, n_tokens = output_tokens.size()

    # # set tokens after eos to TOKEN_PAD
    # eos_mask = torch.zeros([B], device=output_tokens.device).bool()
    # for i in range(n_tokens):
    #     output_tokens[eos_mask, i] = TOKEN_PAD
    #     eos_mask = eos_mask | (output_tokens[:, i] == TOKEN_END)
    #     # eos_mask = eos_mask | (output_tokens[:, i] >= 256) 


    # cal metrics

    # cal acc of tokens.
    output_tokens = output_tokens.view(-1)#.cpu().numpy()
    target_tokens = target_tokens.view(-1)#.cpu().numpy()
    def event_acc(token_type, begin, end, output_tokens, target_tokens):
        # sel_idx = (target_tokens >= begin) &  (target_tokens < end)
        # output_tokens_sel = output_tokens[sel_idx]
        # target_tokens_sel =target_tokens[sel_idx]
        # if len(target_tokens_sel) == 0:
        #     return 0
        # return accuracy_score(target_tokens_sel, output_tokens_sel)
        if end - begin == 1:
            sel_targets = ((target_tokens >= begin) &  (target_tokens < end))
            sel_outputs = ((output_tokens >= begin) & (output_tokens < end))
            P = sel_targets.sum()
            if P == 0:
                return {}
            TP = (output_tokens[sel_targets & sel_outputs] == target_tokens[sel_targets & sel_outputs]).sum()

            FP = sel_outputs.sum() - TP
            if TP + FP == 0:
                precision= 0
            else:
                precision = TP/(TP+FP)
            recall = TP / P
            if (precision + recall) <= 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            d =  {
                "%stoken_%s_p"%(metric_prefix, token_type): float(precision),
                "%stoken_%s_r"%(metric_prefix, token_type): float(recall),
                "%stoken_%s_f1"%(metric_prefix, token_type): float(f1),
            }
        else:
            sel_targets = ((target_tokens >= begin) &  (target_tokens < end))
            num = sel_targets.sum()
            if num == 0:
                return {}
            TP = (output_tokens[sel_targets] == target_tokens[sel_targets]).sum()
            acc = TP / num
            d =  {
                "%stoken_%s_acc"%(metric_prefix, token_type): float(acc),
            }
        return d
    
    metrics_dict.update( event_acc("all", 0, TOKEN_PAD, output_tokens, target_tokens) )
    for token_type in sm_tokenizer.token_type_list:
        name = token_type.get_type_name()
        begin, end = token_type.get_bound()
        metrics_dict.update( event_acc(name, begin, end, output_tokens, target_tokens) )
    
    # EOS
    begin = sm_tokenizer.EOS
    end = begin + 1
    metrics_dict.update( event_acc("EOS", begin, end, output_tokens, target_tokens) )

    return metrics_dict


def cal_multihot_notes_metrics(name, decoder_outputs_note, decoder_targets_note):
    """_summary_

    Args:
        decoder_outputs_note (torch.tensor.float): [B, n_tokens, 128], n_tokens is the max token seq length.
        decoder_targets_note (torch.tensor.long): [B, n_tokens, 128]
    Returns:
        metric_dict (Dict):
    """
    B, n_tokens, n_notes = decoder_targets_note.size()
    decoder_outputs_note = (torch.sigmoid(decoder_outputs_note) > 0.4).long()
    assert (decoder_targets_note == 0).long().sum() + (decoder_targets_note == 1).long().sum() == B * n_tokens * n_notes
    note_num = decoder_targets_note.sum()
    note_p = 0
    note_r = 0
    note_f = 0
    if note_num > 0:
        TP = (decoder_targets_note * decoder_outputs_note).sum()
        FP = ((1-decoder_targets_note) * decoder_outputs_note).sum()
        note_r =  TP / note_num
        if TP + FP > 0:
            note_p = TP/(TP+FP)
        if TP > 0:
            note_f = 2*note_r*note_p /(note_p + note_r)
            
    return {
        name + "_p": note_p,
        name + "_r": note_r,
        name + "_f1": note_f,
    }


def get_intervals_notes(midi_path):
    notes, pedals = pianoroll_parser.get_notes_with_pedal(midi_path)
    onsets = notes["onset"]
    offsets = notes["offset"]
    pit = notes["pitch"]
    # Remove overlap.
    overlap_num = 0
    for p in range(128):
        mask = (pit == p)
        num = mask.sum()
        if  num > 0:
            overlap = (offsets[mask][:-1] > onsets[mask][1:]).astype(int).sum()
            if overlap > 0:
                # print(overlap)
                overlap_num += overlap
                sel_offset = offsets[mask]
                offsets[mask] = np.concatenate( [np.minimum(sel_offset[:-1], onsets[mask][1:]) , sel_offset[-1:] ], axis=0 ) # Replace inplace.
    print("overlap_num:", overlap_num)
    intervals = np.stack([onsets, offsets], axis=1)
    return intervals, pit, pedals


def cal_midi_files_metrics(est_midi_paths: list, ref_midi_paths: list, result_path):
    result_dict = defaultdict(list)
    for est_path, ref_path in tqdm(zip(est_midi_paths, ref_midi_paths)):
        est_intervals, est_notes, est_pedals = get_intervals_notes(est_path)
        ref_intervals, ref_notes, ref_pedals = get_intervals_notes(ref_path)

        est_intervals += 0.01

        est_intervals[:, 1] = np.maximum(est_intervals[:, 1], est_intervals[:, 0] + 0.05) #min dur >= 0.05


        est_pitches = midi_to_hz(est_notes)
        ref_pitches = midi_to_hz(ref_notes)

        # Frames
        # evaluate_frames()

        # Onsets
        p, r, f, o = evaluate_notes(ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None, onset_tolerance=0.05)
        result_dict["onset_precision"].append(p*100)
        result_dict["onset_recall"].append(r*100)
        result_dict["onset_f1"].append(f*100)
        
        # Notes
        p, r, f, o = evaluate_notes(ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=0.05)
        result_dict["note_precision"].append(p*100)
        result_dict["note_recall"].append(r*100)
        result_dict["note_f1"].append(f*100)
        result_dict["note_num"].append(len(ref_pitches))

        # ref_dur = ref_intervals[:, 1] - ref_intervals[:, 0]
        # est_dur = est_intervals[:, 1] - est_intervals[:, 0]
        # errors = np.abs(ref_dur - est_dur)
        # thresholds = np.clip(ref_dur * 0.2, a_min=0.05, a_max=99999999)
        # acc = (errors <= thresholds).sum() / len(ref_dur)


        result_dict["midi_ref"].append(os.path.split(ref_path)[1])
        result_dict["midi_est"].append(os.path.split(est_path)[1])
    # p_avg = np.mean(result_dict["onset_precision"])
    # result_dict["onset_precision"].append(p_avg)
    # r_avg = np.mea
    for k,v in result_dict.items():
        if not k.startswith("midi"):
            avg = np.mean(v)
            result_dict[k].append(avg)
        else:
            result_dict[k].append("Avg")

    df = pd.DataFrame(result_dict)
    df.to_csv(result_path, float_format="%.2f")
    # print(df.tail(10))
    

