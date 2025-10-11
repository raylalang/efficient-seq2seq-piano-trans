import music21
from typing import List, Dict, Union, Optional
from enum import Enum
import numpy as np
import json
import pandas as pd
import pretty_midi
import symusic
from collections import defaultdict
from utils.pianoroll_parser import get_notes_with_pedal


class TokenType(Enum):
    FAMILY = 0
    TEMPO = 1
    TIME_SIG_NUM = 2
    TIME_SIG_DENUM = 3
    ONSET_SEC = 4
    PITCH = 5
    DUR_TATUM = 6
    VELOCITY = 7
    BEAT = 8
    STAFF = 9
    VOICE = 10
    ONSET_IN_BAR = 11
    # PROGRAM = 5
    # CHORD = 6
    # BAR_POSITION = 1
    # REST = 7
    # Pedal


class SequenceType(Enum):
    MIDI_LIKE = 0
    COMPOUND_WORD = 1


# SEQUENCE_TYPE = SequenceType.MIDI_LIKE
SEQUENCE_TYPE = SequenceType.COMPOUND_WORD


class TokenSize(Enum):
    FAMILY = 5  # Measure, Note, EOS, BOS, PAD
    TEMPO = 64
    TIME_SIG_NUM = 16
    TIME_SIG_DENUM = 16
    # ONSET_SEC = 360  # 360 for dur <= 18s, 600 for dur <= 30s
    ONSET_SEC = 600  # 360 for dur <= 18s, 600 for dur <= 30s
    PITCH = 128
    NOTE_OFF = 128
    DUR_TATUM = 128  # max note dur: 2 whole notes.
    VELOCITY = 64
    BEAT = 2  # 0: not beat, 1: beat
    DOWNBEAT = 2  # 0: not downbeat, 1: downbeat
    STAFF = 2
    VOICE = 4
    ONSET_IN_BAR = 256
    ALIGNMENT = 3  # 0: match, 1: insertion, 2: deletion.
    PEDAL = 2  # 0: PEDAL_OFF, 1: PEDAL_ON


TEMPO_DOWN_SAMPLING = 5
ONSET_SEC_UP_SAMPLING = 20  # 12 # 6 #4
DUR_TATUM_UP_SAMPLING = 12  # quarter * 12
ONSET_IN_BAR_UP_SAMPLING = 12
VELOCITY_DOWN_SAMPLING = 2
BEAT_UP_SAMPLING = 2


# name, size, begin, end
class BaseToken:
    def __init__(self, value: int):
        assert value >= 0
        self.value = value


family_name_dict = {0: "Measure", 1: "Note", 2: "EOS", 3: "BOS", 4: "PAD"}


class FamilyType(Enum):
    MEASURE = 0
    NOTE = 1
    BOS = 2
    EOS = 3
    PAD = 4


assert len(family_name_dict) == TokenSize.FAMILY.value


class TokenBase:
    def __init__(self, value: int):
        if not value < self.__class__.__size:
            print(
                "Token value out of bound: %s (%d), bound: (0, %d)"
                % (self.__class__.name, value, self.__class__.__size)
            )
        assert value >= 0 and value < self.__class__.__size
        self.value = value

    # Sub Class
    # 在子类定义时调用
    def __init_subclass__(cls):
        cls.begin_index = None
        cls.__size = None
        cls.cp_index = None  # the position of token type in compound word

    @classmethod
    def set_begin(cls, begin_index: int):
        assert cls.begin_index is None  # 确保只被赋值一次
        cls.begin_index = begin_index

    @classmethod
    def set_size(cls, size: int):
        assert cls.__size is None
        cls.__size = size

    @classmethod
    def get_size(cls) -> int:
        return cls.__size

    @classmethod
    def get_type_name(cls) -> str:
        return cls.name

    @classmethod
    def get_bound(cls) -> tuple:
        begin = cls.begin_index
        size = cls.__size
        return (begin, begin + size)

    @classmethod
    def get_value(cls, token_val: int) -> int:
        value = token_val - cls.begin_index
        if (value < 0) or (value >= cls.__size):
            print(
                "Token value out of bound: %s (%d, %d), bound: (%d, %d)"
                % (cls.get_type_name(), token_val, value, cls.begin_index, cls.__size)
            )
            np.clip(value, 0, cls.__size - 1)
        return int(value)

    @classmethod
    def is_instance(cls, token_val: int) -> bool:
        begin, end = cls.get_bound()
        return begin <= token_val < end

    @classmethod
    def set_cp_index(cls, index: int, force=False):
        if force == False:
            assert cls.cp_index is None
        cls.cp_index = index

    ##########################
    # Object Method
    def get_token_val(self) -> int:
        return self.__class__.begin_index + self.value

    def get_name(self) -> str:
        return self.__class__.name + "_%d" % self.value

    @classmethod
    def get_class_name(cls) -> str:
        return str(cls.name)

    def __str__(self):
        return self.get_name()


# 0
class TokenFamily(TokenBase):  # 0
    name = "Family"

    def get_name(self) -> str:
        return family_name_dict[self.value]


class TokenTempo(TokenBase):  # 1
    name = "Tempo"


class TokenTimeSigNum(TokenBase):  # 2
    name = "TSig_N"


class TokenTimeSigDenum(TokenBase):  # 3
    name = "TSig_D"


class TokenOnset(TokenBase):  # 4
    name = "Onset"


class TokenOnsetInBar(TokenBase):
    name = "Onset_InBar"


class TokenPitch(TokenBase):  # 5
    name = "Pitch"


class TokenDur(TokenBase):  # 6
    name = "Dur"


class TokenNoteOff(TokenBase):  # 6
    name = "NoteOff"


class TokenVel(TokenBase):  # 7
    name = "Vel"


class TokenBeat(TokenBase):  # 8
    name = "Beat"


class TokenDownbeat(TokenBase):  # 8
    name = "Downbeat"


class TokenStaff(TokenBase):  # 9
    name = "Staff"


class TokenVoice(TokenBase):  # 10
    name = "Voice"


class TokenAlignment(TokenBase):  # 11
    name = "Alignment"


class TokenPedal(TokenBase):  # 12
    name = "Pedal"


class SymbolicMusicTokenizer:
    def __init__(self):
        self.reverse_type_dict = {}  # token value -> token type
        self.token_type_list: List[TokenBase] = []
        self._initialize_token_dicts()
        # self.EOS = TokenFamily(FamilyType.EOS.value).get_token_val()
        # self.BOS = TokenFamily(FamilyType.BOS.value).get_token_val()
        # self.PAD = TokenFamily(FamilyType.PAD.value).get_token_val()
        # self.TOKEN_MEASURE = TokenFamily(FamilyType.MEASURE.value).get_token_val()
        # self.TOKEN_NOTE = TokenFamily(FamilyType.NOTE.value).get_token_val()
        # self.SEQUENCE_TYPE = SEQUENCE_TYPE
        # self.TOKEN_BEAT = TokenBeat(1).get_token_val() # 0: not beat, 1: beat, 2: downbeat
        # self.TOKEN_DOWNBEAT = TokenDownbeat(1).get_token_val()

        # special tokens
        # BOS
        self.BOS = self.vocab_size
        self.vocab_size = self.vocab_size + 1  # for BOS

        # EOS
        self.EOS = self.vocab_size
        self.vocab_size = self.vocab_size + 1  # for EOS

        # BLANK
        self.BLANK = self.vocab_size
        self.vocab_size = self.vocab_size + 1  # for BLANK

        # PAD
        self.PAD = self.vocab_size
        self.vocab_size = self.vocab_size + 1  # for PAD

        self.TAG_MIDI_TASK = self.vocab_size
        self.vocab_size = self.vocab_size + 1  # for task tag

        self.TAG_SCORE_TASK = self.vocab_size
        self.vocab_size = self.vocab_size + 1  # for task tag

    def get_token_val_types():
        pass

    def _initialize_token_dicts(self):
        """初始化"""

        TokenFamily.set_size(TokenSize.FAMILY.value)
        TokenTempo.set_size(TokenSize.TEMPO.value)
        TokenTimeSigNum.set_size(TokenSize.TIME_SIG_NUM.value)
        TokenTimeSigDenum.set_size(TokenSize.TIME_SIG_DENUM.value)
        TokenOnset.set_size(TokenSize.ONSET_SEC.value)
        TokenPitch.set_size(TokenSize.PITCH.value)
        TokenDur.set_size(TokenSize.DUR_TATUM.value)
        TokenVel.set_size(TokenSize.VELOCITY.value)
        TokenBeat.set_size(TokenSize.BEAT.value)
        TokenDownbeat.set_size(TokenSize.DOWNBEAT.value)
        TokenStaff.set_size(TokenSize.STAFF.value)
        TokenVoice.set_size(TokenSize.VOICE.value)
        TokenOnsetInBar.set_size(TokenSize.ONSET_IN_BAR.value)
        TokenAlignment.set_size(TokenSize.ALIGNMENT.value)
        TokenNoteOff.set_size(TokenSize.NOTE_OFF.value)
        TokenPedal.set_size(TokenSize.PEDAL.value)

        # self.token_type_list.append(TokenTempo)

        self.token_type_list.append(TokenPitch)  # 6
        self.token_type_list.append(TokenOnset)
        # self.token_type_list.append(TokenDur)
        # self.token_type_list.append(TokenOnsetInBar)

        # self.token_type_list.append(TokenStaff)
        # self.token_type_list.append(TokenVoice)

        # self.token_type_list.append(TokenFamily)
        # self.token_type_list.append(TokenTimeSigNum) # 3
        # self.token_type_list.append(TokenTimeSigDenum)
        # self.token_type_list.append(TokenBeat) # 9
        # self.token_type_list.append(TokenDownbeat) # 10
        self.token_type_list.append(TokenNoteOff)
        self.token_type_list.append(TokenVel)
        self.token_type_list.append(TokenPedal)
        # self.token_type_list.append(TokenAlignment) # 1

        self.token_tpye_set = set(self.token_type_list)

        # init begin index for each token type.
        idx = 0
        for cp_index, token_type in enumerate(self.token_type_list):
            token_type.set_begin(idx)
            token_type.set_cp_index(cp_index)
            size = token_type.get_size()
            for i in range(size):
                token = idx + i
                self.reverse_type_dict[token] = token_type
            idx += size

        # Ensure onset is the first token type in the compound word.
        assert self.token_type_list[0] == TokenPitch
        if self.token_type_list[0] == TokenPitch:
            cp_index_pitch = TokenPitch.cp_index
            cp_index_onset = TokenOnset.cp_index
            # Swap pitch and onset index
            TokenPitch.set_cp_index(cp_index_onset, force=True)
            TokenOnset.set_cp_index(cp_index_pitch, force=True)

        self.vocab_size = idx
        if SEQUENCE_TYPE == SequenceType.MIDI_LIKE:
            self.compound_word_size = 1
        elif SEQUENCE_TYPE == SequenceType.COMPOUND_WORD:
            self.compound_word_size = len(self.token_type_list)
            self.midi_like_size = len(self.token_type_list) - 1  # exclude family

    def midi_to_dataframe(
        self, midi_path: str, midi: symusic.Score = None
    ) -> pd.DataFrame:
        """
        Args:
            midi_path
        Returns:
            df_data: pd.DataFrame, with columns: type, type_id, pitch, onset_sec, dur_sec, velocity, program, is_drum, is_chord, is_note
        """
        """将MIDI文件转换为DataFrame"""

        def truncate_note_overlaps(df: pd.DataFrame) -> pd.DataFrame:
            result = []
            for pitch in df["pitch"].unique():
                group = df[df["pitch"] == pitch].copy()
                group = group.sort_values(by="onset_sec").reset_index(drop=True)
                for i in range(len(group) - 1):
                    if group.loc[i, "end_sec"] > group.loc[i + 1, "onset_sec"]:
                        group.loc[i, "end_sec"] = group.loc[i + 1, "onset_sec"]
                result.append(group)
            return pd.concat(result, ignore_index=True)

        if isinstance(midi_path, str):
            midi = symusic.Score(midi_path, ttype=symusic.TimeUnit.second)
        else:
            assert isinstance(
                midi, symusic.Score
            ), "midi should be a symusic.Score object or a path to a MIDI file."

        # iterate through tracks, save track name, is_drum, program, notes to dataframes, and concat dataframes together.
        df_list = []

        # notes, pedals = get_notes_with_pedal(midi_path)

        for track in midi.tracks:
            track.name
            track.is_drum
            track.program

            if False:
                notes = track.notes.numpy()  # convert to numpy array
            else:
                notes, pedals = get_notes_with_pedal(midi_path)

            df = pd.DataFrame(
                {
                    "type": "note",
                    "type_id": 1,  # for sorting. 0 for measure, 1 for note
                    "pitch": notes["pitch"],
                    "onset_sec": notes["time"],
                    "dur_sec": notes["duration"],
                    "offset_sec": notes["time"] + notes["duration"],
                    "velocity": notes["velocity"],
                    "program": 0,  # track.program,
                    "is_drum": 0,  # int(track.is_drum), # 0: not drum, 1: drum
                    # "is_chord": 0,
                    "is_note": 1,
                }
            )
            # df = truncate_note_overlaps(df)
            # df["dur_sec"] = df["end_sec"] - df["onset_sec"]
            df_list.append(df)

        df_data = pd.concat(df_list, ignore_index=True)

        return df_data

    def tokenize_dataframe(
        self, df_data: pd.DataFrame, sequence_type="performance"
    ) -> List[str]:
        """
        Args:
            df_data: pd.DataFrame
            sequence_type: "performance" or "score"
        Returns:
            cp_tokens: [n, m], n is number of compound words, m is the size of compound word.
            seconds: [n]
        """
        """将dataframe转换为标记序列"""
        # df_data["start_index"] = df_data["onset_sec"].apply(lambda x: int(np.round(x * ONSET_SEC_UP_SAMPLING)))
        # df_data["onset_sec"] * ONSET_SEC_UP_SAMPLING
        start_index = (df_data["onset_sec"] * ONSET_SEC_UP_SAMPLING).round().astype(int)

        df_data["start_index"] = start_index
        df_data = df_data.sort_values(
            by=["start_index", "type_id", "pitch", "velocity"],
            ascending=[True, True, True, True],
        )
        # df_data = df_data.sort_values(by=["onset_sec", "type_id", "pitch"], ascending=[True, True, True])
        # df_data = df_data.sort_values(by=["pitch", "start_index"], ascending=[True, True])

        cp_tokens = []
        seconds = []

        PAD_TOKEN = self.PAD

        def append_token(cp_word, token):
            if token.__class__ in self.token_tpye_set:
                # cp_word[token.cp_index] = token.get_token_val()
                cp_word.append(token.get_token_val())

        for index, row in df_data.iterrows():
            cp_word = []
            # Onset Token
            onset_sec = row["onset_sec"]
            onset = row["start_index"]

            # Pedal Off
            if row["type"] == "PedalOff":
                pedal_off_token = TokenPedal(0)  # 0 for pedal off
                append_token(cp_word, pedal_off_token)

            elif row["type"] == "NoteOn" or row["type"] == "NoteOff":
                # Pitch Token
                pitch = row["pitch"]
                if row["type"] == "NoteOn":
                    pitch_token = TokenPitch(pitch)
                    onset_token = TokenOnset(onset)
                    append_token(cp_word, onset_token)
                    append_token(cp_word, pitch_token)

                    # Velocity Token
                    velocity = row["velocity"]
                    velocity_downsample = int(
                        np.floor(velocity / VELOCITY_DOWN_SAMPLING)
                    )
                    vel_token = TokenVel(velocity_downsample)
                    append_token(cp_word, vel_token)
                else:
                    pitch_token = TokenNoteOff(pitch)
                    # onset = int(onset / 2) * 2 # Downsample NoteOff time resolution to 1/2
                    onset_token = TokenOnset(onset)
                    append_token(cp_word, onset_token)
                    append_token(cp_word, pitch_token)
                    # vel_token = TokenVel(0)
                    # append_token(cp_word, vel_token)
                    cp_word.append(self.BLANK)

            cp_tokens.extend(cp_word)
            for _ in range(len(cp_word)):
                seconds.append(onset_sec)

        if len(cp_tokens) == 0:
            print("No tokens found in the dataframe.")
            return np.array([])
        cp_tokens = np.array(cp_tokens, dtype=int)
        return cp_tokens, seconds

    def detokenize(self, tokens: List[str], offsets_sec) -> list:
        """
        Args:
            tokens: numpy array of shape (n), n is number of tokens.
            offsets_sec: float in seconds.
        Returns:
            midi_note_list: List of MIDI note data dictionaries.
        """

        num_tokens = len(tokens)

        midi_event_list = []
        curr_onset_sec = offsets_sec
        prev_velocity = 64
        for i in range(num_tokens):
            token = tokens[i]

            # Onset

            if TokenOnset.is_instance(token):
                curr_onset_sec = (
                    TokenOnset.get_value(token) / ONSET_SEC_UP_SAMPLING + offsets_sec
                )

            # Note On
            elif TokenPitch.is_instance(token):
                pitch = TokenPitch.get_value(token)
                curr_event_data = {
                    "pitch": pitch,
                    "time": curr_onset_sec,
                    "type": "NoteOn",
                    "type_id": 2,  # For sorting
                    "velocity": prev_velocity,
                }
                midi_event_list.append(curr_event_data)

            # Note Off
            elif TokenNoteOff.is_instance(token):
                pitch = TokenNoteOff.get_value(token)
                curr_event_data = {
                    "pitch": pitch,
                    "time": curr_onset_sec,
                    "type": "NoteOff",
                    "type_id": 1,  # For sorting
                    "velocity": 0,
                }
                midi_event_list.append(curr_event_data)

            # Pedal Off
            elif TokenPedal.is_instance(token):
                pedal = TokenPedal.get_value(token)
                pedal_off_event = {
                    "pitch": -1,  # Pedal does not have pitch.
                    "time": curr_onset_sec,
                    "type": "PedalOff" if pedal == 0 else "PedalOn",
                    "type_id": 0,
                    "velocity": 0,  # Pedal does not have velocity.
                }
                midi_event_list.append(pedal_off_event)

                # for pit in range(128):
                #     note_off_event = {
                #         "pitch": pit,
                #         "time": curr_onset_sec,
                #         "type": "NoteOff",
                #         "velocity": 0,
                #     }
                #     midi_event_list.append(note_off_event)

            # Velocity
            elif TokenVel.is_instance(token):
                velocity = TokenVel.get_value(token) * VELOCITY_DOWN_SAMPLING
                if len(midi_event_list) > 0:
                    # Set the velocity of the previous note
                    if (
                        midi_event_list[-1]["velocity"] != 0
                    ):  # Only set if it's a NoteOn event
                        midi_event_list[-1]["velocity"] = velocity
                prev_velocity = velocity

        return midi_event_list

    def notes_to_midi_events(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Args:
            midi_note_list: List of MIDI note data dictionaries.
        Returns:
            midi_event_list: List of MIDI event dictionaries.
        """
        midi_event_list = []

        df_note_on = df[df["type"] == "note"].copy()
        df_note_on["type"] = "NoteOn"
        df_note_on["type_id"] = 2  # For sorting
        df_note_on = df_note_on[["pitch", "onset_sec", "type", "type_id", "velocity"]]

        df_note_off = df[df["type"] == "note"].copy()
        df_note_off["type"] = "NoteOff"
        df_note_off["type_id"] = 1  # For sorting
        df_note_off["onset_sec"] = df_note_off["offset_sec"]
        df_note_off["velocity"] = 0  # NoteOff events have velocity 0
        df_note_off = df_note_off[["pitch", "onset_sec", "type", "type_id", "velocity"]]
        # remove duplicate NoteOff events with the same pitch and offset
        df_note_off = df_note_off.drop_duplicates(subset=["pitch", "onset_sec"])

        # df_pedal_off = df[df["type"] == "pedal"].copy()
        # df_pedal_off["type"] = "PedalOff"
        # df_pedal_off["type_id"] = 3  # For sorting
        # df_pedal_off["pitch"] = -1  # Pedal does not have pitch.
        # df_pedal_off["velocity"] = -1  # Pedal does not have velocity.
        # df_pedal_off["onset_sec"] = df_pedal_off["end_sec"]
        # df_pedal_off = df_pedal_off[["pitch", "onset_sec", "type", "type_id", "velocity"]]

        df_midi_events = pd.concat([df_note_on, df_note_off], ignore_index=True)
        df_midi_events = df_midi_events.sort_values(
            by=["onset_sec", "type_id", "pitch"], ascending=[True, True, True]
        )

        return df_midi_events

    def midi_events_to_notes(
        self, midi_event_list: List[Dict[str, Union[int, float]]]
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Args:
            midi_event_list: List of MIDI event dictionaries.
        Returns:
            midi_note_list: List of MIDI note data dictionaries.
        """
        midi_note_list = []
        df = pd.DataFrame(midi_event_list)

        # Group by pitch and calculate the duration
        grouped = df.groupby(["pitch"])
        for (pitch,), group in grouped:
            # group = group.sort_values(by=['time', "type_id", "velocity"], ascending=[True, True, True])
            for i in range(len(group)):
                event = group.iloc[i]
                if event["type"] != "NoteOn":
                    continue
                onset_sec = event["time"]
                velocity = event["velocity"]
                if velocity == 0:
                    continue
                # Find the corresponding NoteOff event
                if i + 1 < len(group):
                    dur_sec = (
                        group.iloc[i + 1]["time"] - onset_sec
                    )  # - 0.01  # Subtract a small value to avoid overlap
                else:
                    dur_sec = 0.05
                dur_sec = np.clip(
                    dur_sec, 0.02, 20.0
                )  # Clip duration to a reasonable range
                midi_note_data = {
                    "pitch": pitch,
                    "onset": onset_sec,
                    "duration": dur_sec,
                    "offset": onset_sec + dur_sec,
                    "velocity": velocity,
                    "staff": 0,  # Default staff
                }
                midi_note_list.append(midi_note_data)
        # Sort the notes by onset time
        midi_note_list.sort(key=lambda x: (x["onset"], x["pitch"]))
        return midi_note_list

    def save_midi(
        self,
        midi_note_list,
        midi_path,
        performance_downbeats=[],
        performance_timesignatures=[],
    ):
        # symusic_score = symusic.Score(midi_path, ttype=symusic.TimeUnit.second)
        midi = pretty_midi.PrettyMIDI()

        ##############################################
        # Set tempo and time signature changes

        prev_ts_str = "4/4"
        bar_dur_q = 4.0

        time_signature_list = []
        time_signature_list = [
            {"time": 0.0, "numerator": 4, "denominator": 4, "bar_dur_q": 4.0}
        ]

        for ts_data in performance_timesignatures:
            t, ts_str = ts_data["time"], ts_data["value"]
            if ts_str != prev_ts_str:
                bar_dur_q = music21.meter.TimeSignature(
                    ts_str
                ).barDuration.quarterLength
                numerator, denominator = ts_str.split("/")
                numerator = int(numerator)
                denominator = int(denominator)
                midi.time_signature_changes.append(
                    pretty_midi.containers.TimeSignature(
                        numerator, denominator, float(t)
                    )
                )
                prev_ts_str = ts_str
                if t <= 0.0:
                    time_signature_list = [
                        {
                            "time": 0.0,
                            "numerator": numerator,
                            "denominator": denominator,
                            "bar_dur_q": bar_dur_q,
                        }
                    ]
                else:
                    time_signature_list.append(
                        {
                            "time": t,
                            "numerator": numerator,
                            "denominator": denominator,
                            "bar_dur_q": bar_dur_q,
                        }
                    )

        tick_scales = []

        def get_timesignature_at_time(time):
            for ts in time_signature_list:
                if ts["time"] >= time:
                    return ts
            return time_signature_list[-1]

        total_ticks = 0
        if len(performance_downbeats) > 0:
            if performance_downbeats[0] > 0.0:
                bar_dur_q = time_signature_list[0]["bar_dur_q"]
                bar_dur_sec = performance_downbeats[0]
                ticks = int(bar_dur_q * midi.resolution)
                tick_scale = bar_dur_sec / ticks  # seconds per tick
                tick_scales.append((total_ticks, tick_scale))
                total_ticks += ticks

            for i in range(len(performance_downbeats) - 1):
                downbeat_sec = performance_downbeats[i]
                # downbeats
                bar_dur_q = get_timesignature_at_time(downbeat_sec)["bar_dur_q"]
                bar_dur_sec = performance_downbeats[i + 1] - performance_downbeats[i]
                ticks = int(bar_dur_q * midi.resolution)
                tick_scale = bar_dur_sec / ticks  # seconds per tick
                tick_scales.append((total_ticks, tick_scale))
                total_ticks += ticks

            midi._tick_scales = tick_scales
            # !!! Important!
            midi._update_tick_to_time(max(ts[0] for ts in midi._tick_scales))

        ##############################################
        # Add notes and chords to the MIDI file
        track_num = len(set(note_data["staff"] for note_data in midi_note_list))
        for i in range(track_num):
            # Create a new instrument for each staff
            midi.instruments.append(pretty_midi.Instrument(program=0, name=f"staff{i}"))

        for note_data in midi_note_list:
            pitch = note_data["pitch"]
            velocity = note_data["velocity"]
            onset_sec = note_data["onset"]
            dur_sec = note_data["duration"]
            end_sec = onset_sec + dur_sec
            staff = note_data["staff"]
            # create note object
            midi_note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=onset_sec,
                end=end_sec,
            )
            midi.instruments[staff].notes.append(midi_note)

        midi.write(midi_path)
        return True


if __name__ == "__main__":
    tokenizer = SymbolicMusicTokenizer()
    midi_path = "dataset/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
    df = tokenizer.midi_to_dataframe(midi_path)
    print(df.head())
    pass
