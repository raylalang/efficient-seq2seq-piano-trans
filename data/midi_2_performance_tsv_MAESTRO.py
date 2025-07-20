import os
import sys
work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)

import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import shutil
from data.constants import sm_tokenizer
import partitura
import symusic

dataset_dir = "dataset/maestro-v3.0.0"

def process_midi_file(midi_path):
    
    midi_path_rel = os.path.relpath(midi_path, dataset_dir)
    notes_tsv_path = os.path.join(dataset_dir, "cache", midi_path_rel.replace("/", ">").replace(".midi", "") + ".midi-notes.tsv")
    midi_path_out = notes_tsv_path.replace(".midi-notes.tsv", ".mid")
    os.makedirs(os.path.dirname(notes_tsv_path), exist_ok=True)
    shutil.copyfile(midi_path, midi_path_out)
    
    # # Pedal events
    # symusic_midi = symusic.Score(midi_path, ttype=symusic.TimeUnit.SECOND)
    # assert len(symusic_midi.tracks) == 1
    # track = symusic_midi.tracks[0]
    # pedals = track.pedals.numpy()
    # pedals["end"] = pedals["time"] + pedals["duration"]
    
    performance = partitura.load_performance(midi_path, merge_tracks=True, pedal_threshold=64)
    
    note_array = performance = performance.note_array()
    df_notes = pd.DataFrame({
        "type": "note",  # type is note
        "type_id": 1,  # 1 for note
        "onset_sec": note_array["onset_sec"],
        "duration_sec": note_array["duration_sec"],
        "offset_sec": note_array["onset_sec"] + note_array["duration_sec"],
        "pitch": note_array["pitch"],
        "velocity": note_array["velocity"],
        # "track": note_array["track"],
    })
    
    df_notes.sort_values(by=["onset_sec", "type_id", "pitch"], ascending=[True, True, True], inplace=True)
    df_notes.to_csv(notes_tsv_path, sep="\t", index=False)
    return True
    
    
if __name__ == "__main__":
    midi_paths = glob(os.path.join(dataset_dir, "**/*.midi"), recursive=True )
    results = []
    if os.environ.get("DEBUG") == "True":
        # try:
        for midi_path in tqdm(midi_paths):
            process_midi_file(midi_path)
            results.append(midi_path)
        # except Exception as e:
        #     print("Fail:", midi_path)
        #     print("Exception:", e)
    else:
        with Pool(processes=16) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_midi_file, midi_paths), total=len(midi_paths)):
                if result != False:
                    results.append(result)
    print("Total:", len(midi_paths))
    print("Success:", len(results))
    exit()
