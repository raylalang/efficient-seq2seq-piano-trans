
import numpy as np
import pandas as pd
from glob import glob
import metrics.transcription_metrics  as transcription_metrics

subset = "validation"
subset = "test"

predict_output_dir = "runs/HPPformer_targets_notes_state_offset+pedal/240326-014525_BS5_MAESTRO_Meta_Dataset/cpt/steps_250000.ckpt_predict"

dataset_dir = 'dataset/maestro-v3.0.0/'
# Save outputs to MIDI files.
# load data
meta_csv_path = dataset_dir + '/maestro-v3.0.0.csv'
meta_df = pd.read_csv(meta_csv_path, header=0)
meta_df_subset = meta_df[meta_df["split"] == subset]

audio_ids = meta_df_subset.index.to_list() # [:3]

print(audio_ids)

ref_midi_paths = []
est_midi_paths = []
ref_midi_all = meta_df["midi_filename"]
for audio_id in audio_ids:
    output_midi_path = predict_output_dir + "/pred_audio-id=%04d.midi"%audio_id
    est_midi_paths.append(output_midi_path)

    ref_path =  dataset_dir + "/" +  ref_midi_all[audio_id]

    ref_midi_paths.append(ref_path)

metrics_path = predict_output_dir + "/pred_metrics_subset=%s.csv"%subset
transcription_metrics.cal_midi_files_metrics(est_midi_paths, ref_midi_paths, metrics_path)
df_result = pd.read_csv(metrics_path)
print(df_result.tail(5))