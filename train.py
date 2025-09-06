import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torch.optim import AdamW
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data.distributed import DistributedSampler

# from torch.optim.lr_scheduler import CosineAnnealingLR


from omegaconf import OmegaConf
import hydra
from model.T5 import Transformer
from model.HPPNet import HPPNet

# from model.GPT import GPT
from model.context_poolers import BeatPooler, BarPooler


from data.dataset_Audio2Midi import Audio2Midi_Dataset
from data.collate import collate_with_context

# import data.symbolic_music_tokenizer as sm_tokenizer

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# from pytorch_lightning.trainer.states import RunningStage, TrainerFn
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.combined_loader import CombinedLoader

# import wandb
import json
import pandas as pd
from torch.utils.data import IterableDataset, Dataset, ConcatDataset

# import editdistance

# from torchaudio.transforms import MelSpectrogram

# from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from data.constants import *
import gc
import os

# from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from data.mel import MelSpectrogram

# from nnAudio.features import CQT, STFT
# import nnAudio.utils
from PIL import Image

import torchaudio
import metrics.transcription_metrics as transcription_metrics
import numpy as np

# from mir_eval.util import midi_to_hz
# from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import (
    precision_recall_f1_overlap as evaluate_notes_with_velocity,
)
import mir_eval

# import wandb
from datetime import datetime
import time as std_time
import socket

# from torch.utils.tensorboard import SummaryWriter

# from itertools import cycle
import importlib

# from itertools import chain
from tqdm import tqdm
from glob import glob

# from line_profiler import LineProfiler
# from symusic import Score, TimeUnit
from collections import defaultdict

import visualize.transcription_visualizer as transcription_visualizer
import utils.log_memory_usage as log_memory_usage
import utils.sequence_processing as sequence_processing

import shutil


class MT3Trainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config.model.model_name == "HPPNet":
            self.model = HPPNet(config=config.model)
        else:
            self.model = Transformer(config=config.model)
        self.last_time_stamp = std_time.time()
        self.validation_step_count = 0
        self.criterion_list = []
        for loss_name, [
            outputs_name,
            targets_name,
            criterion_module,
        ] in config.training.losses.items():
            criterion_class_name = criterion_module.split(".")[-1]
            criterion_class = getattr(
                importlib.import_module(criterion_module), criterion_class_name
            )
            self.criterion_list.append(
                {
                    "loss_name": loss_name,
                    "outputs_name": outputs_name,
                    "targets_name": targets_name,
                    "criterion": criterion_class(config),
                }
            )

        self.feature_extracters = {}
        # mel_extracter = MelSpectrogram(sample_rate=DEFAULT_SAMPLE_RATE, n_mels=DEFAULT_NUM_MEL_BINS, n_fft=FFT_SIZE, hop_length=DEFAULT_HOP_WIDTH,  f_min=MEL_FMIN, f_max=MEL_FMAX)
        mel_extracter = MelSpectrogram(
            config.data.num_mel_bins,
            config.data.sample_rate,
            config.data.fft_size,
            config.data.hop_length,
            mel_fmin=config.data.mel_fmin,
            mel_fmax=config.data.mel_fmax,
        )
        # cqt_extracter = CQT(sr=config.data.sample_rate, hop_length=config.data.hop_length, fmin=config.data.cqt_fmin, n_bins=config.data.num_cqt_bins, bins_per_octave=config.data.bins_per_octave)
        # log_stft_extracter = STFT(n_fft=config.data.fft_size, freq_bins=config.data.num_cqt_bins, hop_length=config.data.hop_length, freq_scale="log", fmin=config.data.mel_fmin, fmax=config.data.mel_fmax, sr=config.data.sample_rate, output_format="Magnitude")

        # self.feature_extracters["mel"] = mel_extracter.to(self.device)
        # self.feature_extracters["cqt"] = cqt_extracter.to(self.device)
        if config.data.features == "mel":
            self.features_extracter = mel_extracter
        # elif config.data.features == "cqt":
        #     self.features_extracter = cqt_extracter
        # elif config.data.features == "log-stft":
        #     self.features_extracter = log_stft_extracter
        else:
            raise ValueError(f"Unknown feature format: {config.data.features}")

        if getattr(self.config.model, "use_context", False):
            self.beat_pooler = BeatPooler(
                d_model=self.config.model.emb_dim,
                sigma_frames=getattr(self.config.model, "beat_sigma_frames", 6),
                hop_length=self.config.data.hop_length,
                sample_rate=self.config.data.sample_rate,
                add_tempo_features=getattr(
                    self.config.model, "add_tempo_features", False
                ),
            )
            self.bar_pooler = BarPooler(d_model=self.config.model.emb_dim)

        self._legacy_opt_states = None

    def forward(
        self,
        encoder_input_tokens,
        decoder_target_tokens,
        decode,
        decoder_input_tokens=None,
        decoder_positions=None,
        decoder_targets_frame_index=None,
        encoder_decoder_mask=None,
        additional_encoder_memory=None,
    ):
        return self.model.forward(
            encoder_input_tokens,
            decoder_target_tokens,
            decode=decode,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_targets_frame_index=decoder_targets_frame_index,
            encoder_decoder_mask=encoder_decoder_mask,
            additional_encoder_memory=additional_encoder_memory,
        )

    def log_time_event(self, event_name):
        if self.global_rank == 0:
            curr_time = std_time.time()
            dur = curr_time - self.last_time_stamp
            if self.global_step % 200 == 0:
                self.log("time/%s" % event_name, dur)
                print("time/%s" % event_name, "%.3f" % dur)
            self.last_time_stamp = curr_time

    def forward_step(self, batch, batch_idx, forward_type, cal_metrics=False):
        """_summary_

        Args:
            batch (_type_): _description_

        Returns:
            outputs_dict: _description_
            loss_dict:
            metrics_dict:
        """
        # concat combined dataloader manually.
        batch_list = list(batch.values())
        batch = {}
        for k in batch_list[0].keys():
            batch[k] = torch.cat([b[k] for b in batch_list], dim=0)

        outputs_dict = {}
        targets_dict = {}
        loss_dict = {}
        metrics_dict = {}

        n_steps = max(batch_idx, self.global_step)

        self.log_time_event("backword_done")
        with torch.no_grad():
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            inputs = self.features_extracter(inputs[:, :-1]).transpose(-1, -2)
            inputs = inputs.detach()
            if self.config.data.amplitude_to_db:
                # assert self.config.data.features != "mel"
                # if self.config.data.features != "mel":
                inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(inputs)
        decoder_inputs = None

        targets_dict["input_features"] = inputs
        targets = batch["decoder_targets"]
        targets_mask = batch["decoder_targets_mask"]
        targets_len = batch["decoder_targets_len"]
        decoder_target_tokens = targets.clone()

        decoder_targets_frame_index = batch["decoder_targets_frame_index"]
        encoder_decoder_mask = batch["encoder_decoder_mask"]

        # Clip seq len if it is shorter than the max.
        if False:
            _, seq_len = targets.size()
            max_seq_lens = batch["decoder_targets_len"].max()
            if max_seq_lens < seq_len:
                targets = targets[:, :max_seq_lens]
                batch["decoder_targets_mask"] = batch["decoder_targets_mask"][
                    :, :max_seq_lens
                ]

        batch_size, seq_len = targets.size()

        targets_dict["decoder_targets"] = targets
        targets_dict["decoder_targets_mask"] = targets_mask
        targets_dict["decoder_targets_len"] = targets_len

        self.log_time_event("data_load_done")
        # CNN encoder still use input shape of [B, T, F].
        # => [B*T, n_token, vocab_size], [B, T, 128]

        additional_encoder_memory = None
        if getattr(self.config.model, "use_context", False) and (
            "context_audio" in batch
        ):
            ctx = batch["context_audio"]  # [B, n_wave]
            with torch.no_grad():
                ctx_feat = self.features_extracter(ctx[:, :-1]).transpose(
                    -1, -2
                )  # [B, T_ctx, F]
                if self.config.data.amplitude_to_db:
                    ctx_feat = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(
                        ctx_feat
                    )

            # Encode context frames to the same embedding space as the main encoder
            # (dropout allowed during train; disable if you prefer deterministic memory)
            ctx_encoded = self.model.encode(
                ctx_feat, enable_dropout=self.training
            )  # [B, T_ctx, D]

            # Pool to beats/bars using beat times from the dataset
            cbt = batch["context_beat_times"]
            if isinstance(cbt, torch.Tensor):
                beats_list = [
                    cbt[b][cbt[b] >= 0] for b in range(cbt.size(0))
                ]  # keep 0, drop -1 pads
            else:
                beats_list = list(cbt)

            beat_mem, beat_mask = self.beat_pooler(
                ctx_encoded, beats_list
            )  # [B, Kb, D], [B, Kb]
            bar_mem, bar_mask = self.bar_pooler(
                beat_mem,
                beat_mask,
                beats_per_bar=int(getattr(self.config.model, "beats_per_bar", 4)),
            )

            mode = getattr(
                self.config.model, "context_mode", "beat_bar"
            )  # beat|bar|beat_bar
            mems = []
            additional_encoder_memory = None
            if mode in ("beat", "beat_bar") or mode in ("bar", "beat_bar"):
                add = {}
                if mode in ("beat", "beat_bar"):
                    add["beat"] = beat_mem  # [B,Kb,D]
                    add["beat_mask"] = beat_mask  # [B,Kb] 1=valid, 0=pad
                if mode in ("bar", "beat_bar"):
                    add["bar"] = bar_mem  # [B,Kbar,D]
                    add["bar_mask"] = bar_mask  # [B,Kbar]
                additional_encoder_memory = add

        if (
            additional_encoder_memory is not None
            and self.global_rank == 0
            and n_steps < 5
        ):
            if isinstance(additional_encoder_memory, dict):
                bl = additional_encoder_memory.get("beat")
                br = additional_encoder_memory.get("bar")
                print(
                    "context memory lens:",
                    f"beat={0 if bl is None else bl.size(1)},",
                    f"bar={0 if br is None else br.size(1)}",
                )

        outputs_dict = self.forward(
            encoder_input_tokens=inputs,
            decoder_target_tokens=decoder_target_tokens,
            decode=False,
            decoder_input_tokens=decoder_inputs,
            decoder_targets_frame_index=decoder_targets_frame_index,
            encoder_decoder_mask=encoder_decoder_mask,
            additional_encoder_memory=additional_encoder_memory,
        )

        self.log_time_event("forward_done")

        # cal losses
        if "loss" in outputs_dict:
            # If model has already calculated loss.
            loss_dict["loss"] = outputs_dict["loss"]
        else:
            total_loss = 0
            for dic in self.criterion_list:
                outputs = outputs_dict[dic["outputs_name"]]
                _targets = targets_dict[dic["targets_name"]]
                _targets_mask = targets_dict[dic["targets_name"] + "_mask"]
                loss = dic["criterion"](outputs, _targets, _targets_mask)
                if not torch.isfinite(loss):
                    print("Error! Loss is inf!!!", loss)
                    loss = loss * 0
                if type(loss) is dict:
                    for k, v in loss.items():
                        loss_dict[dic["loss_name"] + "_" + k] = v
                        total_loss += v
                else:
                    loss_dict[dic["loss_name"]] = loss
                    total_loss += loss
            loss_dict["loss"] = total_loss
        self.log_time_event("loss_cal_done")

        decoder_outputs = outputs_dict["decoder_outputs"]
        decoder_outputs_probs = torch.softmax(decoder_outputs, dim=-1)[
            ..., : self.config.model.vocab_size
        ]
        decoder_targets_token = targets_dict["decoder_targets"]
        decoder_targets_mask = targets_dict["decoder_targets_mask"]

        # cal metrics
        if cal_metrics:
            decoder_outputs_token = torch.argmax(decoder_outputs_probs, dim=-1)
            decoder_outputs_token[decoder_targets_mask == 0] = (
                TOKEN_PAD  # ignore masked tokens.
            )
            outputs_dict["decoder_outputs_token"] = decoder_outputs_token
            self.cal_decoder_token_metrics(
                decoder_outputs_token,
                decoder_targets_token,
                batch,
                metrics_dict,
                metric_prefix="",
            )

        # Visualize decoder outputs.
        steps_list = self.config.training.save_pianoroll_every_n_steps
        if self.global_rank == 0 and (
            n_steps in steps_list or n_steps % steps_list[-1] == 0
        ):
            img_path = os.path.join(
                self.logger.save_dir, "img"
            ) + "/decoder_outputs_%06d_stack_%s.png" % (n_steps, forward_type)
            decoder_targets_onehot = Functional.one_hot(
                decoder_targets_token, num_classes=self.config.model.vocab_size
            )
            transcription_visualizer.save_frame_pianoroll(
                img_path, decoder_outputs_probs, decoder_targets_onehot
            )

        # Save attention weights to local dir.
        steps_list = self.config.training.save_pianoroll_every_n_steps
        if self.global_rank == 0 and (
            n_steps < 1 or (n_steps in steps_list or n_steps % steps_list[-1] == 0)
        ):
            if "cross_attention_weights" in outputs_dict:
                self.visualize_attention_weights(
                    outputs_dict["cross_attention_weights"],
                    n_steps,
                    forward_type,
                    batch,
                )
            # layer_index = 0
            for k, v in outputs_dict.items():
                if k.startswith("decoder_layer_cross_attention_weights"):
                    layer_index = int(k.split("_")[-1])
                    self.visualize_attention_weights(
                        v, n_steps, forward_type, batch, layer_index=layer_index
                    )
                    # layer_index += 1

        # Save input features to local dir for preview.
        if self.global_rank == 0 and n_steps < 5:
            if "input_features" in targets_dict:
                B, T, F = targets_dict["input_features"].size()
                flatten_input_features = (
                    targets_dict["input_features"].reshape([B * T, F]).cpu().numpy().T
                )
                img_path = os.path.join(
                    self.logger.save_dir, "img"
                ) + "/input_features_%s_%03d.png" % (forward_type, n_steps)
                wav_path = os.path.join(
                    self.logger.save_dir, "img"
                ) + "/input_features_%s_%03d.mp3" % (forward_type, n_steps)
                plt.imsave(img_path, flatten_input_features)
                torchaudio.save(
                    wav_path,
                    batch["inputs"].cpu().detach().flatten()[None, :],
                    sample_rate=self.config.data.sample_rate,
                    format="mp3",
                )

        return outputs_dict, targets_dict, loss_dict, metrics_dict

    def visualize_attention_weights(
        self, attention_weights, n_steps, forward_type, batch, layer_index=-1
    ):
        """save attention weights img to local dir."""
        cross_attention_weights = attention_weights.detach().cpu()
        # normalize attention weights to range [0, 1]
        batch_size, n_heads, output_seq_len, input_seq_len = (
            cross_attention_weights.size()
        )
        for i in range(batch_size):
            if i > 4:
                break
            decoder_targets_len = (
                batch["decoder_targets_len"][i].item() // SEQUENCE_POOLING_SIZE
            )
            img_path = os.path.join(
                self.logger.save_dir, "img"
            ) + "/decoder_cross_attention_%s_%05d_idx=%02d_layer=%d.png" % (
                forward_type,
                n_steps,
                i,
                layer_index,
            )
            # Save attention weights to local dir.
            weight_i = cross_attention_weights[i, :, :decoder_targets_len, :]
            min_i = weight_i.min()
            max_i = weight_i.max()
            if max_i > 0:
                weight_i = (weight_i - min_i) / (max_i - min_i)
            else:
                weight_i = weight_i - min_i
            if min_i < 0 or max_i > 1:
                print("Error! Attention weights out of range [0, 1]!", min_i, max_i)
            transcription_visualizer.save_frame_pianoroll(
                img_path, weight_i, weight_i * 0
            )

    @rank_zero_only
    def cal_decoder_token_metrics(
        self,
        decoder_outputs_token,
        decoder_targets_token,
        batch,
        metrics_dict,
        metric_prefix="",
    ):
        """Visualize decoder outputs.

        Args:
            decoder_outputs_token (_type_): [batch, seq_len]
            decoder_targets_token (_type_): [batch, seq_len]
            batch (_type_): _description_
            metric_prefix (_type_): _description_
        """

        self.log_time_event("metrics_cal_begin")
        metrict_ret = transcription_metrics.cal_decoder_token_metrics(
            decoder_outputs_token, decoder_targets_token, metric_prefix=metric_prefix
        )

        # cal metrics for different datasets.
        if len(self.config.data.dataset_dirs) > 1:
            dataset_indexs = set(batch["dataset_index"].cpu().numpy())
            for dataset_index in dataset_indexs:
                indexs = batch["dataset_index"] == dataset_index
                sel_target_tokens = decoder_targets_token[indexs]
                sel_output_tokens = decoder_outputs_token[indexs]
                sel_metrict_ret = transcription_metrics.cal_decoder_token_metrics(
                    sel_output_tokens, sel_target_tokens, metric_prefix=metric_prefix
                )
                for k, v in sel_metrict_ret.items():
                    metrict_ret["%s_%d" % (k, dataset_index)] = v

        metrics_dict.update(metrict_ret)

    def training_step(
        self, batch, batch_idx, dataloader_idx=None
    ):  # This batch_indx will be reseted to 0 in a new epoch.
        cal_metrics = False
        if (
            self.global_rank == 0
            and self.global_step % self.config.training.cal_metrics_every_n_steps == 0
        ):
            cal_metrics = True
        outputs_dict, targets_dict, loss_dict, metrics_dict = self.forward_step(
            batch, batch_idx, "train", cal_metrics
        )

        if self.global_rank == 0:
            # save checkpoint.
            if self.global_step > 0:
                if (
                    self.global_step in [1000, 2000, 3000, 4000, 5000, 8000, 10000]
                    or self.global_step % self.config.training.checking_steps == 0
                ):
                    cpt_dir = os.path.join(self.logger.save_dir, "cpt")
                    os.makedirs(cpt_dir, exist_ok=True)
                    checkpoint_path = os.path.join(
                        cpt_dir, f"steps_{self.global_step}.ckpt"
                    )

                    # FULL checkpoint: model + optimizer + epoch/step
                    state = {
                        "state_dict": self.model.state_dict(),
                        "global_step": int(self.global_step),
                        "epoch": int(self.current_epoch),
                    }
                    try:
                        if self.trainer.optimizers:
                            state["optimizer_states"] = [
                                opt.state_dict() for opt in self.trainer.optimizers
                            ]
                    except Exception as _e:
                        pass  # model-only if optimizer not available here

                    # torch.save(state, checkpoint_path)
                    self.trainer.save_checkpoint(checkpoint_path)
                    if hasattr(self, "previous_cpt_path"):
                        if self.prev_checkpoint_step in [10000, 50000, 100000, 200000]:
                            pass
                        else:
                            if os.path.exists(self.previous_cpt_path):
                                try:
                                    os.remove(self.previous_cpt_path)
                                except Exception as e:
                                    print("Error removing previous checkpoint:", e)
                    self.previous_cpt_path = checkpoint_path
                    self.prev_checkpoint_step = self.global_step

            # log loss and metrics.
            log_dict = metrics_dict.copy()
            log_dict.update(loss_dict)
            if self.global_step % 10 == 0:
                for k, v in log_dict.items():
                    if isinstance(v, (int, float, bool, torch.Tensor)):
                        self.log("train/%s" % k, v, sync_dist=False)
                # log memory usage
                if self.global_step % 200 == 0:
                    log_memory_usage.log_gpu_memory_usage(self)

        self.log_time_event("forward_step_done")

        return loss_dict["loss"]

    def on_train_epoch_start(self):
        # ensure new RNG split each epoch for DistributedSampler(s)
        dls = getattr(getattr(self.trainer, "train_dataloader", None), "loaders", None)
        if isinstance(dls, dict):
            for dl in dls.values():
                s = getattr(dl, "sampler", None)
                if hasattr(s, "set_epoch"):
                    s.set_epoch(self.current_epoch)
        self.features_extracter = self.features_extracter.to(self.device)

    @rank_zero_only
    def on_train_epoch_end(
        self,
    ):
        # Save checkpoint.
        cpt_dir = os.path.join(os.path.join(self.logger.save_dir, "cpt"))
        os.system("mkdir -p %s" % cpt_dir)
        # torch.save(
        #     {
        #         "state_dict": self.model.state_dict(),
        #         "global_step": int(self.global_step),
        #         "epoch": int(self.current_epoch),
        #         "optimizer_states": (
        #             [opt.state_dict() for opt in self.trainer.optimizers]
        #             if self.trainer.optimizers
        #             else []
        #         ),
        #     },
        #     os.path.join(cpt_dir, "latest.ckpt"),
        # )
        self.trainer.save_checkpoint(os.path.join(cpt_dir, "latest.ckpt"))

        txt_path = cpt_dir + "/latest_epoch.txt"
        with open(txt_path, "w") as f:
            f.write(
                "latest epoch=%d , global step=%d\n"
                % (self.current_epoch, self.global_step)
            )

    def on_fit_start(self):
        self.features_extracter = self.features_extracter.to(self.device)
        # --- legacy optimizer state restore (safe) ---
        if self._legacy_opt_states:
            try:
                for opt, st in zip(self.trainer.optimizers, self._legacy_opt_states):
                    opt.load_state_dict(st)
            except Exception as e:
                print(f"[warn] failed to load legacy optimizer states: {e}")
            self._legacy_opt_states = None

    @torch.no_grad()
    def on_validation_start(self):
        self.features_extracter = self.features_extracter.to(self.device)

        # Manually call test_steps (online testing).
        if (
            self.global_step > 1 and self.config.training.online_testing
        ):  # skip the validation step in the init epoch.
            self.test_outputs_dict = defaultdict(list)
            test_dataloader = self.test_dataloader()
            for batch_idx, batch in tqdm(
                enumerate(test_dataloader),
                desc="Testing ...",
                total=len(test_dataloader),
                disable=(self.global_rank != 0),
            ):
                new_batch = {}
                for k, v in batch.items():
                    new_batch[k] = v.to(self.device)  # GPU memory keak problem here!
                self.test_step(new_batch, batch_idx)
            self.on_test_epoch_end()

        return super().on_validation_start()

    @torch.no_grad()
    def on_test_start(self):
        self.features_extracter = self.features_extracter.to(self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        outputs_dict, targets_dict, loss_dict, metrics_dict = self.forward_step(
            batch, batch_idx, "validation", cal_metrics=True
        )
        log_dict = {}
        log_dict.update(loss_dict)
        log_dict.update(metrics_dict)
        for k, v in log_dict.items():
            if isinstance(v, (int, float, bool, torch.Tensor)):
                self.log("validation/%s" % k, v, on_epoch=True, sync_dist=False)  #

        return loss_dict["loss"]

    # @torch.no_grad()
    # def test_step_(self, batch, batch_idx):
    #     return self.test_step(batch, batch_idx)
    #     lp = LineProfiler()
    #     lp_wrapper = lp(self.test_step_parallel)
    #     lp_wrapper(batch, batch_idx)
    #     lp.print_stats()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        metrics_dict = {}
        # [B, n_wave_samples]
        input_waves = batch["inputs"]
        with torch.no_grad():
            # => [B, T, F]
            encoder_inputs = self.features_extracter(input_waves[:, :-1]).transpose(
                -1, -2
            )
            encoder_inputs = encoder_inputs.detach()
            if self.config.data.amplitude_to_db:
                # assert self.config.data.features != "mel"
                if self.config.data.features != "mel":
                    encoder_inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(
                        encoder_inputs
                    )

        # [B, n_tokens]
        target_tokens = batch["decoder_targets"]
        n_tokens = target_tokens.size()[1]

        # inference speed
        curr_time_sec = std_time.time()

        additional_encoder_memory = None
        if getattr(self.config.model, "use_context", False) and (
            "context_audio" in batch
        ):
            ctx = batch["context_audio"]  # [B, n_wave]
            with torch.no_grad():
                ctx_feat = self.features_extracter(ctx[:, :-1]).transpose(
                    -1, -2
                )  # [B,Tc,F]
                if self.config.data.amplitude_to_db:
                    ctx_feat = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(
                        ctx_feat
                    )

            # Encode and pool to beats/bars
            ctx_encoded = self.model.encode(
                ctx_feat, enable_dropout=False
            )  # [B, Tc, D]
            cbt = batch["context_beat_times"]
            if isinstance(cbt, torch.Tensor):
                beats_list = [cbt[b][cbt[b] >= 0] for b in range(cbt.size(0))]
            else:
                beats_list = list(cbt)

            beat_mem, beat_mask = self.beat_pooler(
                ctx_encoded, beats_list
            )  # [B,Kb,D], [B,Kb]
            bar_mem, bar_mask = self.bar_pooler(
                beat_mem,
                beat_mask,
                beats_per_bar=int(getattr(self.config.model, "beats_per_bar", 4)),
            )

            mode = getattr(self.config.model, "context_mode", "beat_bar")
            additional_encoder_memory = None
            if mode in ("beat", "beat_bar") or mode in ("bar", "beat_bar"):
                add = {}
                if mode in ("beat", "beat_bar"):
                    add["beat"] = beat_mem  # [B,Kb,D]
                    add["beat_mask"] = beat_mask  # [B,Kb] 1=valid, 0=pad
                if mode in ("bar", "beat_bar"):
                    add["bar"] = bar_mem  # [B,Kbar,D]
                    add["bar_mask"] = bar_mask  # [B,Kbar]
                additional_encoder_memory = add

        if (
            additional_encoder_memory is not None
            and self.global_rank == 0
            and batch_idx < 2
        ):
            if isinstance(additional_encoder_memory, dict):
                bl = additional_encoder_memory.get("beat")
                br = additional_encoder_memory.get("bar")
                print(
                    "context memory lens:",
                    f"beat={0 if bl is None else bl.size(1)},",
                    f"bar={0 if br is None else br.size(1)}",
                )

        decoder_output_tokens = self.model.generate(
            encoder_inputs,
            target_seq_length=n_tokens,
            # break_on_eos=True,
            break_on_eos=False,
            global_rank=self.global_rank,
            additional_encoder_memory=additional_encoder_memory,
        )

        dur_time_sec = std_time.time() - curr_time_sec
        generate_token_len = max(1, decoder_output_tokens.size(1))
        tokens_per_sec = generate_token_len / dur_time_sec

        # # get_output_seq_lens
        # output_eos_tokens_flag = decoder_output_tokens
        # output_eos_tokens_flag = (
        #     output_eos_tokens_flag == sm_tokenizer.EOS
        # ).int() * sm_tokenizer.EOS
        # output_seq_lens = sequence_processing.get_sequence_lengths(
        #     output_eos_tokens_flag, sm_tokenizer.EOS
        # )
        # Trim to FIRST EOS per-sample for scoring (fallback = full length if no EOS)
        out = decoder_output_tokens  # [B, T]
        eos_id = sm_tokenizer.EOS
        eos_mask = out.eq(eos_id)  # [B, T]
        has_eos = eos_mask.any(dim=1)  # [B]
        first_eos = eos_mask.float().argmax(dim=1).to(torch.long)  # [B]
        full_len = torch.full_like(first_eos, out.size(1))  # [B]
        output_seq_lens = torch.where(has_eos, first_eos, full_len)

        if self.global_rank == 0 and batch_idx < 3:
            T_enc = int(encoder_inputs.size(1))
            T_tgt = int(n_tokens)
            T_out_trimmed = int(output_seq_lens.float().mean().round().item())
            print(
                f"[gen] enc_T={T_enc} tgt={T_tgt} out={T_out_trimmed} ratio={T_out_trimmed/max(1,T_tgt):.3f}"
            )

        target_sequence_lens = batch["decoder_targets_len"]
        assert len(output_seq_lens) == len(
            target_sequence_lens
        ), f"Output sequence length {len(output_seq_lens)} does not match target sequence length {len(target_sequence_lens)}."

        def pad_to_max_size(tensor, max_size, value=-1):
            pad_size = max_size - tensor.size(0)
            if pad_size > 0:
                padding = (0, 0) * (tensor.dim() - 1) + (
                    0,
                    pad_size,
                )  # Only pad batch dimension
                tensor = torch.nn.functional.pad(tensor, padding, value=value)
            return tensor

        # Pad the tensors of different ranks to the maximum size
        batch_size = batch["audio_ids"].size(0)
        batch_size_t = torch.tensor(
            [batch_size], device=encoder_inputs.device, dtype=torch.int64
        )
        max_batch_size = int(self.all_gather(batch_size_t).max())

        gather_data = self.all_gather(
            {
                "target_tokens": pad_to_max_size(target_tokens, max_batch_size),
                "output_tokens": pad_to_max_size(decoder_output_tokens, max_batch_size),
                "target_tokens_lens": pad_to_max_size(
                    target_sequence_lens, max_batch_size
                ),
                "output_tokens_lens": pad_to_max_size(output_seq_lens, max_batch_size),
                "audio_ids": pad_to_max_size(batch["audio_ids"], max_batch_size),
                "audio_name": pad_to_max_size(batch["audio_name"], max_batch_size),
                "midi_path": pad_to_max_size(batch["midi_path"], max_batch_size),
                "frame_offsets": pad_to_max_size(
                    batch["frame_offsets"], max_batch_size
                ),
            }
        )

        # normalize shapes for single-GPU all_gather
        out_tokens = gather_data["output_tokens"]

        # In single-GPU, Lightning returns tensors without a world dim (e.g., [B, T]).
        # Add a fake rank dim so everything becomes [1, B, ...] and the code below
        # can treat single-GPU and DDP the same way.
        if out_tokens.ndim == 2:  # [B, T] -> [1, B, T]
            for k, v in list(gather_data.items()):
                if isinstance(v, torch.Tensor) and v.size(0) == max_batch_size:
                    gather_data[k] = v.unsqueeze(0)
            out_tokens = gather_data["output_tokens"]

        world_size, B, T = out_tokens.size()

        # Aggregate only on rank zero (Lightning-idiomatic)
        if self.global_rank == 0:
            for w in range(world_size):
                for b in range(B):
                    audio_id = int(gather_data["audio_ids"][w][b])
                    if audio_id < 0:
                        continue
                    tgt_len = int(gather_data["target_tokens_lens"][w][b])
                    out_len = int(gather_data["output_tokens_lens"][w][b])

                    audio_name = "".join(
                        [chr(x) for x in gather_data["audio_name"][w][b].cpu().numpy()]
                    ).strip()
                    midi_path = "".join(
                        [chr(x) for x in gather_data["midi_path"][w][b].cpu().numpy()]
                    ).strip()

                    target_tokens_i = (
                        gather_data["target_tokens"][w][b, :tgt_len].cpu().numpy()
                    )
                    output_tokens_i = (
                        gather_data["output_tokens"][w][b, :out_len].cpu().numpy()
                    )

                    self.test_outputs_dict[audio_id].append(
                        {
                            "target_tokens": target_tokens_i.tolist(),
                            "output_tokens": output_tokens_i.tolist(),
                            "audio_name": audio_name,
                            "midi_path": midi_path,
                            "frame_offsets": int(gather_data["frame_offsets"][w][b]),
                            "audio_ids": audio_id,
                            "target_tokens_lens": tgt_len,
                            "output_tokens_lens": out_len,
                            "batch_size": int(max_batch_size),
                        }
                    )

        return metrics_dict

    @rank_zero_only
    def on_test_epoch_start(
        self,
    ):
        # self.test_outputs_dict = defaultdict(lambda: defaultdict(list))
        self.test_outputs_dict = defaultdict(list)

    @rank_zero_only
    def on_test_epoch_end(
        self,
    ):  #  outputs

        print("on_test_epoch_end, test_outputs_dict size:", len(self.test_outputs_dict))

        metric_dict = {
            "note_precision": [],
            "note_recall": [],
            "note_f1": [],
            "target_length": [],
            "midi_path": [],
        }

        metric_dict = defaultdict(list)

        # Save test tokens to json
        if self.config.training.mode == "test":
            test_output_dir = self.test_output_dir
        else:
            test_output_dir = self.logger.save_dir + "/test/step=%06d" % (
                self.global_step
            )
        os.makedirs(test_output_dir, exist_ok=True)

        for audio_id, data_list in tqdm(
            self.test_outputs_dict.items(),
            desc="Saving test results",
            disable=self.global_rank != 0,
        ):
            data_list = sorted(data_list, key=lambda x: x["frame_offsets"])
            audio_name = data_list[0]["audio_name"]
            target_midi_path = data_list[0]["midi_path"]
            # audio_name = audio_name.replace(" ", "_")

            # copy midi file to test output dir.
            shutil.copy(
                target_midi_path,
                test_output_dir + "/" + os.path.basename(target_midi_path),
            )

            metric_dict["batch_size"].append(data_list[0]["batch_size"])

            concat_output_tokens = sum([row["output_tokens"] for row in data_list], [])
            concat_target_tokens = sum([row["target_tokens"] for row in data_list], [])

            # Calculate Note Level Metrics
            output_event_data_list = []
            target_event_data_list = []
            sec_per_frame = DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
            for row in data_list:
                offsets_sec = row["frame_offsets"] * sec_per_frame
                output_event_data_list += sm_tokenizer.detokenize(
                    row["output_tokens"], offsets_sec
                )
                target_event_data_list += sm_tokenizer.detokenize(
                    row["target_tokens"], offsets_sec
                )
            output_note_data_list = sm_tokenizer.midi_events_to_notes(
                output_event_data_list
            )
            target_note_data_list = sm_tokenizer.midi_events_to_notes(
                target_event_data_list
            )

            # Onset only Metrics
            output_interval = np.array(
                [(note["onset"], note["offset"]) for note in output_note_data_list]
            )
            target_interval = np.array(
                [(note["onset"], note["offset"]) for note in target_note_data_list]
            )
            output_pitches = np.array([note["pitch"] for note in output_note_data_list])
            target_pitches = np.array([note["pitch"] for note in target_note_data_list])
            output_velocities = np.array(
                [note["velocity"] for note in output_note_data_list]
            )
            target_velocities = np.array(
                [note["velocity"] for note in target_note_data_list]
            )
            output_pitches = mir_eval.util.midi_to_hz(output_pitches)
            target_pitches = mir_eval.util.midi_to_hz(target_pitches)

            precision, recall, f1, overlap = evaluate_notes(
                target_interval,
                target_pitches,
                output_interval,
                output_pitches,
                offset_ratio=None,
            )
            metric_dict["note_precision"].append(precision)
            metric_dict["note_recall"].append(recall)
            metric_dict["note_f1"].append(f1)

            precision, recall, f1, overlap = evaluate_notes(
                target_interval, target_pitches, output_interval, output_pitches
            )
            metric_dict["note+offset_precision"].append(precision)
            metric_dict["note+offset_recall"].append(recall)
            metric_dict["note+offset_f1"].append(f1)

            precision, recall, f1, overlap = evaluate_notes_with_velocity(
                target_interval,
                target_pitches,
                target_velocities,
                output_interval,
                output_pitches,
                output_velocities,
                velocity_tolerance=0.1,
            )
            metric_dict["note+offset+velocity_precision"].append(precision)
            metric_dict["note+offset+velocity_recall"].append(recall)
            metric_dict["note+offset+velocity_f1"].append(f1)

            target_len = max(len(concat_target_tokens), 1)  # Avoid division by zero.

            metric_dict["target_length"].append(target_len)
            target_midi_basename = os.path.basename(target_midi_path)

            metric_dict["output_length"].append(len(concat_output_tokens))
            metric_dict["out_to_tgt_ratio"].append(
                len(concat_output_tokens) / max(1, len(concat_target_tokens))
            )

            metric_dict["midi_path"].append(target_midi_basename)

            json_path = (
                test_output_dir
                + "/"
                + os.path.splitext(target_midi_basename)[0]
                + ".output.json"
            )
            with open(json_path, "w") as f:
                data_dict = {
                    "audio_id": audio_id,
                    "audio_name": audio_name,
                    "midi_path": target_midi_basename,
                    "data_list": data_list,
                }
                json.dump(data_dict, f, ensure_ascii=False, indent=2)

            try:
                output_midi_path = (
                    test_output_dir
                    + "/"
                    + os.path.splitext(target_midi_basename)[0]
                    + ".output.mid"
                )
                sm_tokenizer.save_midi(output_note_data_list, output_midi_path)
            except Exception as e:
                print("Error saving midi file:", e)

        df = pd.DataFrame(metric_dict)
        # Calculate mean and insert into the first row
        avg_value = df.select_dtypes(include="number").mean()
        df.loc["Average"] = avg_value
        df = pd.concat([df.loc[["Average"]], df.drop("Average")])
        df.to_csv(test_output_dir + "/test_metrics_summary.csv", index=False)

        # Log Metrics.
        if self.config.training.mode == "train":
            for k, v in avg_value.items():
                self.log("test/%s" % k, v)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), self.config.training.learning_rate)

        # scheduler = CosineAnnealingLR(optimizer, T_max=10)
        # scheduler.get_lr()
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        return optimizer

    def train_dataloader(self):
        # train_data = MIDIDataset(type='train')
        if self.config.data.dataset_name == "Audio2Midi_Dataset":
            datasets = []
            for dataset_index, dataset_dir in enumerate(self.config.data.dataset_dirs):
                if not os.path.exists(dataset_dir):
                    raise ValueError("Dataset dir %s not exists!" % dataset_dir)
                datasets.append(
                    Audio2Midi_Dataset(
                        self.config,
                        dataset_dir,
                        dataset_index=dataset_index,
                        subset="train",
                        random_clip=True,
                    )
                )
        else:
            raise "Unknown dataset:" + self.config.data.dataset_name

        # Set dataloader for multiple datasets
        n_datasets = len(self.config.data.dataset_dirs)
        assert (
            self.config.training.batch % n_datasets == 0
        ), "Batch size should be divisible by the number of datasets."
        world_size = getattr(self.trainer, "world_size", 1)
        rank = getattr(self, "global_rank", 0)
        iterables = {}
        for i, dataset in enumerate(datasets):
            sampler = (
                DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                if world_size > 1
                else None
            )
            shuffle = False if sampler is not None else True
            iterables[str(i)] = DataLoader(
                dataset,
                batch_size=self.config.training.batch // n_datasets,
                num_workers=self.config.training.num_workers // n_datasets,
                sampler=sampler,
                shuffle=shuffle,
                collate_fn=collate_with_context,
                persistent_workers=(self.config.training.num_workers > 0),
                pin_memory=True,
                prefetch_factor=4,
            )
        combined_loader = CombinedLoader(iterables, mode="max_size_cycle")
        return combined_loader

    def val_dataloader(self):
        if self.config.data.dataset_name == "Audio2Midi_Dataset":
            datasets = []
            for dataset_index, dataset_dir in enumerate(self.config.data.dataset_dirs):
                if not os.path.exists(dataset_dir):
                    raise ValueError("Dataset dir %s not exists!" % dataset_dir)
                datasets.append(
                    Audio2Midi_Dataset(
                        self.config,
                        dataset_dir,
                        dataset_index=dataset_index,
                        subset="validation",
                        random_clip=False,
                    )
                )
            validation_data = ConcatDataset(datasets)
            # validation_data = Audio2Midi_Dataset(self.config, subset="validation", random_clip=False)
        else:
            raise "Unknown dataset:" + self.config.data.dataset_name

        # Set dataloader for multiple datasets
        n_datasets = len(self.config.data.dataset_dirs)
        assert (
            self.config.training.batch % n_datasets == 0
        ), "Batch size should be divisible by the number of datasets."
        world_size = getattr(self.trainer, "world_size", 1)
        rank = getattr(self, "global_rank", 0)
        iterables = {}
        for i, dataset in enumerate(datasets):
            sampler = (
                DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                if world_size > 1
                else None
            )
            iterables[str(i)] = DataLoader(
                dataset,
                batch_size=self.config.training.batch // n_datasets,
                num_workers=self.config.training.num_workers // n_datasets,
                sampler=sampler,
                shuffle=False,
                collate_fn=collate_with_context,
                # persistent_workers=(self.config.training.num_workers > 0),
                persistent_workers=False,
                pin_memory=False,
                prefetch_factor=2,
            )
        combined_loader = CombinedLoader(iterables, mode="max_size_cycle")
        return combined_loader

    def test_dataloader(self):
        if self.config.data.dataset_name == "Audio2Midi_Dataset":
            datasets = []
            for dataset_index, dataset_dir in enumerate(self.config.data.dataset_dirs):
                if not os.path.exists(dataset_dir):
                    raise ValueError("Dataset dir %s not exists!" % dataset_dir)
                datasets.append(
                    Audio2Midi_Dataset(
                        self.config,
                        dataset_dir,
                        dataset_index=dataset_index,
                        subset="test",
                        random_clip=False,
                    )
                )
            test_data = ConcatDataset(datasets)
            # test_data = Audio2Midi_Dataset(self.config, subset="test", random_clip=False)
        else:
            raise "Unknown dataset:" + self.config.data.dataset_name

        world_size = getattr(self.trainer, "world_size", 1)
        rank = getattr(self, "global_rank", 0)
        sampler = (
            DistributedSampler(
                test_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        testloader = DataLoader(
            test_data,
            batch_size=self.config.training.batch_test,
            num_workers=self.config.training.num_workers,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_with_context,
            # persistent_workers=(self.config.training.num_workers > 0),
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=2,
        )
        return testloader


@rank_zero_only
def Init_rank_zero_only(log_dir):
    os.environ["MT3-LOGGER-PATH"] = log_dir


def _is_lightning_ckpt(path: str) -> bool:
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(obj, dict) and (
        "pytorch-lightning_version" in obj or "loops" in obj
    )


@hydra.main(config_path="config", config_name="main_config", version_base=None)
def my_main(config: OmegaConf):
    torch.cuda.empty_cache()
    gc.collect()

    # Get log dir
    if config.training.log_dir is None:
        model_name = config.model.model_name
        time_str = datetime.now().strftime("%y%m%d-%H%M%S")
        # loss_name = config.training.loss
        batch_size = "BS" + str(config.training.batch)
        token_types = str(config.data.token_types)
        log_workdir = "runs"
        if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
            log_workdir = "runs-debug"
        experiment_name = "_".join([model_name])
        assert (
            config.training.notes is not None and len(config.training.notes) > 0
        ), "Please set config.training.notes to a non-empty string."
        run_name = "_".join(
            [time_str, config.training.notes]
        )  # batch_size, config.data.dataset_name
        log_dir = os.path.join(log_workdir, experiment_name, run_name)
        Init_rank_zero_only(log_dir)
        log_dir = os.environ["MT3-LOGGER-PATH"]
        config.training.log_dir = log_dir
    else:
        log_dir = config.training.log_dir
    ####################################################
    # Create model.
    model = MT3Trainer(config)
    print(model)

    # Load checkpoint.
    # if config.model.checkpoint_path is None:
    #     assert (
    #         config.model.froze_encoder == False
    #     ), "If you want to train from scratch, please set config.model.froze_encoder to False and config.model.checkpoint_path to None."
    # else:
    #     ckpt = torch.load(config.model.checkpoint_path, map_location="cpu")
    #     state_dict = ckpt.get("state_dict", ckpt)  # supports both formats
    #     if config.model.checkpoint_ignore_layers is not None:
    #         for key in config.model.checkpoint_ignore_layers:
    #             if key in state_dict:
    #                 del state_dict[key]
    #     model.model.load_state_dict(state_dict, strict=config.model.strict_checkpoint)

    resume_ckpt = getattr(config.training, "resume_from_ckpt", None)
    init_ckpt = getattr(config.model, "checkpoint_path", None)

    ckpt_path_for_trainer = None
    if resume_ckpt:  # exact resume from a Lightning checkpoint (recommended & exact)
        if _is_lightning_ckpt(str(resume_ckpt)):
            ckpt_path_for_trainer = str(resume_ckpt)
        else:
            raise ValueError(
                f"training.resume_from_ckpt={resume_ckpt} is not a Lightning checkpoint; "
                "exact resume requires a Lightning ckpt produced by save_checkpoint."
            )
    elif init_ckpt:  # legacy torch.save load (weights + optimizer states only, safe)
        ckpt_obj = torch.load(init_ckpt, map_location="cpu")
        state_dict = ckpt_obj.get("state_dict", ckpt_obj)
        # allow Lightning-style prefixes
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        ignore = set(getattr(config.model, "checkpoint_ignore_layers", []) or [])
        for k in list(state_dict.keys()):
            if k in ignore:
                del state_dict[k]
        strict = bool(getattr(config.model, "strict_checkpoint", True))
        model.model.load_state_dict(state_dict, strict=strict)

        # try to restore optimizer states safely later during on_fit_start
        legacy_opts = ckpt_obj.get("optimizer_states", None)
        if legacy_opts:
            model._legacy_opt_states = legacy_opts  # Lightning will have optimizers initialized by on_fit_start

        # NOTE: we DO NOT force set epoch/global_step here: doing that without PL’s loop
        # state is brittle. Exact resume is guaranteed only for Lightning ckpts.

    # model.model = torch.compile(model.model)

    # Create logger.
    # if "DEBUG" in os.environ and os.environ["DEBUG"] == "True":
    #     wandb_logger = WandbLogger(
    #         name=experiment_name + config.training.notes,
    #         project="mt3-score-pytorch-debug",
    #         offline=config.training.debug_log_offline,
    #         save_dir=log_dir,
    #     )  # , rank_zero_only=True
    # else:
    #     wandb_logger = WandbLogger(
    #         name=experiment_name + config.training.notes,
    #         project="AMT-audio-to-midi",
    #         offline=False,
    #         save_dir=log_dir,
    #         notes=config.training.notes,
    #     )  # , rank_zero_only=True
    tensorboard_logger = TensorBoardLogger(save_dir=log_dir)

    # Save informations to log dir.
    if model.global_rank == 0:
        # save model architecture to txt
        os.makedirs(log_dir, exist_ok=True)
        with open(log_dir + "/model_architecture.txt", "w") as f:
            f.write(str(config) + "\n" * 5)
            f.write("\n".join([socket.gethostname(), " "]))
            f.write(str(model))

        # image dir.
        img_dir = os.path.join(log_dir, "img")
        os.system('mkdir -p "%s"' % img_dir)
        score_dir = os.path.join(log_dir, "score")
        os.system('mkdir -p "%s"' % score_dir)
        config

        # save source code
        if False:
            src_path = os.path.join(log_dir, "src").replace("'", "\\'")
            os.makedirs(src_path, exist_ok=True)
            OmegaConf.save(config, src_path + "/experiment_config.yaml")
            os.system(
                "rsync -av  --exclude='*/runs-debug/' --exclude='*/runs/' --include='*/' --prune-empty-dirs --progress --include='*.py' --exclude='*' $(pwd) %s"
                % src_path
            )
            os.system("zip -r %s/src.zip %s" % (log_dir, src_path))
            os.system("rm -rf %s" % src_path)  # remove src dir.

        # save config yaml.
        OmegaConf.save(config, log_dir + "/experiment_config.yaml")

        config_dict = OmegaConf.to_container(config)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            config_dict["training"]["cuda"] = os.environ["CUDA_VISIBLE_DEVICES"]
        # wandb_logger.log_hyperparams(config_dict)

    logger_list = [
        # wandb_logger,
        tensorboard_logger
    ]
    if config.training.mode != "train":
        logger_list = []
    trainer = pl.Trainer(
        devices=config.devices,
        accelerator=config.accelerator,  # "gpu"
        logger=logger_list,
        #  val_check_interval=0.0, # 0.0:disable, None:total training batch.
        check_val_every_n_epoch=config.training.evaluation_epochs,
        max_steps=config.training.training_steps,
        # reload_dataloaders_every_n_epochs=5,
        reload_dataloaders_every_n_epochs=0,
        log_every_n_steps=50,
        #  strategy="dp",
        # strategy='ddp_find_unused_parameters_true'
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
        # callbacks=[val_call_back,]
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy=DDPStrategy(find_unused_parameters=False),
        precision="16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.fit(model, ckpt_path=ckpt_path_for_trainer)


if __name__ == "__main__":
    my_main()
