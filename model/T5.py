"""
Converted jax-based code https://github.com/magenta/mt3/blob/main/mt3/network.py#L158 to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as Functional

from model.Encoder import Encoder
from model.Decoder import Decoder, CompoundDecoder
from model.Layers import *
from model.Mask import *

# from model.HPPNet import HPPNet
from data.constants import *

from torch.nn.utils.rnn import pad_sequence
from config.utils import DictToObject

from tqdm import tqdm
from utils.log_memory_usage import profile_cuda_memory


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        config = DictToObject(dict(config))
        config.dtype = eval(config.dtype)
        # config.dtype = torch.float32 #
        self.config = config
        # select encoder
        if config.encoder_name == "TransformerEncoder":
            self.encoder = Encoder(config)
        # elif config.encoder_name == "CNNEncoder":
        #     self.encoder = CNNEncoder(config)
        # elif config.encoder_name == "HPPNetEncoder":
        #     self.encoder = HPPNet(config)
        # elif config.encoder_name == "hFT_Transformer_Encoder":
        #     self.encoder = hFT_Transformer_Encoder(config)
        else:
            raise "Unknown encoder: " + config.encoder_name
        # froze encoder
        if config.froze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # select decoder
        # self.decoder = Decoder(config=config)
        sub_token_names = [t.get_class_name() for t in sm_tokenizer.token_type_list]

        if config.decoder_name == "TransformerDecoder":
            self.decoder = Decoder(config=config)
        elif config.decoder_name == "CompoundTransformerDecoder":
            self.decoder = CompoundDecoder(
                config=config, sub_token_names=sub_token_names
            )

        D = config.emb_dim
        # Projections for concat along D (2D when one of beat/bar; 3D when both)
        self._fuse2 = nn.Linear(2 * D, D)
        self._fuse3 = nn.Linear(3 * D, D)
        self._fuse_ln = nn.LayerNorm(D)
        self._fuse_drop = nn.Dropout(getattr(config, "dropout_rate", 0.1))
        # Optional residual gate (starts at 0 so it’s safe)
        self._fuse_alpha = nn.Parameter(torch.zeros(1))
        self._ctx_mha = nn.MultiheadAttention(
            embed_dim=D,
            num_heads=config.num_heads,
            dropout=getattr(config, "dropout_rate", 0.1),
            batch_first=True,
        )
        self._ctx_ln = nn.LayerNorm(D)
        self._ctx_drop = nn.Dropout(getattr(config, "dropout_rate", 0.1))
        self._attn_alpha = nn.Parameter(torch.zeros(1))  # gated residual

        self.pad_token = TOKEN_PAD
        self.eos_token = TOKEN_END

    def encode(
        self, encoder_input_tokens, encoder_segment_ids=None, enable_dropout=True
    ):
        assert encoder_input_tokens.ndim == 3  # (batch, length, depth)

        encoder_mask = make_attention_mask(
            torch.ones(encoder_input_tokens.shape[:-1]),
            torch.ones(encoder_input_tokens.shape[:-1]),
            dtype=self.config.dtype,
        )

        if encoder_segment_ids is not None:
            encoder_mask = combine_masks(
                encoder_mask,
                make_attention_mask(
                    encoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype,
                ),
            )

        encoder_mask = encoder_mask.to(encoder_input_tokens.device)

        if self.config.froze_encoder:
            with torch.no_grad():
                encoder_outputs = self.encoder(
                    encoder_input_tokens, encoder_mask, deterministic=not enable_dropout
                )
                encoder_outputs = encoder_outputs.detach()
        else:
            encoder_outputs = self.encoder(
                encoder_input_tokens, encoder_mask, deterministic=not enable_dropout
            )
        return encoder_outputs

    def decode(
        self,
        encoded,
        encoder_input_tokens,
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        decoder_positions=None,
        enable_dropout=True,
        decode=False,  # decode: Whether to prepare and use an autoregressive cache
        max_decode_length=None,
        use_preframe_tokens=True,
        decoder_targets_frame_index=None,
        encoder_decoder_mask=None,
    ):

        encoder_decoder_mask_0 = encoder_decoder_mask

        # We use pooling to reduce decoder sequence length.
        # decoder_target_tokens here just use for generating decoder_mask and attention_mask.
        B, T = decoder_target_tokens.size()
        decoder_target_tokens = decoder_target_tokens  # [:, :T//SEQUENCE_POOLING_SIZE]

        if decode:
            decoder_mask = None
            encoder_decoder_mask = make_attention_mask(
                torch.ones_like(decoder_target_tokens).to(encoded.device),
                torch.ones(encoder_input_tokens.shape[:-1]).to(encoded.device),
                dtype=self.config.dtype,
            )
        else:
            decoder_mask = make_decoder_mask(
                decoder_target_tokens=decoder_target_tokens,
                dtype=self.config.dtype,
                decoder_segment_ids=decoder_segment_ids,
            )
            encoder_decoder_mask = make_attention_mask(
                decoder_target_tokens > 0,
                torch.ones(encoder_input_tokens.shape[:-1]).to(encoded.device),
                dtype=self.config.dtype,
            )

        if encoder_segment_ids is not None:
            if decode:
                raise ValueError(
                    "During decoding, packing should not be used but `encoder_segment_ids` was passed to `Transformer.decode`."
                )

            encoder_decoder_mask = combine_masks(
                encoder_decoder_mask,
                make_attention_mask(
                    decoder_segment_ids,
                    encoder_segment_ids,
                    torch.equal,
                    dtype=self.config.dtype,
                ),
            )

        # If the encoder_decoder_mask is provided, we use the default mask.
        if encoder_decoder_mask_0 is not None:
            if (
                hasattr(self.config, "encoder_decoder_slide_window_size")
                and self.config.encoder_decoder_slide_window_size > 0
            ):
                encoder_decoder_mask = encoder_decoder_mask_0

        # Hierarchical pooling
        encoded_pooling_dict = None
        if (
            hasattr(self.config, "cross_attention_hierarchy_pooling")
            and self.config.cross_attention_hierarchy_pooling
        ):
            encoded_pooling_dict = {}
            for pooling in set(self.config.pooling_sizes):
                encoded_i = encoded
                if pooling > 1:
                    encoded_i = encoded_i.reshape(
                        encoded_i.size(0),
                        encoded_i.size(1) // pooling,
                        pooling,
                        encoded_i.size(2),
                    ).mean(
                        dim=2
                    )  # [batch, 1, encoded_length, emb_dim]
                encoded_pooling_dict[pooling] = (
                    encoded_i  # Save the encoded_i for later use
                )
            encoded = None

        decoder_output_dict = self.decoder(
            encoded,
            encoded_pooling_dict=encoded_pooling_dict,
            decoder_input_tokens=decoder_input_tokens,
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            deterministic=not enable_dropout,
            decode=decode,
            decoder_targets_frame_index=decoder_targets_frame_index,
        )
        return decoder_output_dict

    def _shift_right(self, input_ids, shift_step=1):
        BOS = TOKEN_START

        shifted_input_ids = torch.zeros_like(input_ids)
        if shift_step is None:
            shift_step = SEQUENCE_POOLING_SIZE
        shifted_input_ids[..., shift_step:] = input_ids[..., :-shift_step].clone()
        shifted_input_ids[..., :shift_step] = BOS

        return shifted_input_ids

    def forward(
        self,
        encoder_input_tokens=None,
        decoder_target_tokens=None,
        decoder_input_tokens=None,
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        encoder_positions=None,
        decoder_positions=None,
        enable_dropout=True,
        decode=False,
        dur_inputs=None,
        dur_targets=None,
        decoder_targets_frame_index=None,
        encoder_decoder_mask=None,
        additional_encoder_memory=None,
    ):
        res_dict = {}

        if decoder_input_tokens is None:
            decoder_input_tokens = self._shift_right(
                decoder_target_tokens, shift_step=1
            )

        encoder_outputs = self.encode(
            encoder_input_tokens,
            encoder_segment_ids=encoder_segment_ids,
            enable_dropout=enable_dropout,
        )

        fuse = getattr(self.config, "context_fuse", None)

        if additional_encoder_memory is not None:
            if isinstance(additional_encoder_memory, dict) and fuse in (
                "concat",
                "attn",
            ):
                if fuse == "concat":
                    encoder_outputs = self._fuse_context_concat(
                        encoder_outputs, additional_encoder_memory
                    )
                else:  # "attn"
                    encoder_outputs = self._fuse_context_attn(
                        encoder_outputs, additional_encoder_memory
                    )
                # NOTE: T unchanged; do NOT extend encoder_decoder_mask here.
            else:
                # Legacy: time-append
                if isinstance(additional_encoder_memory, dict):
                    additional_encoder_memory = self._collect_ctx(
                        additional_encoder_memory
                    )
                if additional_encoder_memory is not None:
                    additional_encoder_memory = additional_encoder_memory.to(
                        encoder_outputs.dtype
                    ).to(encoder_outputs.device)
                    encoder_outputs = torch.cat(
                        [encoder_outputs, additional_encoder_memory], dim=1
                    )
                    if encoder_decoder_mask is not None:
                        B, H, T_dec, T_enc = encoder_decoder_mask.shape
                        T_add = additional_encoder_memory.size(1)
                        add_mask = torch.ones(
                            (B, H, T_dec, T_add),
                            dtype=encoder_decoder_mask.dtype,
                            device=encoder_decoder_mask.device,
                        )
                        encoder_decoder_mask = torch.cat(
                            [encoder_decoder_mask, add_mask], dim=-1
                        )

        decoder_output_dict = self.decode(
            encoder_outputs,
            encoder_outputs,
            decoder_input_tokens,
            decoder_target_tokens,
            encoder_segment_ids=encoder_segment_ids,
            decoder_segment_ids=decoder_segment_ids,
            decoder_positions=decoder_positions,
            enable_dropout=enable_dropout,
            decode=decode,
            decoder_targets_frame_index=decoder_targets_frame_index,
            encoder_decoder_mask=encoder_decoder_mask,
        )

        res_dict.update(decoder_output_dict)

        return res_dict

    # @profile_cuda_memory
    def generate(
        self,
        encoder_inputs,
        target_seq_length=1024,
        break_on_eos=False,
        global_rank=0,
        additional_encoder_memory=None,
    ):
        self.decoder.initialize_decoder_cache()

        batch_size, T, n_mel = encoder_inputs.size()
        gen_tokens = torch.full(
            (batch_size, target_seq_length),
            TOKEN_PAD,
            dtype=torch.long,
            device=encoder_inputs.device,
        )
        eos_flags = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)
        curr_frame_index = torch.zeros(batch_size, dtype=int).to(encoder_inputs.device)
        max_num_tokens = 0
        encoded = self.encode(encoder_inputs, enable_dropout=False)

        # concat memory for inference
        fuse = getattr(self.config, "context_fuse", None)
        if additional_encoder_memory is not None:
            if isinstance(additional_encoder_memory, dict) and fuse in (
                "concat",
                "attn",
            ):
                if fuse == "concat":
                    encoded = self._fuse_context_concat(
                        encoded, additional_encoder_memory
                    )
                else:
                    encoded = self._fuse_context_attn(
                        encoded, additional_encoder_memory
                    )
                # T unchanged
                legacy_time_append = False
            else:
                if isinstance(additional_encoder_memory, dict):
                    additional_encoder_memory = self._collect_ctx(
                        additional_encoder_memory
                    )
                legacy_time_append = additional_encoder_memory is not None
                if legacy_time_append:
                    encoded = torch.cat([encoded, additional_encoder_memory], dim=1)
        else:
            legacy_time_append = False

        # build masks with the new encoder length
        T = encoded.size(1)
        decoder_mask = torch.tril(
            torch.ones((target_seq_length, target_seq_length), dtype=self.config.dtype),
            diagonal=0,
        ).to(encoder_inputs.device)
        decoder_mask = (
            decoder_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.config.num_heads, -1, -1)
        )
        encoder_decoder_mask = torch.ones(
            (batch_size, self.config.num_heads, target_seq_length, T),
            dtype=self.config.dtype,
        ).to(
            encoder_inputs.device
        )  # [batch_size, 1, target_seq_length, T]

        pred_step = 1
        if (
            hasattr(self.config, "predict_onset_by_AttnMap")
            and self.config.predict_onset_by_AttnMap
        ):
            pred_step = 2

        use_kv_cache = True
        curr_token = torch.full(
            (batch_size, pred_step),
            TOKEN_START,
            dtype=torch.long,
            device=encoder_inputs.device,
        )  # [batch_size, pred_step]
        # for i in tqdm(
        #     range(0, target_seq_length, pred_step),
        #     desc="Generating tokens (rank %d)" % global_rank,
        # ):
        print("Generating tokens")
        for i in range(0, target_seq_length, pred_step):
            encoded_i = encoded[:, :T, :]  # [batch_size, T, n_mel]

            if use_kv_cache:
                decoder_input_tokens_i = curr_token  #
                encoder_decoder_mask_i = encoder_decoder_mask[
                    :, :, i : i + pred_step, :T
                ]
                decoder_mask_i = decoder_mask[:, :, i : i + pred_step, : i + pred_step]
                decoder_positions_i = (
                    torch.arange(i, i + pred_step, device=encoder_inputs.device)
                    .unsqueeze(0)
                    .expand([batch_size, -1])
                )  # [1, 1, pred_step]
            else:
                decoder_input_tokens_i = self._shift_right(
                    gen_tokens[:, : i + pred_step], shift_step=1
                )  # [batch_size, i+pred_step]
                encoder_decoder_mask_i = encoder_decoder_mask[
                    :, :, : i + pred_step, :T
                ]  # [batch_size, 1, i+pred_step, T]
                decoder_mask_i = decoder_mask[:, :, : i + pred_step, : i + pred_step]
                decoder_positions_i = None

            # Slide window for encoder-decoder attention
            if (
                i % 3 != 0
                and hasattr(self.config, "encoder_decoder_slide_window_size")
                and self.config.encoder_decoder_slide_window_size > 0
            ):
                W = int(self.config.encoder_decoder_slide_window_size)
                # integer division already floors; also clamp to last full/partial window
                start = (curr_frame_index // W) * W  # [B]
                start = torch.clamp(start, max=max(0, T - 1)).long()  # stay in [0, T-1]

                # slice per batch (handles last partial window without padding)
                enc_chunks, mask_chunks = [], []
                for b in range(batch_size):
                    s = int(start[b].item())
                    e = min(T, s + W)
                    enc_chunks.append(encoded[b : b + 1, s:e, :])
                    mask_chunks.append(encoder_decoder_mask_i[b : b + 1, :, :, s:e])
                encoded_i = torch.cat(enc_chunks, dim=0)
                encoder_decoder_mask_i = torch.cat(mask_chunks, dim=0)

                # ALWAYS re-append context memory if using legacy time-append
                if additional_encoder_memory is not None and legacy_time_append:
                    T_add = additional_encoder_memory.size(1)
                    add_mem = additional_encoder_memory.to(encoded_i.dtype).to(
                        encoded_i.device
                    )
                    encoded_i = torch.cat([encoded_i, add_mem], dim=1)
                    add_mask = torch.ones(
                        (
                            batch_size,
                            self.config.num_heads,
                            encoder_decoder_mask_i.size(2),
                            T_add,
                        ),
                        dtype=encoder_decoder_mask_i.dtype,
                        device=encoder_decoder_mask_i.device,
                    )
                    encoder_decoder_mask_i = torch.cat(
                        [encoder_decoder_mask_i, add_mask], dim=-1
                    )

            # Hierarchical pooling
            encoded_pooling_dict = None
            if (
                hasattr(self.config, "cross_attention_hierarchy_pooling")
                and self.config.cross_attention_hierarchy_pooling
            ):
                encoded_pooling_dict = {}
                for pooling in set(self.config.pooling_sizes):
                    encoded_pool = encoded_i
                    if pooling > 1:
                        encoded_pool = encoded_pool.reshape(
                            batch_size,
                            encoded_pool.size(1) // pooling,
                            pooling,
                            encoded_pool.size(2),
                        ).mean(
                            dim=2
                        )  # [batch, 1, encoded_length, emb_dim]
                    encoded_pooling_dict[pooling] = (
                        encoded_pool  # Save the encoded_i for later use
                    )
                encoded_i = None

            decoder_output_dict = self.decoder(
                encoded_i,
                encoded_pooling_dict=encoded_pooling_dict,
                decoder_input_tokens=decoder_input_tokens_i,
                decoder_positions=decoder_positions_i,
                decoder_mask=decoder_mask_i,
                encoder_decoder_mask=encoder_decoder_mask_i,
                deterministic=True,
                decode=use_kv_cache,
            )

            # Take just the step we’re decoding and slice to actual vocab size
            logits_step = decoder_output_dict["decoder_outputs"][
                :, -pred_step:, : self.config.vocab_size
            ]

            # (A) Optional “format check” now reads from logits (argmax identical to softmax argmax)
            if getattr(self.config, "hybrid_global_local_cross_attn", False):
                tokens_ori = logits_step.argmax(dim=-1)
                eos_flags_int = eos_flags.int()
                for b in range(batch_size):
                    token_i = tokens_ori[b, 0].item()
                    if eos_flags_int[b] == 1:
                        continue
                    if (
                        Tokenizer.TokenOnset.is_instance(token_i)
                        or token_i == self.eos_token
                    ):
                        token_type_i = Tokenizer.TokenOnset
                    elif Tokenizer.TokenPitch.is_instance(token_i):
                        token_type_i = Tokenizer.TokenPitch
                    elif (
                        Tokenizer.TokenVel.is_instance(token_i)
                        or token_i == TOKEN_BLANK
                    ):
                        token_type_i = Tokenizer.TokenVel
                    elif Tokenizer.TokenNoteOff.is_instance(token_i):
                        token_type_i = Tokenizer.TokenNoteOff
                    else:
                        token_type_i = "Other"
                    if i % 3 == 0:
                        dest_token_type_i = (Tokenizer.TokenOnset,)
                    elif i % 3 == 1:
                        dest_token_type_i = (
                            Tokenizer.TokenPitch,
                            Tokenizer.TokenNoteOff,
                        )
                    elif i % 3 == 2:
                        prev_token_i = gen_tokens[b, i - 1].item()
                        if (
                            Tokenizer.TokenNoteOff.is_instance(prev_token_i)
                            and token_i != TOKEN_BLANK
                        ):
                            print(
                                f"Warning: velocity after NoteOff should be {TOKEN_BLANK}, got {token_i}."
                            )
                        dest_token_type_i = (Tokenizer.TokenVel,)
                    if token_type_i not in dest_token_type_i:
                        print(
                            f"Warning: token {token_i} at (b={b}, i={i}) unexpected. Expected {dest_token_type_i}, got {token_type_i}."
                        )

            # (B) Apply “forcing” as **logit biases** (equivalent to adding log-prob mass)
            if getattr(self.config, "hybrid_global_local_cross_attn", False):
                if i % 3 == 0:  # Onset or EOS
                    b, e = Tokenizer.TokenOnset.get_bound()
                    logits_step[:, :, b:e] += 1.0
                    logits_step[:, :, self.eos_token] += 1.0
                elif i % 3 == 1:  # Pitch or NoteOff
                    b1, e1 = Tokenizer.TokenPitch.get_bound()
                    logits_step[:, :, b1:e1] += 1.0
                    b2, e2 = Tokenizer.TokenNoteOff.get_bound()
                    logits_step[:, :, b2:e2] += 1.0
                elif i % 3 == 2:  # Velocity or BLANK
                    b, e = Tokenizer.TokenVel.get_bound()
                    logits_step[:, :, b:e] += 1.0
                    logits_step[:, :, TOKEN_BLANK] += 1.0

            curr_token = logits_step.argmax(dim=-1)  # [batch_size, pred_step]

            gen_tokens[:, i : i + pred_step] = curr_token
            max_num_tokens += pred_step
            # Check  EOS
            eos_flags = eos_flags | (curr_token[:, 0] == self.eos_token).to(
                eos_flags.dtype
            )
            if break_on_eos and eos_flags.int().sum() == batch_size:
                break

            # Update current frame index
            gen_onsets = curr_token[:, 0]
            frames_per_second = DEFAULT_SAMPLE_RATE / DEFAULT_HOP_WIDTH
            eos_flags_int = eos_flags.int()
            for b in range(batch_size):
                onset_i = gen_onsets[b].item()
                if eos_flags_int[b] == 1:
                    curr_frame_index[b] = T - 1
                    continue
                if Tokenizer.TokenOnset.is_instance(
                    onset_i
                ):  # Check if it is a onset token
                    onset_val = Tokenizer.TokenOnset.get_value(onset_i)
                    onset_sec = onset_val / Tokenizer.ONSET_SEC_UP_SAMPLING
                    onset_frame_index = int(onset_sec * frames_per_second)
                    curr_frame_index[b] = min(T - 1, onset_frame_index)

        return gen_tokens  # [:, :max_num_tokens]

    def _collect_ctx(self, add_mem_dict, return_mask=False):
        mems, masks = [], []
        if "beat" in add_mem_dict and add_mem_dict["beat"] is not None:
            mems.append(add_mem_dict["beat"])
            if return_mask and "beat_mask" in add_mem_dict:
                masks.append(add_mem_dict["beat_mask"])
        if "bar" in add_mem_dict and add_mem_dict["bar"] is not None:
            mems.append(add_mem_dict["bar"])
            if return_mask and "bar_mask" in add_mem_dict:
                masks.append(add_mem_dict["bar_mask"])
        if not mems:
            return (None, None) if return_mask else None
        C = torch.cat(mems, dim=1)  # [B,Kc,D]
        if not return_mask or not masks:
            return (C, None) if return_mask else C
        M = torch.cat(masks, dim=1)  # [B,Kc], 1=valid, 0=pad
        return C, M

    def _fuse_context_concat(self, E_main, add_mem_dict):
        """Mean over time for each ctx stream, concat on D, project to D, gated residual."""
        B, T, D = E_main.shape
        tiles = [E_main]
        if "beat" in add_mem_dict:
            c = add_mem_dict["beat"].mean(dim=1, keepdim=True).expand(B, T, D)
            tiles.append(c)
        if "bar" in add_mem_dict:
            c = add_mem_dict["bar"].mean(dim=1, keepdim=True).expand(B, T, D)
            tiles.append(c)
        if len(tiles) == 1:
            return E_main
        E_cat = torch.cat(tiles, dim=-1)  # [B, T, n*D]
        if E_cat.size(-1) == 2 * D:
            fused = self._fuse2(E_cat)
        else:
            fused = self._fuse3(E_cat)
        fused = self._fuse_ln(self._fuse_drop(fused))
        return E_main + self._fuse_alpha * fused

    def _fuse_context_attn(self, E_main, add_mem_dict):
        C, M = self._collect_ctx(add_mem_dict, return_mask=True)  # [B,Kc,D], [B,Kc]
        if C is None:
            return E_main
        kpm = None
        if M is not None:
            # MultiheadAttention expects True where positions are PAD
            kpm = ~(M > 0)  # [B,Kc]
        attn_out, _ = self._ctx_mha(
            E_main, C, C, key_padding_mask=kpm, need_weights=False
        )
        attn_out = self._ctx_ln(self._ctx_drop(attn_out))
        return E_main + self._attn_alpha * attn_out
