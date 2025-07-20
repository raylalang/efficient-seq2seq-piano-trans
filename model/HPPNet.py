import torch
import torch.nn as nn
import torch.nn.functional as Functional
from omegaconf import OmegaConf
import sys
import os
work_dir = os.path.split(__file__)[0] + "/../"
sys.path.append(work_dir)

from model.layers.HPPNetLayers import CNNTrunk, FreqGroupLSTM
from model.Encoder import Encoder as TransformerEncoder
from model.Layers import FixedEmbed, MlpBlock


class HPPNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        model_size = config.encoder_size
        channel_harmonic = config.channel_harmonic
        octave_bins = config.octave_bins
        channel_in = config.channel_in

        self.trunk = CNNTrunk(c_in=channel_in, c_har=channel_harmonic, embedding=model_size, fixed_dilation=octave_bins)

        if config.temporal_layer == "FG-LSTM":
            self.head = FreqGroupLSTM(channel_in= model_size, channel_out=1, lstm_size=model_size, activation="none")
        elif config.temporal_layer == "Attention":
            self.freq_attn = TransformerEncoder(config)
        else:
            raise "Unknown temporal layer: " + config.temporal_layer
        
        if config.squeeze_channel_dim:
            self.linear = nn.Linear(model_size, 1)
            self.linear_out = nn.Linear(88, model_size)
            
        self.encoder_pos_emb = FixedEmbed(self.config.emb_dim)
        self.encoder_outputs_mlp = MlpBlock(emb_dim=config.encoder_size * 360 // 4, intermediate_dim=512, emb_dim_out=config.emb_dim)

    def forward(self, x, mask=None, deterministic=False):
        """_summary_

        Args:
            x (torch.float): shape [B, c_in, T, F] or [B, T, F], spectgram features.
        Returns:
            hidden_states: shape [B, T, F/4, C_out]
        """
        outputs = {}
        if x.ndim == 3:
            x = x[:, None, :, :]
        assert x.ndim == 4
        # [B, c_in, T, F] => [B, c_har, T, F]
        x = self.trunk(x)
        if self.config.temporal_layer == "FG-LSTM":
            hidden_states = self.head(x)
            # => [B, T, F, lstm_size]
            hidden_states = hidden_states.permute([0, 2, 3, 1])
        elif self.config.temporal_layer == "Attention":
            B, C, T, F = x.size()
            # => [B, F, T, C] => [B*F, T, C]
            x = torch.swapdims(x, 1, 3).reshape([B*F, T, C])
            x = self.freq_attn(x)
            # => [B, T, F, C]
            hidden_states = x["encoder_outputs"].reshape([B, F, T, C]).swapdims(1, 2)

        if self.config.squeeze_channel_dim:
            # => [B, T, F, 1] => [B, T, F]
            hidden_states = self.linear(hidden_states).squeeze(dim=3)
            hidden_states = torch.relu(hidden_states)
            B, T, F = hidden_states.size()
            # => [B, T+99, F]
            hidden_states = Functional.pad(hidden_states, [0, 0, 0, 99])
            # => [B, T, F, 100]
            hidden_states = hidden_states.unfold(dimension=1, size=100, step=1)
            # => [B, T, 100, F]
            hidden_states = torch.swapdims(hidden_states, 2, 3)
            # # => [B, T, 100, emb_dim]
            hidden_states = torch.relu( self.linear_out(hidden_states) )
            
        encoder_outputs = hidden_states
        encoder_outputs = torch.relu(encoder_outputs)
        # encoder_outputs = encoder_outputs
        
        
        # # [B, T, F, enc_size] => [B, T, F, 8]
        # encoder_outputs = self.enc2dec_linear(encoder_outputs)
        # [B, T, F, H] => [B, T, F*H] (F*H = emb_dim)
        B, T, F, H = encoder_outputs.size()
        encoder_outputs = encoder_outputs.reshape([B, T, F*H])
        # => [B, T, emb_dim]
        encoder_outputs = self.encoder_outputs_mlp(encoder_outputs)
        
        # Add pos_emb to encoded_seq
        seq_length = encoder_outputs.size()[1]
        # [batch, length, depth] -> [batch, length, emb_dim]
        encoder_output_positions = torch.arange(seq_length)[None, :].to(encoder_outputs.device)
        encoder_outputs = encoder_outputs + self.encoder_pos_emb(encoder_output_positions)
            

        return encoder_outputs

if __name__ == "__main__":
    config = OmegaConf.load("config/model/HPPNet.yaml")
    hppnet = HPPNet(config)
    print(hppnet)

