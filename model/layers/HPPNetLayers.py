from lib2to3.pgen2.token import NT_OFFSET
from math import sin
from multiprocessing import pool
import torch

import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torchvision
import matplotlib.pyplot as plt
import os

from .lstm import BiLSTM

from model.Attention import Multi_Head_Attention
from model.Mask import make_causal_mask, make_harmonic_awared_mask
from model.Layers import FixedEmbed


class FreqGroupLSTM(nn.Module):
    def __init__(self, channel_in, channel_out, lstm_size, activation="sigmoid") -> None:
        super().__init__()

        self.channel_out = channel_out

        self.lstm_size = lstm_size
        self.lstm = BiLSTM(channel_in, lstm_size, bidirectional=True)
        # self.linear = nn.Linear(lstm_size, channel_out)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise "Unknown activation function: %d!"%activation 
            

    def forward(self, x):
        # inputs: [b x c_in x T x freq]
        # outputs: [b x c_out x T x freq]

        b, c_in, t, n_freq = x.size() 

        # => [b x freq x T x c_in] 
        x = torch.permute(x, [0, 3, 2, 1])

        # => [(b*freq) x T x c_in]
        x = x.reshape([b*n_freq, t, c_in])
        # => [(b*freq) x T x lstm_size]
        x = self.lstm(x)
        hidden_states = x
        # => [(b*freq) x T x c_out]
        # x = self.linear(x)
        # => [b x freq x T x c_out]
        # x = x.reshape([b, n_freq, t, self.channel_out])
        hidden_states = hidden_states.reshape([b, n_freq, t, self.lstm_size])

        # x = self.activation(x)

        # => [b x c_out x T x freq]
        # x = torch.permute(x, [0, 3, 2, 1])
        hidden_states = torch.permute(hidden_states, [0, 3, 2, 1])
        # return x, hidden_states
        return hidden_states


class HarmonicAwaredAttention(nn.Module):
    def __init__(self, emb_dim = 128, num_heads = 2, head_dim = 64, num_layers=4, drop_out_rate=0.1, bins_per_octave=48) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Multi_Head_Attention(
                num_heads=num_heads,
                emb_dim=emb_dim,
                head_dim=head_dim,
                dropout_rate=drop_out_rate,
            )
            for _ in range(num_layers)
        ])
        
        self.harmonic_awared_mask = make_harmonic_awared_mask(
            batch_size=1, 
            seq_len=emb_dim, 
            bins_per_octave=bins_per_octave,
            n_har=12,
            har_width=0)
        
        self.pos_emb = FixedEmbed(emb_dim)
        

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): [batch_size, channel, seq_len, freq_bins]

        Returns:
            tensor: [batch_size, channel, seq_len, freq_bins]
        """
        batch_size, channel, seq_len, freq_bins = x.size()
        # => [batch_size, seq_len, freq_bins, channel]
        x = torch.permute(x, [0, 2, 3, 1])
        # => [batch_size * seq_len, freq_bins, channel]
        x = x.reshape([batch_size * seq_len, freq_bins, channel])
        
        pos_idx = torch.arange(x.size()[1])[None, :].to(x.device)
        x = x + self.pos_emb(pos_idx)
        
        mask = self.harmonic_awared_mask
        if mask.size(0) != batch_size * seq_len:
            mask = mask.repeat(batch_size * seq_len, 1,  1, 1)
            self.harmonic_awared_mask = mask
        
        for layer_id, layer in enumerate( self.layers ):
            x = layer(x, x, x, mask=mask)
            
        # => [batch_size, seq_len, freq_bins, channel]
        x = x.reshape([batch_size, seq_len, freq_bins, channel])
        # => [batch_size, channel, seq_len, freq_bins]
        x = torch.permute(x, [0, 3, 1, 2])
        
        return x
    
class HarmonicAwaredEncoder(nn.Module):
    def __init__(self, c_in=1, c_har=16, embedding=128, fixed_dilation=24) -> None:
        super().__init__()

        self.block_1 = get_conv2d_block(c_in, c_har, kernel_size=7)
        self.block_2 = get_conv2d_block(c_har, c_har, kernel_size=7, pool_size=[1, 1])
        self.block_2_5 = get_conv2d_block(c_har, c_har, kernel_size=7, pool_size=[1, 1])

        c3_out = embedding
        
        self.conv_3 = HarmonicDilatedConv(c_har, c3_out)

        self.block_4 = get_conv2d_block(c3_out, c3_out, pool_size=[1, 4], dilation=[1, 48])
        self.block_5 = get_conv2d_block(c3_out, c3_out, pool_size=[1, 1], dilation=[1, 12])
        self.block_6 = get_conv2d_block(c3_out, c3_out, [3,3], pool_size=[1, 1])
        self.block_7 = get_conv2d_block(c3_out, c3_out, [3,3], pool_size=[1, 1])
        self.block_8 = get_conv2d_block(c3_out, c3_out, [3,3], pool_size=[1, 1])

        self.attention = HarmonicAwaredAttention(emb_dim=c3_out)

    def forward(self, log_gram_db):
        x = self.block_1(log_gram_db)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)

        x = self.attention(x)

        x = self.block_5(x)
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x

        return x
    
    

class HarmonicDilatedConv(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 48])
        self.conv_2 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 76])
        self.conv_3 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 96])
        self.conv_4 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 111])
        self.conv_5 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 124])
        self.conv_6 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 135])
        self.conv_7 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 144])
        self.conv_8 = nn.Conv2d(c_in, c_out, [1, 3], padding='same', dilation=[1, 152])
    def forward(self, x):
        x = self.conv_1(x) + self.conv_2(x) + self.conv_3(x) + self.conv_4(x) +\
            self.conv_5(x) + self.conv_6(x) + self.conv_7(x) + self.conv_8(x)
        x = torch.relu(x)
        return x
    
def get_conv2d_block(channel_in,channel_out, kernel_size = [5, 3], pool_size = None, dilation = [1, 1]):
    if(pool_size == None):
        return nn.Sequential( 
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ReLU(),
            # nn.BatchNorm2d(channel_out),
            nn.InstanceNorm2d(channel_out),
            
        )
    else:
        return nn.Sequential( 
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),
            # nn.BatchNorm2d(channel_out),
            nn.InstanceNorm2d(channel_out)
        )


class CNNTrunk(nn.Module):
    def __init__(self, c_in = 1, c_har = 16,  embedding = 128, fixed_dilation = 24) -> None:
        super().__init__()

        self.block_1 = get_conv2d_block(c_in, c_har, kernel_size=7)
        self.block_2 = get_conv2d_block(c_har, c_har, kernel_size=7, pool_size=[1, 1])
        self.block_2_5 = get_conv2d_block(c_har, c_har, kernel_size=7, pool_size=[1, 1])

        c3_out = embedding
        
        self.conv_3 = HarmonicDilatedConv(c_har, c3_out)

        self.block_4 = get_conv2d_block(c3_out, c3_out, pool_size=[1, 4], dilation=[1, 48])
        self.block_5 = get_conv2d_block(c3_out, c3_out, pool_size=[1, 1], dilation=[1, 12])
        self.block_6 = get_conv2d_block(c3_out, c3_out, [3,3], pool_size=[1, 1])
        self.block_7 = get_conv2d_block(c3_out, c3_out, [3,3], pool_size=[1, 1])
        self.block_8 = get_conv2d_block(c3_out, c3_out, [3,3], pool_size=[1, 1])
        # self.conv_9 = nn.Conv2d(c3_out, 64,1)
        # self.conv_10 = nn.Conv2d(64, 1, 1)

    def forward(self, log_gram_db):
        # inputs: [b x 2 x T x n_freq] , [b x 1 x T x 88]
        # outputs: [b x T/16 x 88]


        # img_path = 'logspecgram_preview.png'
        # if not os.path.exists(img_path):
        #     img = torch.permute(log_gram_db, [2, 0, 1]).reshape([352, 640*4]).detach().cpu().numpy()
        #     # x_grid = torchvision.utils.make_grid(x.swapaxes(0, 1), pad_value=1.0).swapaxes(0, 2).detach().cpu().numpy()
        #     # plt.imsave(img_path, (x_grid+80)/100)
        #     plt.imsave(img_path, img)

        # => [b x 1 x T x 352]
        # x = torch.unsqueeze(log_gram_db, dim=1)



        x = self.block_1(log_gram_db)
        x = self.block_2(x)
        x = self.block_2_5(x)
        x = self.conv_3(x)
        x = self.block_4(x)
        # => [b x 1 x T x 88]

        x = self.block_5(x)
        # => [b x ch x T x 88]
        x = self.block_6(x) # + x
        x = self.block_7(x) # + x
        x = self.block_8(x) # + x
        # x = self.conv_9(x)
        # x = torch.relu(x)
        # x = self.conv_10(x)
        # x = torch.sigmoid(x)

        return x
    
class Head(nn.Module):
    def __init__(self, model_size, channel_out=1, activation="sigmoid") -> None:
        super().__init__()
        self.head = FreqGroupLSTM(model_size, channel_out, model_size, activation=activation)
    def forward(self, x):
        # input: [B x model_size x T x 88]
        # output: [B x channel_out x T x 88]
        y = self.head(x)
        return y