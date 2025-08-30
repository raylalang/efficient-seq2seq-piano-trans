import os
import sys
work_dir = os.path.split(__file__)[0] + "/../"
import torch
import torch.nn.functional as Functional
import numpy as np
import matplotlib.pyplot as plt
import torchvision
# from data.constants import *

def save_frame_pianoroll(img_path, output_pianoroll, target_pianoroll):
    """_summary_
    Args:
        output_pianoroll (torch.long): [B, T, F]
        target_pianoroll (torch.long): [B, T, F]
    """
    B, T, N = target_pianoroll.size()
    # output_pianoroll = (output_pianoroll >= 0.5).int()
    ones = torch.ones_like(target_pianoroll)
    # => [B, 3, T, 128] => [B, 3, 128, T]
    stacked = torch.stack([1-target_pianoroll, 1 - output_pianoroll, ones], dim = 1).transpose(2,3).float()
    stacked = torchvision.utils.make_grid(stacked, nrow=B, pad_value=0.5, padding=1).permute(1,2,0).cpu().numpy()
    mask = np.all(stacked == [0, 0, 1],axis=-1)
    stacked[mask] = [0.8, 0.8, 0.8] # Set True Positive color to grey.
    plt.imsave(img_path, stacked)

