import torch
import torch.nn as nn
import torch.nn.functional as Functional

from data.constants import *


class CrossEntropyLoss(nn.Module):
    def __init__(self, config = None) -> None:
        super().__init__()
        self.config = config
        class_weights = torch.ones([self.config.model.vocab_size, ])
        # class_weights[128:256] = config.data.offset_weight # set Offset weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=TOKEN_PAD) # config.data.TOKEN_PAD
        self.TOKEN_PAD = TOKEN_PAD
    def forward(self, outputs:torch.tensor, targets:torch.tensor, targets_mask:torch.tensor):
        """ Cross Entropy Loss with Softmax Layer.
        Args:
            outputs : shape [batch, ..., n_class]
            targets : shape [batch, ...]
            targets_mask : shape [batch, ...]
        Returns:
            torch.tensor: loss
        """
        targets = torch.clone(targets)
        targets[targets_mask == 0] = self.TOKEN_PAD
        outputs_dim = list(range(outputs.ndim))
        # move the last dim to the dim 1 while keep the order of others the same.
        outputs_dim = outputs_dim[:1] + outputs_dim[-1:] + outputs_dim[1:-1]
        outputs = outputs.permute(outputs_dim)
        self.criterion = self.criterion.to(outputs.device)
        loss = self.criterion(outputs, targets)
        return loss