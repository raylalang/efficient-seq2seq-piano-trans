import torch
import torch.nn as nn
import torch.nn.functional as Functional

from data.constants import *

class FrameLevelMultiPitchCELoss(nn.Module):
    def __init__(self, config = None) -> None:
        super().__init__()
        self.topk = 5
        # self.config = config
        # self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD, reduction="sum")
    def forward(self, outputs, targets, targets_mask):
        """_summary_

        Args:
            outputs (torch.float): shape [B, T, F], F = 128
            targets (torch.long): shape [B, T, F], F = 128 , multihot
            targets_mask (torch.long): shape [B, T, F]
            loss_name (str):
        Returns:
            loss:
            loss_pos:
            loss_neg:
        """
        assert outputs.ndim == 3 and targets.ndim == 3
        assert targets.size()[2] == 128
        
        targets = targets.clone()
        
        B, T, F = targets.size()
        outputs = outputs.reshape([B*T, F])
        targets = targets.reshape([B*T, F])
        
        # pitch_num = targets.sum(dim=1)
        
        # [B*T, F] => [B*T, topk], [B*T, topk]
        (values, tokens) = torch.topk(targets, k=self.topk, dim=-1)
        tokens[values == 0] = TOKEN_PAD
        
        batch_idx = torch.arange(0, B*T, device=outputs.device)
        
        loss = 0
        
        min_out = min(outputs.min(), -9999999999.0)
        
        for i in range(self.topk):
            outputs_ignore = torch.clone(targets)
            tokens_i = tokens[:, i] # => [B*T,]
            values_i = values[:, i] # => [B*T,]
            if(values_i.sum() == 0):
                break
            tokens_i[values_i==0] = 0
            assert tokens_i.max() < 128
            outputs_ignore[batch_idx, tokens_i] = 0 # 
            
            outputs_ignore = outputs_ignore.float() * min_out
            outputs_ignore = outputs_ignore.detach()
            outputs_i = outputs + outputs_ignore
            tokens_i[values_i==0] = TOKEN_PAD
            loss += self.criterion(outputs_i, tokens_i.detach().clone())
        
        
        num = targets.sum()
        if num > 0:
            loss = loss / num
        else:
            loss = loss * 0

        
        return loss



if __name__ == "__main__":
    criterion = FrameLevelMultiPitchCELoss()
    outputs = torch.tensor([-1, -2, -3, 0.6, 0.7])
    targets = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    targets = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    targets = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
    loss, loss_pos, loss_neg = criterion(outputs, targets)
    print(loss, loss_pos, loss_neg)
    exit()

