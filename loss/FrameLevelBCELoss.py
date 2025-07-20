import torch
import torch.nn as nn
import torch.nn.functional as Functional

class FrameLevelBCELoss(nn.Module):
    def __init__(self, config = None) -> None:
        super().__init__()
        # self.config = config
        # self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.criterion = nn.BCEWithLogitsLoss(reduction="sum")
    def forward(self, outputs, targets, targets_mask):
        """_summary_

        Args:
            outputs (torch.float): shape [B, T, F], F = 128
            targets (torch.long): shape [B, T, F], F = 128
            targets_mask (torch.long): shape [B, T, F]
            loss_name (str):
        Returns:
            loss:
            loss_pos:
            loss_neg:
        """
        assert outputs.ndim == 3 and targets.ndim == 3
        
        B, T, F = targets.size()
        
        # => [B, T]
        num = torch.clip(targets.sum(dim=2) , 0, 1).sum() * F
        
        loss = self.criterion(outputs, targets.float())
        loss *= 20
        
        if num == 0:
            loss *= 0
        else:
            loss /= num
        
        
        return loss
        
        vocab_size = outputs.size()[2]
        # assert targets.min() == 0 and targets.max() == 1
        idx_pos = targets == 1
        idx_neg = targets == 0
        assert idx_pos.sum() + idx_neg.sum() == (targets>= 0).sum() # Ensure targets are binary: 0/1.
        losses = self.criterion(outputs * targets_mask, targets.float() * targets_mask)
        
        num = targets_mask.sum()
        loss_sum = losses.sum()
        if(num ==0 ):
            loss = loss_sum * 0
        else:
            loss = loss_sum / num

        
        return loss
    
        # loss = losses.mean()


        loss_pos = 0
        loss_neg = 0
        if idx_pos.sum() > 0:
            loss_pos = losses[idx_pos].mean()
        if idx_neg.sum() > 0:
            loss_neg = losses[idx_neg].mean()
        loss = loss_pos + loss_neg
        loss_dict = {
            loss_name: loss,
            loss_name + "_pos": loss_pos,
            loss_name + "_neg": loss_neg,
        }
        return loss, loss_dict
    

if __name__ == "__main__":
    criterion = FrameLevelBCELoss()
    outputs = torch.tensor([-1, -2, -3, 0.6, 0.7])
    targets = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    targets = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    targets = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
    loss, loss_pos, loss_neg = criterion(outputs, targets)
    print(loss, loss_pos, loss_neg)
    exit()

