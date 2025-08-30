from train import MT3Trainer
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
from collections import defaultdict


@hydra.main(config_path="config", config_name="main_config", version_base = None)
def my_main(config: OmegaConf):
    # Create model.
    config.training.mode = "test"
    model = MT3Trainer(config)
    print(model)
    # Load checkpoint.
    state_dict = torch.load(config.model.checkpoint_path) # , map_location=torch.device('cpu')
    
    # Remove keys that are in the ignore list.
    for key in list(state_dict.keys()):
        if key in config.model.checkpoint_ignore_layres:
            print(f"Removing key {key} from state_dict")
            del state_dict[key]
    
    model.model.load_state_dict(state_dict, strict=True)
    
    trainer = pl.Trainer(
        logger=[],
        devices=config.devices, # 1 [1,2, 4, 5, 6,7]
        accelerator=config.accelerator, # "gpu"
        strategy=DDPStrategy(find_unused_parameters=True),
        )

    model.test_output_dir = config.model.checkpoint_path + "_test"
    os.makedirs(model.test_output_dir, exist_ok=True)
    trainer.test(model.eval())
    
if __name__ == "__main__":
    my_main()