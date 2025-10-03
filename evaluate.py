from train import MT3Trainer
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os


@hydra.main(config_path="config", config_name="main_config", version_base=None)
def my_main(config: OmegaConf):
    # Force test mode
    config.training.mode = "test"

    # Build module
    model = MT3Trainer(config)
    print(model)

    # Load checkpoint
    state_dict = torch.load(config.model.checkpoint_path)

    # Remove keys that are in the ignore list.
    for key in list(state_dict.keys()):
        if key in config.model.checkpoint_ignore_layers:
            print(f"Removing key {key} from state_dict")
            del state_dict[key]

    model.model.load_state_dict(state_dict, strict=True)

    strategy = (
        DDPStrategy(find_unused_parameters=True) if config.devices > 1 else "auto"
    )

    trainer = pl.Trainer(
        logger=[],
        devices=config.devices,
        accelerator=config.accelerator,
        strategy=strategy,
        precision="16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.test(model)


if __name__ == "__main__":
    my_main()
