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

    # Load checkpoint (raw state_dict saved by your training loop)
    state_dict = torch.load(config.model.checkpoint_path, map_location="cpu")

    # Only do this filter if the list is provided
    if getattr(config.model, "checkpoint_ignore_layres", None):
        for k in list(state_dict.keys()):
            if k in config.model.checkpoint_ignore_layres:
                del state_dict[k]

    # Match your training behavior: use strict flag from config
    strict = bool(getattr(config.model, "strict_checkpoint", True))
    model.model.load_state_dict(state_dict, strict=strict)

    # Where to write test artifacts (CSVs, MIDIs, JSON)
    if not getattr(model, "test_output_dir", None):
        # nicer than "<ckpt>.ckpt_test"
        base = os.path.splitext(os.path.basename(config.model.checkpoint_path))[0]
        outdir = config.get("training", {}).get("log_dir") or f"./runs_eval/{base}"
        model.test_output_dir = outdir
    os.makedirs(model.test_output_dir, exist_ok=True)

    # Trainer for test
    strategy = (
        DDPStrategy(find_unused_parameters=True) if config.devices > 1 else "auto"
    )
    trainer = pl.Trainer(
        logger=[],  # no TB/WB for test
        devices=config.devices,
        accelerator=config.accelerator,  # "gpu"
        strategy=strategy,
        precision="16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.test(model)  # no need for .eval(); Lightning sets it


if __name__ == "__main__":
    my_main()
