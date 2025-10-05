from train import MT3Trainer
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os


@hydra.main(config_path="config", config_name="main_config", version_base=None)
def my_main(config: OmegaConf):
    # Force test mode
    config.training.mode = "test"

    # Safety: require a proper Lightning checkpoint
    ckpt_path = config.model.checkpoint_path
    assert ckpt_path and os.path.exists(
        ckpt_path
    ), f"Missing Lightning ckpt: {ckpt_path}"

    # Build module *from* Lightning checkpoint (loads weights automatically)
    model = MT3Trainer.load_from_checkpoint(ckpt_path, config=config)
    print(model)

    # Strategy: DDP only if >1 device; otherwise let Lightning pick.
    strategy = (
        DDPStrategy(find_unused_parameters=True) if int(config.devices) > 1 else "auto"
    )

    trainer = pl.Trainer(
        logger=[],
        devices=config.devices,
        accelerator=config.accelerator,
        strategy=strategy,
        precision="16-mixed",
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )

    trainer.test(model)


if __name__ == "__main__":
    my_main()
