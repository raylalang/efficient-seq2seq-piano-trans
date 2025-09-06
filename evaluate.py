from train import MT3Trainer
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"


@hydra.main(config_path="config", config_name="main_config", version_base=None)
def my_main(config: OmegaConf):
    assert (
        config.model.checkpoint_path is not None
    ), "Set +model.checkpoint_path=...</cpt/steps_xxx.ckpt>"
    ckpt_path = Path(str(config.model.checkpoint_path))
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # Locate the run's saved config (if any) and MERGE so CLI overrides WIN.
    candidates = [
        ckpt_path.parent.parent
        / "experiment_config.yaml",  # runs/.../experiment_config.yaml
        ckpt_path.parent
        / "experiment_config.yaml",  # runs/.../cpt/experiment_config.yaml (fallback)
    ]
    run_cfg_path = next((p for p in candidates if p.is_file()), None)
    if run_cfg_path is not None:
        run_cfg = OmegaConf.load(str(run_cfg_path))
        config = OmegaConf.merge(run_cfg, config)  # right order: CLI wins

    # Force test mode and pin the exact ckpt path
    config.training.mode = "test"
    config.model.checkpoint_path = str(ckpt_path)

    # Make single dataset_dir override effective
    if getattr(config.data, "dataset_dir", None):
        config.data.dataset_dirs = [config.data.dataset_dir]

    # Build module
    model = MT3Trainer(config)

    # Load weights (Lightning ckpt or raw state_dict)
    obj = torch.load(config.model.checkpoint_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = {
            k.replace("model.", "", 1): v
            for k, v in obj["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        sd = obj
    ignore = set(getattr(config.model, "checkpoint_ignore_layers", []) or [])
    for k in list(sd.keys()):
        if k in ignore:
            del sd[k]
    strict = bool(getattr(config.model, "strict_checkpoint", True))
    model.model.load_state_dict(sd, strict=strict)

    # Where to write eval artifacts (CSV/MIDI/JSON)
    outdir = getattr(config.training, "log_dir", None)
    if not outdir:
        if run_cfg_path is not None:
            outdir = os.path.join(str(run_cfg_path.parent), "eval")
        else:
            base = os.path.splitext(os.path.basename(config.model.checkpoint_path))[0]
            outdir = f"./runs_eval/{base}"
    model.test_output_dir = outdir
    os.makedirs(outdir, exist_ok=True)

    # Trainer for test
    devs = config.devices
    num_devs = len(devs) if isinstance(devs, (list, tuple)) else int(devs)
    strategy = DDPStrategy(find_unused_parameters=True) if num_devs > 1 else "auto"
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
