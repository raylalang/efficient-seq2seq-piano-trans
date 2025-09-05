from train import MT3Trainer
import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
from pathlib import Path


@hydra.main(config_path="config", config_name="main_config", version_base=None)
def my_main(config: OmegaConf):
    # --- Auto-load & merge the run's experiment_config.yaml before building the module ---
    # Keep CLI/Hydra overrides in `config` taking precedence over the loaded run config.
    assert (
        config.model.checkpoint_path is not None
    ), "Set +model.checkpoint_path=...</cpt/steps_xxx.ckpt>"
    ckpt_path = Path(str(config.model.checkpoint_path))
    # common layouts:
    #   <run_dir>/cpt/steps_xxx.ckpt -> <run_dir>/experiment_config.yaml
    #   <run_dir>/latest.ckpt        -> <run_dir>/experiment_config.yaml
    candidates = [
        ckpt_path.parent.parent
        / "experiment_config.yaml",  # runs/.../experiment_config.yaml
        ckpt_path.parent
        / "experiment_config.yaml",  # runs/.../cpt/experiment_config.yaml (fallback)
        ckpt_path / "experiment_config.yaml",  # if a directory was passed
    ]
    run_cfg_path = next((p for p in candidates if p.is_file()), None)
    if run_cfg_path is not None:
        run_cfg = OmegaConf.load(str(run_cfg_path))
        # merge so that values in `config` (CLI overrides) win
        config = OmegaConf.merge(run_cfg, config)
    # Force test mode and ensure we evaluate the exact ckpt path the user supplied
    config.training.mode = "test"
    config.model.checkpoint_path = str(ckpt_path)

    # Build module
    model = MT3Trainer(config)

    # Load checkpoint (supports raw state_dict or Lightning ckpt)
    ckpt_obj = torch.load(config.model.checkpoint_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        sd = {
            k.replace("model.", "", 1): v
            for k, v in ckpt_obj["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        sd = ckpt_obj
    ignore = set(getattr(config.model, "checkpoint_ignore_layers", []) or [])
    for k in list(sd.keys()):
        if k in ignore:
            print(f"Removing key {k} from state_dict")
            del sd[k]
    strict = bool(getattr(config.model, "strict_checkpoint", True))
    model.model.load_state_dict(sd, strict=strict)

    # Where to write test artifacts (CSVs, MIDIs, JSON)
    if not getattr(model, "test_output_dir", None):
        # Prefer the run dir if we found a run config; otherwise fall back to a local eval dir
        if run_cfg_path is not None:
            run_dir = str(run_cfg_path.parent)
            outdir = os.path.join(run_dir, "eval")
        else:
            base = os.path.splitext(os.path.basename(config.model.checkpoint_path))[0]
            outdir = f"./runs_eval/{base}"
        model.test_output_dir = outdir
    os.makedirs(model.test_output_dir, exist_ok=True)

    # Trainer for test
    # - precision to speed up GPU inference
    # - no sanity val
    # - DDP only if devices>1
    devs = config.devices
    num_devs = len(devs) if isinstance(devs, (list, tuple)) else int(devs)
    strategy = DDPStrategy(find_unused_parameters=True) if num_devs > 1 else "auto"
    trainer = pl.Trainer(
        logger=[],  # no TB/WB for test
        devices=config.devices,
        accelerator=config.accelerator,  # "gpu"
        strategy=strategy,
        precision="16-mixed",
        num_sanity_val_steps=0,
    )

    trainer.test(model)


if __name__ == "__main__":
    my_main()
