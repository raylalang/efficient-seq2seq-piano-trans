# evaluate.py
from train import MT3Trainer
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
import sys
from typing import Tuple


def _resolve_state_dict(ckpt_obj: dict) -> dict:
    """
    Accept either:
      - raw state dict (param_name -> tensor)
      - checkpoint dict containing {"state_dict": ...}
    """
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        return ckpt_obj["state_dict"]
    return ckpt_obj


def _load_weights(model: MT3Trainer, cfg: DictConfig) -> Tuple[list, list]:
    """
    Load weights into model.model using cfg.model.checkpoint_path,
    honoring cfg.model.checkpoint_ignore_layers and cfg.model.strict_checkpoint.
    Returns (missing_keys, unexpected_keys) for visibility.
    """
    ckpt_path = cfg.model.checkpoint_path
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _resolve_state_dict(ckpt)

    ignore = getattr(cfg.model, "checkpoint_ignore_layers", None)
    if ignore:
        for k in ignore:
            if k in state_dict:
                del state_dict[k]

    strict = bool(getattr(cfg.model, "strict_checkpoint", True))
    missing, unexpected = model.model.load_state_dict(state_dict, strict=strict)
    return missing, unexpected


def _make_test_outdir(cfg: DictConfig, ckpt_path: str) -> str:
    """
    Decide where to write test artifacts (CSVs, MIDIs, JSON).
    Prefer an explicit test_output_dir if already present on the model;
    otherwise, use training.log_dir or ./runs_eval/<ckpt_base>.
    """
    base = os.path.splitext(os.path.basename(ckpt_path))[0]
    # If user provided a log_dir in config, nest under it for clarity
    log_dir = cfg.get("training", {}).get("log_dir", None)
    if log_dir:
        outdir = os.path.join(log_dir, "test_eval", base)
    else:
        outdir = os.path.join("./runs_eval", base)
    os.makedirs(outdir, exist_ok=True)
    return outdir


@hydra.main(config_path="config", config_name="main_config", version_base=None)
def main(config: OmegaConf):
    cfg: DictConfig = config  # type: ignore

    # ---- Force test mode & disable any train-time online testing path ----
    cfg.training.mode = "test"
    if hasattr(cfg.training, "online_testing"):
        cfg.training.online_testing = False

    # Optional: reproducibility if seed exists
    if hasattr(cfg.training, "seed") and cfg.training.seed is not None:
        pl.seed_everything(int(cfg.training.seed), workers=True)

    # ---- Build module ----
    model = MT3Trainer(cfg)

    # ---- Load weights (robust to raw or wrapped checkpoint) ----
    missing, unexpected = _load_weights(model, cfg)
    if missing:
        print(
            f"[evaluate] Missing keys (showing up to 20): {missing[:20]} ... total={len(missing)}",
            file=sys.stderr,
        )
    if unexpected:
        print(
            f"[evaluate] Unexpected keys (up to 20): {unexpected[:20]} ... total={len(unexpected)}",
            file=sys.stderr,
        )

    # ---- Decide output dir for test artifacts ----
    if not getattr(model, "test_output_dir", None):
        model.test_output_dir = _make_test_outdir(cfg, cfg.model.checkpoint_path)

    # ---- Build Trainer for test ----
    # Use leaner DDP unless you truly need find_unused_parameters=True
    if int(cfg.devices) > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"

    # Precision: honor config if provided; default to 16-mixed for speed
    precision = getattr(cfg, "precision", None) or "16-mixed"

    trainer = pl.Trainer(
        logger=[],  # no TB/W&B for test-only
        devices=cfg.devices,
        accelerator=cfg.accelerator,  # "gpu" or "cpu"
        strategy=strategy,
        precision=precision,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        log_every_n_steps=0,
        enable_progress_bar=True,
    )

    # ---- Run Lightning test loop ----
    trainer.test(model)


if __name__ == "__main__":
    main()
