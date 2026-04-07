import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from vlex_ria.core.utils import get_logger, set_log_level
from vlex_ria.core.config import VLexRIAConfig, load_config
from vlex_ria.data import create_dataloaders
from vlex_ria.data.dataset import get_tokenizer
from vlex_ria.model import VLexRIAModel, print_model_summary
from vlex_ria.training import (
    CompositeReward,
    DPOTrainer,
    GRPOTrainer,
    LengthReward,
    PPOTrainer,
    RuleBasedReward,
    create_trainer,
)

logger = get_logger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_config_path(config_path: Optional[str]) -> Optional[str]:
    if config_path is None:
        return None

    candidate = Path(config_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    root = _project_root()
    root_candidate = root / candidate
    if root_candidate.exists():
        return str(root_candidate.resolve())

    configs_candidate = root / "configs" / candidate.name
    if configs_candidate.exists():
        return str(configs_candidate.resolve())

    return str(candidate)


def parse_args(default_mode: str = "pretrain", include_algorithm: bool = False):
    parser = argparse.ArgumentParser(description="VLexRIA Training")
    parser.add_argument(
        "--mode",
        type=str,
        default=default_mode,
        choices=["pretrain", "sft", "rl"],
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    if include_algorithm:
        parser.add_argument(
            "--algorithm",
            type=str,
            choices=["dpo", "grpo", "ppo", "all"],
            default=None,
        )
    return parser.parse_args()


def _apply_common_overrides(config: VLexRIAConfig, args, mode: str) -> None:
    mode_cfg = {
        "pretrain": config.pretraining,
        "sft": config.sft,
        "rl": config.rl,
    }[mode]

    if args.batch_size is not None:
        mode_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        mode_cfg.learning_rate = args.learning_rate
    if args.max_steps is not None:
        mode_cfg.max_steps = args.max_steps
    if args.device is not None:
        config.pretraining.device = args.device
        config.sft.device = args.device
        config.rl.device = args.device

    if args.test:
        config.pretraining.max_steps = min(config.pretraining.max_steps, 50)
        config.pretraining.eval_steps = min(config.pretraining.eval_steps, 20)
        config.pretraining.save_steps = min(config.pretraining.save_steps, 25)
        config.pretraining.logging_steps = min(config.pretraining.logging_steps, 5)
        config.sft.max_steps = min(config.sft.max_steps, 30)
        config.sft.eval_steps = min(config.sft.eval_steps, 15)
        config.rl.max_steps = min(config.rl.max_steps, 20)
        config.data.sft_max_samples = min(config.data.sft_max_samples, 100)
        config.data.rl_max_samples = min(config.data.rl_max_samples, 50)

    if mode == "pretrain":
        current_dataset = config.data.pretrain_dataset_name
        logger.info(f"[Dataset] Using dataset: {current_dataset}")

    if mode == "rl" and hasattr(args, "algorithm") and args.algorithm and args.algorithm != "all":
        config.rl.algorithm = args.algorithm


def _load_checkpoint_if_available(model: VLexRIAModel, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint loaded successfully")


def _create_rl_trainer(
    model: VLexRIAModel,
    ref_model: Optional[VLexRIAModel],
    config: VLexRIAConfig,
    train_loader,
    val_loader,
    tokenizer,
):
    reward_fn = CompositeReward(
        [
            (RuleBasedReward(), 0.7),
            (LengthReward(target_length=50), 0.3),
        ]
    )
    algorithm = config.rl.algorithm
    if algorithm == "dpo":
        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config.rl,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            beta=config.rl.dpo_beta,
            label_smoothing=config.rl.dpo_label_smoothing,
        )
    if algorithm == "ppo":
        return PPOTrainer(
            model=model,
            ref_model=ref_model,
            config=config.rl,
            vis_config=config.visualization,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            value_coef=config.rl.value_coef,
            entropy_coef=config.rl.entropy_coef,
        )
    return GRPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config.rl,
        vis_config=config.visualization,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
    )


def _effective_max_samples(config: VLexRIAConfig, mode: str, args) -> Optional[int]:
    if args.max_samples is not None:
        return args.max_samples
    if mode == "pretrain":
        return config.data.pretrain_max_samples
    if mode == "sft":
        return config.data.sft_max_samples
    return config.data.rl_max_samples


def run_training(mode: str, args) -> Dict[str, Any]:
    if args.test:
        set_log_level(logging.DEBUG)

    resolved_config = resolve_config_path(args.config)
    config = load_config(resolved_config)
    _apply_common_overrides(config, args, mode)
    config.print_config()

    tokenizer = get_tokenizer(config.data)
    config.model.vocab_size = len(tokenizer)

    model = VLexRIAModel(config.model)
    print_model_summary(model, config.model)
    _load_checkpoint_if_available(model, args.checkpoint)

    ref_model = None
    if mode == "rl" and config.rl.use_reference_model:
        ref_model = VLexRIAModel(config.model)
        ref_model.load_state_dict(model.state_dict())

    mode_cfg = {
        "pretrain": config.pretraining,
        "sft": config.sft,
        "rl": config.rl,
    }[mode]

    train_loader, val_loader = create_dataloaders(
        config=config.data,
        tokenizer=tokenizer,
        mode=mode,
        batch_size=mode_cfg.batch_size,
        max_samples=_effective_max_samples(config, mode, args),
    )

    if mode == "rl":
        trainer = _create_rl_trainer(
            model=model,
            ref_model=ref_model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
        )
    else:
        trainer = create_trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            mode=mode,
            ref_model=ref_model,
        )

    metrics = trainer.train()
    logger.info("=" * 70)
    logger.info("Training Complete")
    logger.info("=" * 70)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    return metrics


def main(default_mode: str = "pretrain", include_algorithm: bool = False):
    args = parse_args(default_mode=default_mode, include_algorithm=include_algorithm)
    if include_algorithm and getattr(args, "algorithm", None) == "all":
        outcomes = {}
        for algorithm in ["dpo", "grpo", "ppo"]:
            args.algorithm = algorithm
            outcomes[algorithm] = run_training("rl", args)
        return outcomes
    return run_training(args.mode, args)

if __name__ == "__main__":
    main()
