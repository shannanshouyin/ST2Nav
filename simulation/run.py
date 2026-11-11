#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import torch
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.config.default import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="optional experiment name override; otherwise use EXP_NAME from yaml",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "dagger", "eval", "inference"],
        required=True,
        help="run type; train and dagger are equivalent",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None, local_rank=None,
            exp_name=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train", "eval", or "inference".
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    config = get_config(exp_config, opts)
    config.defrost()

    exp_name = str(exp_name if exp_name is not None else config.EXP_NAME).strip()
    if not exp_name:
        raise ValueError("EXP_NAME must be set in the experiment YAML")

    is_train = run_type in ("train", "dagger")

    config.TENSORBOARD_DIR = os.path.join(config.TENSORBOARD_DIR, exp_name)
    config.CHECKPOINT_FOLDER = os.path.join(config.CHECKPOINT_FOLDER, exp_name)
    if is_train:
        config.TENSORBOARD_DIR = config.CHECKPOINT_FOLDER
    config.RESULTS_DIR = os.path.join(config.RESULTS_DIR, exp_name, "eval_results")
    config.VIDEO_DIR = os.path.join(config.VIDEO_DIR, exp_name)
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    config.local_rank = local_rank
    os.makedirs("data/logs/running_log", exist_ok=True)
    os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
    if is_train:
        logger.add_filehandler(os.path.join(config.CHECKPOINT_FOLDER, config.LOG_FILE))
    elif run_type == "eval":
        config.EVAL.METRIC_LOG_FILE = os.path.join(
            "data/logs/running_log", config.LOG_FILE
        )
    else:
        logger.add_filehandler(os.path.join("data/logs/running_log", config.LOG_FILE))
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_name = config.TRAINER_NAME
    if is_train and config.GRPO.enabled:
        from vlnce_baselines import GRPO_trainer_ETP_R1  # noqa: F401
        trainer_name = 'GRPO-R1'
    trainer_init = baseline_registry.get_trainer(trainer_name)
    print("trainer_init\n", trainer_init) 
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    
    if is_train:
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()

if __name__ == "__main__":
    main()
