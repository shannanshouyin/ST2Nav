from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat.config.default import Config as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.EXP_NAME = "etp_r1_stgasa_mtr"
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.VIDEO_RESOLUTION = CN()
_C.VIDEO_RESOLUTION.enabled = False
_C.VIDEO_RESOLUTION.width = 0
_C.VIDEO_RESOLUTION.height = 0
_C.VIDEO_RESOLUTION.scale = 1.0
# 原始top-down map栅格分辨率；提高它会让右侧拓扑/地图面板更细，不改变RGB传感器输入。
_C.VIDEO_TOPO_MAP_RESOLUTION = 512
# eval视频中是否在top-down map持续叠加教师参考路径。
_C.VIDEO_DRAW_TEACHER_PATH = False
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.ZERO_SHOT_PRETRAIN = False
_C.EVAL.METRIC_LOG_FILE = ""
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.LANGUAGES = ["en-US", "en-IN"]
_C.INFERENCE.SAMPLE = False
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"
_C.INFERENCE.FORMAT = "rxr"  # either 'rxr' or 'r2r'
# -----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# -----------------------------------------------------------------------------
_C.GRPO = CN()
_C.GRPO.enabled = False
_C.GRPO.use_mtr = True
_C.GRPO.freeze_mtr = False
_C.GRPO.mtr_mask_ratio = 0.15
_C.GRPO.mtr_loss_weight = 0.1
_C.GRPO.mtr_birth_loss_weight = 1.0
_C.GRPO.mtr_observation_loss_weight = 1.0
_C.GRPO.mtr_distance_loss_weight = 0.1  # Independent of mtr_loss_weight.
_C.GRPO.mtr_edge_loss_weight = 0.1
_C.GRPO.lr = 1e-5
_C.GRPO.iters = 5000
_C.GRPO.log_every = 100
_C.GRPO.checkpoint_every = 500
_C.GRPO.batch_size = 1
_C.GRPO.load_from_ckpt = True
_C.GRPO.ckpt_to_load = ''
_C.GRPO.is_requeue = False
_C.GRPO.sample_num = 4
_C.GRPO.update_epochs = 1
_C.GRPO.normalize_by_trajectory = False  # Policy/KL only; preserve legacy reduction by default.
_C.GRPO.grpo_beta = 0.01
_C.GRPO.grpo_epsilon = 0.2
_C.GRPO.max_grad_norm = 1.0
_C.GRPO.enable_amp = True
_C.GRPO.enable_all_dropouts = False
_C.GRPO.dropout_in_sampling = False
_C.GRPO.dropout_rate = 0.0
_C.GRPO.warmup_iters = 100
_C.GRPO.min_lr_ratio = 0.1
_C.GRPO.max_traj_len = 15
_C.GRPO.max_text_len = 150
_C.GRPO.loc_noise = 0.5
_C.GRPO.ghost_aug = 0.0
_C.GRPO.waypoint_aug = True
_C.GRPO.back_algo = 'control'
_C.GRPO.tryout = True
_C.GRPO.reward_success_distance = 1.5
_C.GRPO.reward_distance_scale = 6.0
_C.GRPO.use_metric_aligned_reward = False  # R2R only: keep success threshold; cap distance bonus inside it.

_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.batch_size = 5
_C.IL.epochs = 4
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 3.2
# load an already trained model for fine tuning
_C.IL.waypoint_aug = False
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
_C.IL.st_gasa_lr = -1.0
_C.IL.sap_lr = -1.0
_C.IL.mtr_lr = -1.0
_C.IL.use_online_mtr = False
_C.IL.mtr_mask_ratio = 0.15
_C.IL.mtr_loss_weight = 0.1
_C.IL.mtr_birth_loss_weight = 1.0
_C.IL.mtr_observation_loss_weight = 1.0
_C.IL.mtr_distance_loss_weight = 0.1  # Independent of mtr_loss_weight.
_C.IL.mtr_edge_loss_weight = 0.1
_C.IL.expert_stop_distance = 1.5
_C.IL.RP_DAGGER = CN()
_C.IL.RP_DAGGER.enabled = False
_C.IL.RP_DAGGER.lambda_reg = 0.1
_C.IL.RP_DAGGER.tau = 3.0  # Geodesic distance scale in meters, not GASA distance / 30.
_C.IL.sample_ratio = 0.75
_C.IL.decay_interval = 2000
_C.IL.warmup_iters = 500
_C.IL.min_lr_ratio = 0.1
_C.IL.checkpoint_every = 5000
_C.IL.TRAINABLE = CN()
_C.IL.TRAINABLE.language_embeddings = False
_C.IL.TRAINABLE.language_encoder = False
_C.IL.TRAINABLE.panorama_embeddings = False
_C.IL.TRAINABLE.global_map_embeddings = False
_C.IL.TRAINABLE.global_encoder_x_layers = True
_C.IL.TRAINABLE.graph_query_text = True
_C.IL.TRAINABLE.dpft_temporal = True
_C.IL.TRAINABLE.graph_attentioned_txt = True
_C.IL.TRAINABLE.global_sap_head = True
_C.IL.TRAINABLE.stgasa = True
_C.IL.TRAINABLE.mtr = False
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False
# it True, start training from the saved epoch
# -----------------------------------------------------------------------------
# IL: RXR TRAINER CONFIG
# -----------------------------------------------------------------------------
_C.IL.RECOLLECT_TRAINER = CN()
_C.IL.RECOLLECT_TRAINER.preload_trajectories_file = True
_C.IL.RECOLLECT_TRAINER.trajectories_file = (
    "data/trajectories_dirs/debug/trajectories.json.gz"
)
# if set to a positive int, episodes with longer paths are ignored in training
_C.IL.RECOLLECT_TRAINER.max_traj_len = -1
# if set to a positive int, effective_batch_size must be some multiple of
# IL.batch_size. Gradient accumulation enables an arbitrarily high "effective"
# batch size.
_C.IL.RECOLLECT_TRAINER.effective_batch_size = -1
_C.IL.RECOLLECT_TRAINER.preload_size = 30
_C.IL.RECOLLECT_TRAINER.use_iw = True
_C.IL.RECOLLECT_TRAINER.gt_file = (
    "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
)
# -----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# -----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.update_size = 5000
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.load_space = False
# if True, load saved observation space and action space
_C.IL.DAGGER.lmdb_map_size = 1.0e12
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens/
_C.IL.DAGGER.lmdb_commit_frequency = 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = (
    "data/trajectories_dirs/debug/trajectories.lmdb"
)
# -----------------------------------------------------------------------------
# RL CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = [
    "CenterCropperPerSensor",
]
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = [
    ("rgb", (224, 298)),
    ("depth", (256, 341)),
]
# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"  # or "Seq2SeqPolicy"
_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False
_C.MODEL.stgasa = False
_C.MODEL.dpft_temporal = False
_C.MODEL.STGASA = CN()
_C.MODEL.STGASA.relation_hidden_size = 128
_C.MODEL.STGASA.time_max_bucket = 6
_C.MODEL.STGASA.eval_gamma = None
_C.MODEL.mtr = False
_C.MODEL.MTR = CN()
_C.MODEL.MTR.relation_hidden_size = 128
_C.MODEL.MTR.spatial_recovery = False  # Keep old temporal-only checkpoints/callers unchanged.
_C.MODEL.MTR.eval_predict = False
_C.MODEL.MTR.eval_mask_ratio = 0.15

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = False

_C.MODEL.spatial_output = True
_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.output_size = 256

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
