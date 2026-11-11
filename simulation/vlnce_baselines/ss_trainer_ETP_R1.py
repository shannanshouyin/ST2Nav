import os
import sys
import random
from collections import defaultdict
from typing import Dict
import jsonlines

import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.common.rp_dagger import rp_dagger_options, relative_distance_regularizer
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST, resolve_graph_step_id_mode
from vlnce_baselines.models.etp.stgasa import sample_symmetric_pair_mask
from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import gather_list_and_concat
from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict

@baseline_registry.register_trainer(name="SS-ETP-R1")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.illegal_episodes_count = 0
        self.graph_step_id_mode = 'arrival'

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        if self.config.ONLY_LAST_SAVEALL and (not iteration == self.config.IL.iters):
            torch.save(
                obj={
                            "state_dict": self.policy.state_dict(),
                            "config": self.config,
                            "graph_step_id_mode": self.graph_step_id_mode,
                            "iteration": iteration
                        },
                        f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
                    )
        else:
            torch.save(
                obj={
                    "state_dict": self.policy.state_dict(),
                    "config": self.config,
                    "graph_step_id_mode": self.graph_step_id_mode,
                    "optim_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "iteration": iteration,
                },
                f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
            )
        self._log_stgasa_summary(iteration)

    def _restore_graph_step_id_mode(self, checkpoint=None):
        # Reset on every policy load, including sequential checkpoint evaluation.
        self.graph_step_id_mode = resolve_graph_step_id_mode(checkpoint or {})
        if checkpoint is not None and 'graph_step_id_mode' not in checkpoint:
            logger.warning(
                '[GraphInput] Unversioned checkpoint: inferred step_ids=%s from raw '
                'parameter keys (independent of ST-GASA/MTR switches).',
                self.graph_step_id_mode,
            )
        else:
            logger.info('[GraphInput] step_ids=%s', self.graph_step_id_mode)

    def _filter_mismatched_state_dict(self, state_dict):
        model_state = self.policy.state_dict()
        filtered_state = dict(state_dict)
        removed = []
        old_stgasa_fragments = (
            "st_dist_embed", "st_mlp", "st_rel_mlp", "st_phi",
            "st_psi", "st_lambda_v", "st_time_bucket",
            "st_time_embedding", "st_time_key_proj", "st_dir_embedding",
            "st_dir_bias_proj", "st_value_proj", "st_gate_t", "st_gate_d", "st_gate_v",
            "instruction_gate", "age_embedding", "spatial_mask_bias",
            "st_planner", "st_progress_head", "st_progress_residual", "st_progress_beta",
            "mstr_head.age_head", "mstr_head.spatial_head",
            "mtr_head.context.", "mtr_head.distance_head.",
        )
        for key in list(filtered_state.keys()):
            if any(fragment in key for fragment in old_stgasa_fragments):
                reason = ('replaced_spatial_recovery' if 'mtr_head.' in key else 'old_stgasa_v1')
                removed.append((key, reason, "reinitialized"))
                filtered_state.pop(key)

        def candidate_model_keys(key):
            keys = [key]
            if key.startswith("module."):
                keys.append(key[7:])
            if key.startswith("net."):
                keys.append("net.module." + key[4:])
            if key.startswith("module.net."):
                keys.append("net." + key[len("module.net."):])
                keys.append("net.module." + key[len("module.net."):])
            return keys

        for key in list(filtered_state.keys()):
            matched_key = None
            for candidate in candidate_model_keys(key):
                if candidate in model_state:
                    matched_key = candidate
                    break
            if matched_key is None:
                continue
            if tuple(filtered_state[key].shape) != tuple(model_state[matched_key].shape):
                removed.append((key, tuple(filtered_state[key].shape), tuple(model_state[matched_key].shape)))
                filtered_state.pop(key)
        if removed and self.local_rank < 1:
            logger.warning(
                "Checkpoint shape mismatch; reinitializing incompatible keys: %s",
                removed[:12],
            )
        self._ckpt_shape_mismatch_count = len(removed)
        return filtered_state

    def _reset_missing_st_gasa_parameters_from_ckpt(self, state_dict):
        net = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
        vb = getattr(net, "vln_bert", None)
        if vb is None or not hasattr(vb, "reset_missing_st_gasa_parameters"):
            return
        prefixes = ("net.", "module.net.", "module.")
        normalized_keys = []
        for key in state_dict.keys():
            k = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                        changed = True
            normalized_keys.append(k)
        vb.reset_missing_st_gasa_parameters(normalized_keys)

    def _apply_video_topdown_map_resolution(self):
        resolution = int(getattr(self.config, "VIDEO_TOPO_MAP_RESOLUTION", 512))
        if resolution <= 0:
            return
        self.config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.MAP_RESOLUTION = resolution
        self.config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.DRAW_REFERENCE_PATH = bool(
            getattr(self.config, "VIDEO_DRAW_TEACHER_PATH", False)
        )
        if self.local_rank < 1:
            logger.info("VIDEO top-down map resolution: %d", resolution)

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        print(f"init camera information: resize_config:{resize_config}, crop_config:{crop_config}, new_camera_heading:{camera_orientations}")
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self._apply_video_topdown_map_resolution()
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    @staticmethod
    def _is_st_gasa_param(name):
        return ".stgasa_relation." in name

    @staticmethod
    def _is_sap_param(name):
        return "global_sap_head" in name

    @staticmethod
    def _is_mtr_param(name):
        return "vln_bert.mtr_head" in name

    def _trainable_cfg(self):
        return getattr(self.config.IL, "TRAINABLE", None)

    def _trainable_flag(self, key, default=False):
        cfg = self._trainable_cfg()
        if cfg is None:
            return default
        return bool(getattr(cfg, key, default))

    def _policy_param_group(self, name):
        if self._is_mtr_param(name):
            return "mtr"
        if self._is_st_gasa_param(name):
            return "stgasa"
        if self._is_sap_param(name):
            return "global_sap_head"
        if "vln_bert.dpft_temporal." in name:
            return "dpft_temporal"
        if "vln_bert.graph_query_text" in name:
            return "graph_query_text"
        if "vln_bert.graph_attentioned_txt_embeds_transform" in name:
            return "graph_attentioned_txt"
        if "vln_bert.embeddings" in name:
            return "language_embeddings"
        if "vln_bert.lang_encoder" in name:
            return "language_encoder"
        if "vln_bert.img_embeddings" in name:
            return "panorama_embeddings"
        if "vln_bert.global_encoder.encoder.x_layers" in name:
            return "global_encoder_x_layers"
        if "vln_bert.global_encoder" in name:
            return "global_map_embeddings"
        if "rgb_encoder" in name or "depth_encoder" in name:
            return "frozen_backbone"
        return "frozen_other"

    def _mtr_grad_norm(self):
        squared = 0.0
        for name, param in self.policy.named_parameters():
            if self._is_mtr_param(name) and param.grad is not None:
                squared += float(param.grad.detach().float().pow(2).sum().item())
        return math.sqrt(squared)

    def _apply_sft_trainable_modules(self, config, log_summary=True):
        trainable_cfg = self._trainable_cfg()
        if trainable_cfg is None:
            raise ValueError("IL.TRAINABLE is required; set explicit module train/freeze flags.")

        if (
            self._trainable_flag("language_embeddings", False)
            or self._trainable_flag("language_encoder", False)
        ) and bool(config.MODEL.fix_lang_embedding):
            raise ValueError(
                "IL.TRAINABLE enables language training, but MODEL.fix_lang_embedding=True. "
                "Set MODEL.fix_lang_embedding=False."
            )
        if self._trainable_flag("panorama_embeddings", False) and bool(config.MODEL.fix_pano_embedding):
            raise ValueError(
                "IL.TRAINABLE.panorama_embeddings=True conflicts with "
                "MODEL.fix_pano_embedding=True; set MODEL.fix_pano_embedding=False."
            )
        if bool(getattr(config.IL, 'use_online_mtr', False)):
            if not bool(getattr(config.MODEL, 'mtr', False)):
                raise ValueError('IL.use_online_mtr=True requires MODEL.mtr=True')
            if not self._trainable_flag('mtr', False):
                raise ValueError('IL.use_online_mtr=True requires IL.TRAINABLE.mtr=True')

        use_online_mtr = bool(getattr(config.IL, 'use_online_mtr', False))
        for name, param in self.policy.named_parameters():
            group = self._policy_param_group(name)
            enabled = self._trainable_flag(group, False)
            # MODEL.mtr controls whether the pretrained head exists, while this
            # switch alone controls whether online fine-tuning executes it.
            if group == "mtr" and not use_online_mtr:
                enabled = False
            param.requires_grad_(enabled)

        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        if self.local_rank < 1 and log_summary:
            if use_online_mtr and not bool(getattr(config.MODEL, 'stgasa', False)):
                logger.info(
                    'Online MTR: ST-GASA disabled; birth/observation predictions '
                    'are auxiliary supervision only, not SAP attention inputs.'
                )
            trainable_tensors = [n for n, p in self.policy.named_parameters() if p.requires_grad]
            trainable_params = sum(p.numel() for _, p in self.policy.named_parameters() if p.requires_grad)
            group_counts = defaultdict(int)
            group_params = defaultdict(int)
            for name, param in self.policy.named_parameters():
                if not param.requires_grad:
                    continue
                group = self._policy_param_group(name)
                group_counts[group] += 1
                group_params[group] += param.numel()
            logger.info(
                "SFT trainable modules: tensors=%d params=%d flags=%s",
                len(trainable_tensors),
                trainable_params,
                {k: bool(v) for k, v in trainable_cfg.items()},
            )
            for group in sorted(group_counts.keys()):
                logger.info(
                    "  trainable_group: %s tensors=%d params=%d",
                    group, group_counts[group], group_params[group],
                )

    def _build_sft_optimizer(self, config, log_summary=True):
        base_lr = float(config.IL.lr)
        st_lr = float(getattr(config.IL, "st_gasa_lr", -1.0))
        sap_lr = float(getattr(config.IL, "sap_lr", -1.0))
        mtr_lr = float(getattr(config.IL, "mtr_lr", -1.0))
        if st_lr <= 0:
            st_lr = base_lr
        if sap_lr <= 0:
            sap_lr = base_lr
        if mtr_lr <= 0:
            mtr_lr = base_lr
        buckets = {"base": [], "stgasa": [], "sap": [], "mtr": []}
        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_mtr_param(name):
                buckets["mtr"].append((name, param))
            elif self._is_st_gasa_param(name):
                buckets["stgasa"].append((name, param))
            elif self._is_sap_param(name):
                buckets["sap"].append((name, param))
            else:
                buckets["base"].append((name, param))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []

        def add_bucket(pairs, lr):
            if not pairs:
                return
            decay_params = [p for n, p in pairs if not any(nd in n for nd in no_decay)]
            nodecay_params = [p for n, p in pairs if any(nd in n for nd in no_decay)]
            if decay_params:
                optimizer_grouped_parameters.append({
                    'params': decay_params,
                    'weight_decay': 0.01,
                    'lr': lr,
                })
            if nodecay_params:
                optimizer_grouped_parameters.append({
                    'params': nodecay_params,
                    'weight_decay': 0.0,
                    'lr': lr,
                })

        add_bucket(buckets["base"], base_lr)
        add_bucket(buckets["stgasa"], st_lr)
        add_bucket(buckets["sap"], sap_lr)
        add_bucket(buckets["mtr"], mtr_lr)
        if not optimizer_grouped_parameters:
            raise ValueError("No trainable parameters for SFT optimizer; check IL.TRAINABLE module flags.")

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=base_lr)
        if self.local_rank < 1 and log_summary:
            logger.info(
                "SFT optimizer: base_tensors=%d stgasa_tensors=%d sap_tensors=%d mtr_tensors=%d "
                "lr=%.6g st_gasa_lr=%.6g sap_lr=%.6g mtr_lr=%.6g"
                % (
                    len(buckets["base"]),
                    len(buckets["stgasa"]),
                    len(buckets["sap"]),
                    len(buckets["mtr"]),
                    base_lr,
                    st_lr,
                    sap_lr,
                    mtr_lr,
                )
            )

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        log_training_setup: bool = True,
    ):
        start_iter = 0
        self._restore_graph_step_id_mode()
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        logger.info(f"-------------------Load pretrain weight: {config.MODEL.pretrained_path}-------------------")
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict']) 
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)

        self._apply_sft_trainable_modules(config, log_summary=log_training_setup)
        self._build_sft_optimizer(config, log_summary=log_training_setup)
        num_warmup_steps = self.config.IL.warmup_iters
        num_training_steps = self.config.IL.iters
        min_lr_ratio = self.config.IL.min_lr_ratio

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            return decayed_lr_multiplier
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # self.scheduler.step()

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                search_pattern = os.path.join(config.CHECKPOINT_FOLDER, "*.pth")
                ckpt_list = glob.glob(search_pattern)
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self._restore_graph_step_id_mode(ckpt_dict)
            if config.IL.is_requeue:
                start_iter = ckpt_dict["iteration"]
            else:
                start_iter = 0
            filtered_state = self._filter_mismatched_state_dict(ckpt_dict["state_dict"])

            if 'module' in list(filtered_state.keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                incompatible_keys = self.policy.load_state_dict(filtered_state, strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            elif 'module' not in list(filtered_state.keys())[0] and self.config.GPU_NUMBERS > 1:
                new_state_dict = OrderedDict()
                for k, v in filtered_state.items():
                    if k.startswith("net."):
                        name = k.replace("net.", "net.module.", 1)
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                incompatible_keys = self.policy.load_state_dict(new_state_dict, strict=False)
            else:
                incompatible_keys = self.policy.load_state_dict(filtered_state, strict=False)
            self._reset_missing_st_gasa_parameters_from_ckpt(filtered_state)
            
            if self.local_rank < 1:
                print("\n" + "="*25 + " Weight loading mismatch report " + "="*25)
                if incompatible_keys.missing_keys:
                    print("The following network layers exist in the model but are missing in the weight file (initial values will be used):")
                    for key in sorted(incompatible_keys.missing_keys):
                        print(f"  - {key}")
                else:
                    print("All network layers present in the model were found in the weight file.")
                if incompatible_keys.unexpected_keys:
                    print("\nThe following network layers exist in the weight file but are missing in the model (will be ignored):")
                    for key in sorted(incompatible_keys.unexpected_keys):
                        print(f"  - {key}")
                else:
                    print("\nThere are no extra network layers in the weight file.")
                print("="*75 + "\n")

            can_restore_optim = (
                config.IL.is_requeue
                and getattr(self, "_ckpt_shape_mismatch_count", 0) == 0
                and "optim_state" in ckpt_dict
                and "scheduler_state" in ckpt_dict
            )
            if can_restore_optim:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                self.scheduler.load_state_dict(ckpt_dict["scheduler_state"])
            elif config.IL.is_requeue and self.local_rank < 1:
                logger.warning(
                    "Skip optimizer/scheduler restore because checkpoint has shape-mismatched model keys "
                    "or does not contain optimizer/scheduler state."
                )
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _expert_stop_distance(self):
        return float(getattr(self.config.IL, "expert_stop_distance", 1.5))

    def _teacher_action_new(
        self,
        batch_gmap_vp_ids,
        batch_no_vp_left,
        is_train,
        return_costs=False,
    ):
        teacher_actions = []
        batch_costs = []
        cur_episodes = self.envs.current_episodes()
        stop_distance = self._expert_stop_distance()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            costs = np.full(len(gmap_vp_ids), np.nan, dtype=np.float32) if return_costs else None
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal", {"is_train": is_train})
            if curr_dis_to_goal < stop_distance:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy in ('spl', 'ndtw'):
                    # Sample a merged ghost's representative once: the hard
                    # teacher and regularizer must use precisely the same pose.
                    ghost_vp_pos = [
                        (vp, random.choice(pos))
                        for vp, pos in gmap.ghost_real_pos.items()
                    ]
                    if self.config.IL.expert_policy == 'spl':
                        distances = [self.envs.call_at(
                            i, "point_dist_to_goal", {"pos": pos, "is_train": is_train}
                        ) for _, pos in ghost_vp_pos]
                        distances = np.asarray(distances, dtype=np.float64)
                        distances = np.where(np.isfinite(distances) & (distances >= 0), distances, np.inf)
                        target_ghost_vp = (ghost_vp_pos[int(np.argmin(distances))][0]
                                           if np.isfinite(distances).any() else None)
                    else:
                        # One RPC: advancing the RxR teacher twice could select
                        # different subgoals for CE and the distance labels.
                        result = self.envs.call_at(i, "ghost_dist_to_ref", {
                            "ghost_vp_pos": ghost_vp_pos,
                            "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                            "return_costs": return_costs,
                        })
                        if return_costs:
                            target_ghost_vp, distances = result
                        else:
                            target_ghost_vp = result
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp) if target_ghost_vp is not None else -100)
                    if return_costs:
                        indices = {vp: j for j, vp in enumerate(gmap_vp_ids)}
                        for (vp, _), distance in zip(ghost_vp_pos, distances):
                            costs[indices[vp]] = distance
                else:
                    raise NotImplementedError
            if return_costs:
                batch_costs.append(torch.from_numpy(costs))
        teacher_actions = torch.tensor(teacher_actions).cuda()
        if return_costs:
            return teacher_actions, pad_sequence(
                batch_costs, batch_first=True, padding_value=float('nan')
            ).to(teacher_actions.device)
        return teacher_actions

    def _sap_loss(self, nav_logits, teacher_actions, nav_inputs, cur_vp, teacher_distances=None):
        ce_loss = F.cross_entropy(
            nav_logits, teacher_actions, reduction='sum', ignore_index=-100
        )
        enabled, weight, tau = rp_dagger_options(self.config.IL)
        if not enabled:
            return ce_loss
        if teacher_distances is None:
            raise ValueError('RP-DAgger requires distances from the same teacher query as the CE labels')
        ghost_mask = nav_inputs['gmap_masks'].bool() & ~nav_inputs['gmap_visited_masks'].bool()
        reg_loss, valid_count = relative_distance_regularizer(
            nav_logits, teacher_actions, teacher_distances, ghost_mask, tau,
        )
        count = int(valid_count.item())
        self.logs['RP_reg_loss'].append(float(reg_loss.detach()) / max(count, 1))
        self.logs['RP_valid_states'].append(count)
        self.logs['RP_weighted_reg'].append(float(weight * reg_loss.detach()) / max(len(teacher_actions), 1))
        self.logs['RP_ce_loss'].append(float(ce_loss.detach()) / max(int((teacher_actions != -100).sum()), 1))
        return ce_loss + weight * reg_loss

    def _online_mtr_loss(self, nav_outs, active_envs):
        losses = nav_outs.get('mtr_losses')
        accuracies = nav_outs.get('mtr_accuracies')
        pair_mask = nav_outs.get('mtr_pair_mask')
        if losses is None or accuracies is None or pair_mask is None:
            return nav_outs['global_logits'].new_tensor(0.0)

        birth_weight = float(getattr(self.config.IL, 'mtr_birth_loss_weight', 1.0))
        observation_weight = float(getattr(self.config.IL, 'mtr_observation_loss_weight', 1.0))
        mtr_loss = birth_weight * losses['birth'] + observation_weight * losses['observation']
        self.logs['MTR_loss'].append(float(mtr_loss.detach().item()))
        self.logs['MTR_birth_loss'].append(float(losses['birth'].detach().item()))
        self.logs['MTR_birth_acc'].append(float(accuracies['birth'].detach().item()))
        self.logs['MTR_observation_loss'].append(float(losses['observation'].detach().item()))
        self.logs['MTR_observation_acc'].append(float(accuracies['observation'].detach().item()))
        self.logs['MTR_pairs'].append(float(pair_mask.sum().item() // 2))

        weight = float(getattr(self.config.IL, 'mtr_loss_weight', 0.1))
        auxiliary = mtr_loss * weight
        if 'mtr_edge_loss' in nav_outs:
            auxiliary = auxiliary + self.config.IL.mtr_edge_loss_weight * nav_outs['mtr_edge_loss']
            for name in ('edge_loss', 'edge_balanced_acc', 'edge_precision', 'edge_recall'):
                self.logs['MTR_' + name].append(float(nav_outs['mtr_' + name].detach()))
        if 'mtr_distance_loss' in nav_outs:
            distance_loss = nav_outs['mtr_distance_loss']
            auxiliary = auxiliary + self.config.IL.mtr_distance_loss_weight * distance_loss
            self.logs['MTR_distance_loss'].append(float(distance_loss.detach()))
            # Distances supplied to GASA are normalized by MAX_DIST.
            self.logs['MTR_distance_mae_m'].append(float(nav_outs['mtr_distance_mae']) * MAX_DIST)
        return auxiliary * int(active_envs)

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []
        
        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True

            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])

            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))

        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _iter_stgasa_attention_modules(self):
        net = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
        vb = getattr(net, "vln_bert", None)
        if vb is None:
            return
        for layer in vb.global_encoder.encoder.x_layers:
            relation = getattr(layer, "stgasa_relation", None)
            if relation is not None:
                yield relation

    def _log_stgasa_summary(self, iteration):
        if self.local_rank >= 1 or not hasattr(self, "policy"):
            return
        summaries = []
        for idx, relation in enumerate(self._iter_stgasa_attention_modules() or []):
            stats = relation.last_stats
            scale = torch.tanh(relation.gamma.detach()).cpu()
            if stats:
                temporal_abs = float(stats["bias_abs_mean"].cpu().item())
                spatial_abs = float(stats.get("spatial_abs_mean", torch.tensor(float('nan'))).cpu().item())
                finite = bool(stats.get("bias_finite", torch.tensor(True)).cpu().item()) and bool(
                    stats.get("spatial_finite", torch.tensor(True)).cpu().item()
                )
            else:
                temporal_abs = spatial_abs = float('nan')
                finite = True
            summaries.append(
                "L%d:scale=[%.4f,%.4f],|space|=%.6f,|time|=%.6f,finite=%s" % (
                    idx,
                    float(scale.min().item()), float(scale.max().item()),
                    spatial_abs, temporal_abs, finite,
                )
            )
        if summaries:
            logger.info("[STGASA] iter=%s %s", iteration, " | ".join(summaries))
        net = self.policy.net.module if hasattr(self.policy.net, "module") else self.policy.net
        temporal = getattr(getattr(net, 'vln_bert', None), 'dpft_temporal', None)
        if temporal is not None:
            logger.info('[DPFT-Time] iter=%s alpha=%.6f scale=%.6f', iteration,
                        temporal.alpha.detach().float().item(),
                        torch.tanh(temporal.alpha.detach().float()).item())

    def _write_eval_metric_log(
        self, total, aggregated_states, checkpoint_index=None,
        checkpoint_path=None, eval_time_seconds=None,
    ):
        metric_log_file = getattr(getattr(self.config, "EVAL", None), "METRIC_LOG_FILE", "")
        if not metric_log_file:
            return
        os.makedirs(os.path.dirname(metric_log_file), exist_ok=True)
        with open(metric_log_file, "a") as f:
            f.write(f"Checkpoint path: {checkpoint_path}\n")
            f.write(f"Checkpoint iter: {checkpoint_index}\n")
            f.write(f"Episodes evaluated: {total}\n")
            for key in sorted(aggregated_states.keys()):
                f.write(f"Average episode {key}: {aggregated_states[key]:.6f}\n")
            if eval_time_seconds is not None:
                average_time = eval_time_seconds / max(float(total), 1.0)
                f.write(f"Total eval time (seconds): {eval_time_seconds:.3f}\n")
                f.write(f"Average eval time per episode (seconds): {average_time:.6f}\n")
            f.write("\n")

    def _reset_eval_rng(self):
        seed = int(self.config.TASK_CONFIG.SEED)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori, task_type, mode=None):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        batch_gmap_task_embeddings = []
        batch_gmap_birth_steps, batch_gmap_observation_steps = [], []
        batch_gmap_temporal_valid_masks, batch_gmap_node_type_ids = [], []
        batch_node_positions, batch_adjacency = [], []
        use_topology = bool(self.config.MODEL.mtr and self.config.MODEL.MTR.spatial_recovery)
        use_temporal = bool(getattr(self.config.MODEL, "stgasa", False)) or bool(
            getattr(self.config.MODEL, "dpft_temporal", False)
        ) or bool(getattr(self.config.MODEL, "mtr", False))

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            if use_topology:
                positions, adjacency = gmap.get_topology_targets(gmap_vp_ids)
                batch_node_positions.append(torch.from_numpy(positions))
                batch_adjacency.append(torch.from_numpy(adjacency))
            gmap_step_ids = gmap.get_step_ids(gmap_vp_ids, self.graph_step_id_mode)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)
            temporal_state = None
            if use_temporal:
                temporal_state = gmap.get_temporal_state(gmap_vp_ids, cur_vp[i])
                batch_gmap_node_type_ids.append(torch.LongTensor(
                    temporal_state['node_type_ids'].tolist()
                ))

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            if use_temporal:
                batch_gmap_birth_steps.append(torch.from_numpy(temporal_state['birth_steps']))
                batch_gmap_observation_steps.append(
                    torch.from_numpy(temporal_state['observation_steps'])
                )
                batch_gmap_temporal_valid_masks.append(
                    torch.from_numpy(temporal_state['valid_mask'])
                )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            gmap_step_ids_tensor = torch.LongTensor(gmap_step_ids)
            batch_gmap_step_ids.append(gmap_step_ids_tensor)
            batch_gmap_task_embeddings.append(torch.full_like(gmap_step_ids_tensor, task_type))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_task_embeddings = pad_sequence(batch_gmap_task_embeddings, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()
        if use_temporal:
            batch_gmap_node_type_ids = pad_sequence(batch_gmap_node_type_ids, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        out = {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left, 'gmap_task_embeddings': batch_gmap_task_embeddings
        }
        if use_temporal:
            out['gmap_node_type_ids'] = batch_gmap_node_type_ids

        if use_topology:
            out['gmap_node_positions'] = pad_sequence(batch_node_positions, batch_first=True).cuda()
            adjacency = torch.zeros(bs, max_gmap_len, max_gmap_len, dtype=torch.bool)
            for i, labels in enumerate(batch_adjacency):
                adjacency[i, :labels.size(0), :labels.size(1)] = labels
            out['gmap_adjacency'] = adjacency.cuda()

        if use_temporal:
            out['gmap_birth_steps'] = pad_sequence(
                batch_gmap_birth_steps, batch_first=True, padding_value=0
            ).cuda()
            out['gmap_observation_steps'] = pad_sequence(
                batch_gmap_observation_steps, batch_first=True, padding_value=0
            ).cuda()
            out['gmap_temporal_valid_mask'] = pad_sequence(
                batch_gmap_temporal_valid_masks, batch_first=True, padding_value=False
            ).cuda()
        return out

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]
        return envs, batch

    def train(self):
        rp_enabled, rp_weight, rp_tau = rp_dagger_options(self.config.IL)
        if rp_enabled:
            expected_teacher = {'r2r': 'spl', 'rxr': 'ndtw'}.get(self.config.MODEL.task_type)
            if self.config.IL.expert_policy != expected_teacher:
                raise ValueError('RP-DAgger requires expert_policy=spl for R2R and ndtw for RxR')
            logger.info('[RP-DAgger] enabled=True lambda_reg=%g tau_m=%g; ghost-only regularizer, STOP CE unchanged',
                        rp_weight, rp_tau)
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))
        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        checkpoint_every = int(
            getattr(self.config.IL, "checkpoint_every", log_every)
        )
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')

        if self.config.local_rank < 1:
            config_path = os.path.join(self.config.CHECKPOINT_FOLDER, "config.yaml")
            with open(config_path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Configuration saved to {config_path}")
        
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (
                idx // self.config.IL.decay_interval + 1
            )
            if sample_ratio <= 0.15:
                sample_ratio = 0.0
            logger.info("sample ratio: %s", sample_ratio)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.add_scalar('train/lr', current_lr, cur_iter)
                logger.info(loss_str)
                logger.info(f"lr: {current_lr}")
                if cur_iter % checkpoint_every == 0 or cur_iter == total_iter:
                    self.save_checkpoint(cur_iter)

    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        self.sap_loss = 0.
        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.
            
            with autocast():
                self.rollout('train', ml_weight, sample_ratio)
            self.scaler.scale(self.loss).backward()
            if bool(getattr(self.config.IL, 'use_online_mtr', False)):
                self.scaler.unscale_(self.optimizer)
                self.logs['MTR_grad_norm'].append(self._mtr_grad_norm())
            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        zero_shot_pretrain = bool(getattr(self.config.EVAL, "ZERO_SHOT_PRETRAIN", False))
        if self.local_rank < 1:
            logger.info(
                f"checkpoint_path: {checkpoint_path}  ZERO_SHOT_PRETRAIN={zero_shot_pretrain}  "
                f"pretrained_path={self.config.MODEL.pretrained_path}"
            )
        self._reset_eval_rng()
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        if not zero_shot_pretrain:
            self.config.IL.ckpt_to_load = checkpoint_path
        # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append('POSITION_INFER')
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self._apply_video_topdown_map_resolution()
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            self.config.VIDEO_DIR = self.config.VIDEO_DIR + "_" + self.config.EVAL.SPLIT
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return

        if self.config.EVAL.fast_eval:
            episodes_allowed = self.traj[::5]
        elif self.config.EVAL.EPISODE_ID:
            episodes_allowed = self.config.EVAL.EPISODE_ID
        else:
            episodes_allowed = self.traj
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=episodes_allowed,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=not zero_shot_pretrain,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            log_training_setup=False,
        )
        self.policy.eval()
        self.waypoint_predictor.eval()
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_start_time = time.perf_counter()
        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        local_eval_time = time.perf_counter() - eval_start_time

        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        eval_time = torch.tensor(local_eval_time, dtype=torch.float64).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
            distr.reduce(eval_time, dst=0, op=distr.ReduceOp.MAX)
        total = total.item()
        eval_time = eval_time.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            self._write_eval_metric_log(
                total,
                aggregated_states,
                checkpoint_index=checkpoint_index,
                checkpoint_path=checkpoint_path,
                eval_time_seconds=eval_time,
            )
            logger.info(
                "Evaluation result: checkpoint=%s, iter=%s",
                checkpoint_path,
                checkpoint_index,
            )
            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index if isinstance(checkpoint_index, int) else 0
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            logger.info(
                "Eval inference time: total=%.3f seconds, average_per_episode=%.6f seconds",
                eval_time,
                eval_time / max(float(total), 1.0),
            )
            print(f"Checkpoint iter: {checkpoint_index} ({checkpoint_path})")
            print(f"Episodes evaluated: {total}")

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]

        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()
        
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
 
        instr_max_len = self.config.IL.max_text_len
        instr_pad_id = 1
        if self.config.MODEL.task_type == 'r2r':
            task_type = 1
        elif self.config.MODEL.task_type == 'rxr':
            task_type = 2
        else:
            print("self.config.MODEL.task_type Error")
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause) 
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_task_encoding = batch['txt_task_encoding']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_task_encoding=all_txt_task_encoding,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        
        not_done_index = list(range(self.envs.num_envs)) 
        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION) 
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost, 
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs

        for stepk in range(self.max_len): 
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks
            txt_embeds = all_txt_embeds
            
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug), 
            )

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i) 
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1] 
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                        cur_vp[i], cur_pos[i], cur_embeds,
                                        cand_vp[i], cand_pos[i], cand_embeds,
                                        cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, task_type, mode=mode)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds, 
                'txt_masks': txt_masks,
            })
            if mode == 'train' and bool(getattr(self.config.IL, 'use_online_mtr', False)):
                nav_inputs['mtr_pair_mask'] = sample_symmetric_pair_mask(
                    nav_inputs['gmap_temporal_valid_mask'],
                    float(getattr(self.config.IL, 'mtr_mask_ratio', 0.15)),
                    training=True,
                )
            no_vp_left = nav_inputs.pop('no_vp_left') 
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item() 

            teacher_distances = None
            if mode == 'train':
                rp_enabled, _, _ = rp_dagger_options(self.config.IL)
                teacher_result = self._teacher_action_new(
                    nav_inputs['gmap_vp_ids'], no_vp_left, True, return_costs=rp_enabled,
                )
                if rp_enabled:
                    teacher_actions, teacher_distances = teacher_result
                else:
                    teacher_actions = teacher_result
            elif self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(
                    nav_inputs['gmap_vp_ids'], no_vp_left, False
                )
            if mode == 'train': 
                loss += self._sap_loss(nav_logits, teacher_actions, nav_inputs, cur_vp, teacher_distances)
                if bool(getattr(self.config.IL, 'use_online_mtr', False)):
                    loss += self._online_mtr_loss(nav_outs, self.envs.num_envs)

            # determine action
            if feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                a_t = c.sample().detach()
                use_teacher = (torch.rand_like(a_t, dtype=torch.float) <= sample_ratio) & (teacher_actions >= 0)
                a_t = torch.where(use_teacher, teacher_actions, a_t)

            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING) 
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]: 
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]

                    if self.config.IL.back_algo == 'control': 
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout
                            },
                            'vis_info': vis_info,
                        }
                    )
                    
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp) 
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp, preserve_temporal=True)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    metric['high_level_step'] = stepk
                    if ep_id in self.stat_eps:
                        print("ERROR!!!!!!!!!! ", ep_id)
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        all_txt_ids = torch.cat((all_txt_ids[:i], all_txt_ids[i + 1:]), dim=0)
                        all_txt_task_encoding = torch.cat((all_txt_task_encoding[:i], all_txt_task_encoding[i + 1:]), dim=0)
                        all_txt_masks = torch.cat((all_txt_masks[:i], all_txt_masks[i + 1:]), dim=0)
                        all_txt_embeds = torch.cat((all_txt_embeds[:i], all_txt_embeds[i + 1:]), dim=0)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, \
                                                                        max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions 
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
