"""Load ST2Nav inference modules without importing its training runtime.

Neural modules and the ETP forward method are read from the pinned official
source. No model layers or weight names are silently substituted.
"""
import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import math
import os
import pickle
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache/etpr1/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path.home() / ".cache/etpr1/huggingface/hub"))
os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache/etpr1/torch"))

import clip
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, PretrainedConfig

from model_options import MODELS, model_id

SELECTED_MODEL = model_id(os.environ.get('ETPR1_MODEL', 'st2nav'))
WORKSPACE = ROOT
UPSTREAM = ROOT / 'vendor' / MODELS[SELECTED_MODEL]['source']
HABITAT = ROOT / "vendor/habitat-lab"


class ArchivedTrainingConfig(dict):
    """Inert metadata container; inference never reads this training config."""


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("habitat.config.default", "Config"):
            return ArchivedTrainingConfig
        return super().find_class(module, name)


CHECKPOINT_PICKLE = types.ModuleType("etpr1_checkpoint_pickle")
CHECKPOINT_PICKLE.Unpickler = CheckpointUnpickler
CHECKPOINT_PICKLE.load = pickle.load
CHECKPOINT_PICKLE.loads = pickle.loads


def source_definition(path, class_name, method_name=None, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method_name:
        definition = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), definition], type_ignores=[])
    ast.fix_missing_locations(module)
    context = dict(namespace or {})
    exec(compile(module, str(path), "exec"), context)
    return context[method_name or class_name]


def setup_packages():
    # Skip only package registration side effects which import Habitat trainers.
    # Every imported implementation remains the actual official source file.
    for name in ["vlnce_baselines", "vlnce_baselines.common", "vlnce_baselines.models", "vlnce_baselines.models.etp", "vlnce_baselines.waypoint_pred", "vlnce_baselines.waypoint_pred.transformer", "vlnce_baselines.waypoint_pred.transformer.pytorch_transformer"]:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(UPSTREAM / name.replace(".", "/"))]
        sys.modules[name] = module


def asset_path(kind):
    spec = json.loads((ROOT / 'weights/manifest.json').read_text())[kind]
    override = 'NAV_CHECKPOINT' if kind == 'model' else 'NAV_WAYPOINT'
    return Path(os.environ.get(override, str(ROOT / 'weights' / spec['filename'])))


def verify_inputs():
    sources = json.loads((ROOT / 'source_manifest.json').read_text())
    for relative, expected in sources['files'].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Missing or changed source: ' + relative)
    assets = json.loads((ROOT / 'weights/manifest.json').read_text())
    for kind, spec in assets.items():
        path = asset_path(kind)
        if not path.is_file():
            raise RuntimeError('Install required asset: ' + spec['filename'])
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != spec['sha256']:
            raise RuntimeError('Checkpoint checksum mismatch: ' + spec['filename'])


def strict_load(module, state, name):
    result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{name} checkpoint mismatch")
    return len(state)


class RGBEncoder(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.model = clip.model.build_model(state).float().eval()
        # build_model is followed by strict loading rather than accepting missing layers.
        state = {k: v for k, v in state.items() if k not in {"input_resolution", "context_length", "vocab_size"}}
        strict_load(self.model, state, "CLIP")
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1), persistent=False)

    def forward(self, observations):
        rgb = observations["rgb"].permute(0, 3, 1, 2).float() / 255.0
        return self.model.encode_image(((rgb - self.mean) / self.std).contiguous()).float()


class DepthEncoder(nn.Module):
    def __init__(self, state):
        super().__init__()
        path = HABITAT / "habitat_baselines/rl/ddppo/policy/resnet.py"
        spec = importlib.util.spec_from_file_location("etpr1_habitat_resnet", path)
        resnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resnet)
        encoder = source_definition(HABITAT / "habitat_baselines/rl/ddppo/policy/resnet_policy.py", "ResNetEncoder", namespace={"nn": nn, "torch": torch, "F": F})
        space = types.SimpleNamespace(spaces={"depth": types.SimpleNamespace(shape=(256, 256, 1))})
        self.visual_encoder = encoder(space, baseplanes=32, ngroups=16, make_backbone=resnet.resnet50, normalize_visual_inputs=False)
        strict_load(self.visual_encoder, state, "depth")

    def forward(self, observations):
        return self.visual_encoder(observations)


class OfficialPolicy(nn.Module):
    def __init__(self, checkpoint=None, device="cpu"):
        super().__init__()
        verify_inputs()
        setup_packages()
        checkpoint = checkpoint or asset_path('model')
        self.device = torch.device(device)
        # The source and ZIP revision are pinned and locally hashed before loading.
        data = torch.load(checkpoint, map_location="cpu", weights_only=False, pickle_module=CHECKPOINT_PICKLE)
        policy_state = {k.replace("net.module.", "net.", 1): v for k, v in data["state_dict"].items()}
        self.checkpoint_iteration = data.get("iteration")
        saved_config = data.get('config', {}).get('MODEL', {})
        def namespace(value):
            return types.SimpleNamespace(**{k: namespace(v) for k, v in value.items()}) if isinstance(value, dict) else value
        self.runtime_config = namespace(saved_config)
        self.graph_step_id_mode = data.get('graph_step_id_mode', 'arrival')
        self.model_id = SELECTED_MODEL
        del data
        def branch(prefix):
            return {k[len(prefix):]: v for k, v in policy_state.items() if k.startswith(prefix)}
        config = PretrainedConfig.from_pretrained(str(UPSTREAM / "bert_config/xlm-roberta-base"), local_files_only=True)
        settings = dict(type_vocab_size=2, max_action_steps=100, image_feat_size=512, use_depth_embedding=True, depth_feat_size=128, angle_feat_size=4, num_l_layers=12, num_pano_layers=2, num_x_layers=4, graph_sprels=True, glocal_fuse="global", fix_lang_embedding=False, fix_pano_embedding=False, update_lang_bert=True, output_attentions=True, pred_head_dropout_prob=0.1, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, use_lang2visn_attn=True, max_txt_task_embeddings=4, max_gmap_task_embeddings=3)
        for key, value in settings.items():
            setattr(config, key, value)
        if SELECTED_MODEL != 'etpr1':
            for key in ('stgasa', 'dpft_temporal', 'mtr', 'st_planner'):
                setattr(config, key, bool(saved_config.get(key, False)))
            groups = {
                'STGASA': {'relation_hidden_size': ('stgasa_relation_hidden_size', 128),
                           'time_max_bucket': ('stgasa_time_max_bucket', 6),
                           'time_clip_steps': ('stgasa_time_clip_steps', 20),
                           'gate_init': ('stgasa_gate_init', 0.),
                           'eval_gamma': ('stgasa_eval_gamma', None)},
                'MTR': {'relation_hidden_size': ('mtr_relation_hidden_size', 128),
                        'spatial_recovery': ('mtr_spatial_recovery', False),
                        'eval_predict': ('mtr_eval_predict', False),
                        'eval_mask_ratio': ('mtr_eval_mask_ratio', .15)},
                'ST_PLANNER': {key: ('st_planner_' + key, value) for key, value in
                               {'num_bins': 10, 'max_label_dist': 3., 'beta_init': 0.,
                                'use_progress_score': True, 'use_type_embedding': True,
                                'use_step_embedding': True, 'type_embedding_mode': 'learned',
                                'step_embedding_mode': 'learned'}.items()},
            }
            for group, fields in groups.items():
                for key, (attribute, default) in fields.items():
                    setattr(config, attribute, saved_config.get(group, {}).get(key, default))
            if SELECTED_MODEL == 'st2nav':
                # Source defaults omitted by older checkpoints are explicit at inference.
                if not hasattr(self.runtime_config, 'MTR'):
                    self.runtime_config.MTR = types.SimpleNamespace()
                self.runtime_config.MTR.spatial_recovery = config.mtr_spatial_recovery
        neural = importlib.import_module("vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt")
        self.vln_bert = neural.GlocalTextPathNavCMT(config)
        self.loaded_keys = {"planner": strict_load(self.vln_bert, branch("net.vln_bert."), "planner")}
        self.rgb_encoder = RGBEncoder(branch("net.rgb_encoder.model."))
        self.depth_encoder = DepthEncoder(branch("net.depth_encoder.visual_encoder."))
        self.loaded_keys["rgb"] = len(branch("net.rgb_encoder.model."))
        self.loaded_keys["depth"] = len(branch("net.depth_encoder.visual_encoder."))
        handled = ("net.vln_bert.", "net.rgb_encoder.model.", "net.depth_encoder.visual_encoder.")
        self.unused_checkpoint_keys = [k for k in policy_state if not k.startswith(handled)]
        if self.unused_checkpoint_keys:
            raise RuntimeError(f"Unhandled checkpoint keys: {self.unused_checkpoint_keys[:10]}")
        del policy_state
        wp_module = importlib.import_module("vlnce_baselines.waypoint_pred.TRM_net")
        self.waypoint_predictor = wp_module.BinaryDistPredictor_TRM(device=self.device)
        wp = torch.load(asset_path('waypoint'), map_location="cpu", weights_only=False)
        self.loaded_keys["waypoint"] = strict_load(self.waypoint_predictor, wp["predictor"]["state_dict"], "waypoint")
        del wp
        self.drop_env = nn.Dropout(p=0.4)
        self.space_pool_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=2))
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=2))
        self.pano_img_idxes = np.arange(12, dtype=np.int64)
        def angle_features(headings):
            return torch.stack([torch.sin(headings), torch.cos(headings), torch.zeros_like(headings), torch.ones_like(headings)]).float().T
        self.pano_angle_fts = angle_features(torch.from_numpy((1 - self.pano_img_idxes / 12) * 2 * math.pi))
        self.tokenizer = AutoTokenizer.from_pretrained(str(UPSTREAM / "bert_config/xlm-roberta-base"), local_files_only=True)
        wp_utils = importlib.import_module("vlnce_baselines.waypoint_pred.utils")
        waypoint_nms=wp_utils.nms
        self.waypoint_postprocess_revision='upstream'
        if self.model_id=='st2nav':
            from .waypoint_input import circular_waypoint_nms
            waypoint_nms=circular_waypoint_nms
            self.waypoint_postprocess_revision='periodic-angle-no-padding-actions-v1'
        original_forward = source_definition(UPSTREAM / "vlnce_baselines/models/R1Policy.py", "ETP", "forward", {"torch": torch, "np": np, "math": math, "deepcopy": deepcopy, "F": F, "angle_feature_torch": angle_features, "nms": waypoint_nms})
        self.forward = types.MethodType(original_forward, self)
        self.to(self.device).eval()

    def language(self, instruction):
        batch = self.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=150)
        ids = batch["input_ids"].to(self.device)
        # Official R2R uses XLM-R padding=1; token 0 is the valid sentence start.
        masks = ids != self.tokenizer.pad_token_id
        tasks = torch.ones_like(ids)
        return self(mode="language", txt_ids=ids, txt_task_encoding=tasks, txt_masks=masks), masks

    def encode_scan(self, observations, candidate_filter=None):
        from .waypoint_input import ordered_views
        observations=ordered_views(observations)
        observations = {key: torch.from_numpy(value).to(self.device) for key, value in observations.items()}
        waypoints = self(mode="waypoint", observations=observations, waypoint_predictor=self.waypoint_predictor, in_train=False)
        if candidate_filter is not None:
            keep = np.asarray([candidate_filter(a, d) for a, d in zip(waypoints["cand_angles"][0], waypoints["cand_distances"][0])], dtype=bool)
            for name in ["cand_rgb", "cand_depth", "cand_angle_fts"]:
                value = waypoints[name][0]
                waypoints[name][0] = value[torch.as_tensor(keep, device=value.device)]
            waypoints["cand_img_idxes"][0] = waypoints["cand_img_idxes"][0][keep]
            for name in ["cand_angles", "cand_distances"]:
                waypoints[name][0] = [v for v, ok in zip(waypoints[name][0], keep) if ok]
        candidate_slots = np.zeros(12, dtype=bool)
        candidate_slots[waypoints["cand_img_idxes"][0]] = True
        indexes = torch.from_numpy(~candidate_slots).to(self.device)
        rgb = torch.cat([waypoints["cand_rgb"][0], waypoints["pano_rgb"][0][indexes]])[None]
        depth = torch.cat([waypoints["cand_depth"][0], waypoints["pano_depth"][0][indexes]])[None]
        # Official pano angle feature lives on CPU in the upstream implementation.
        locations = torch.cat([waypoints["cand_angle_fts"][0].cpu(), waypoints["pano_angle_fts"][~candidate_slots]])[None].to(self.device)
        count = len(waypoints["cand_angles"][0])
        nav_types = torch.tensor([[1] * count + [0] * int((~candidate_slots).sum())], device=self.device)
        panorama, masks = self(mode="panorama", rgb_fts=rgb, dep_fts=depth, loc_fts=locations, nav_types=nav_types, view_lens=torch.tensor([rgb.shape[1]], device=self.device))
        return waypoints, panorama, masks
