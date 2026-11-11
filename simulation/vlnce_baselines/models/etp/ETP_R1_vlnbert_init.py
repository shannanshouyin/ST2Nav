import torch
from habitat import logger


def _canonical_residual_scale_key(key):
    """Map supported pretrain/online prefixes to an online model key.

    Transformers 3.x treats every key containing ``gamma``/``beta`` as a
    legacy LayerNorm key and renames it to ``weight``/``bias`` inside
    ``from_pretrained``. ST-GASA uses ``gamma`` for a genuine head-wise
    residual parameter, so this key must bypass that compatibility path.
    """
    key = str(key)
    prefixes = ("module.", "net.vln_bert.", "vln_bert.", "bert.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break

    if (
        key.startswith("global_encoder.encoder.x_layers.")
        and key.endswith(".stgasa_relation.gamma")
    ):
        return key
    return None


def _extract_residual_scale_weights(state_dict, remove=False):
    """Collect scale tensors before HuggingFace can rename their keys."""
    extracted = {}
    for source_key in list(state_dict.keys()):
        target_key = _canonical_residual_scale_key(source_key)
        if target_key is None:
            continue
        extracted[target_key] = state_dict[source_key]
        if remove:
            state_dict.pop(source_key)
    return extracted


def _restore_residual_scale_weights(model, scale_weights):
    """Restore pretraining residual scales after ``from_pretrained``."""
    if not scale_weights:
        return {}, {}
    model_state = model.state_dict()
    restored = {}
    skipped = {}
    with torch.no_grad():
        for key, value in scale_weights.items():
            if key not in model_state:
                skipped[key] = "target_missing_or_disabled"
                continue
            expected_shape = tuple(model_state[key].shape)
            actual_shape = tuple(value.shape)
            if actual_shape != expected_shape:
                skipped[key] = {
                    "checkpoint": actual_shape,
                    "model": expected_shape,
                }
                continue
            model_state[key].copy_(
                value.to(
                    device=model_state[key].device,
                    dtype=model_state[key].dtype,
                )
            )
            restored[key] = {
                "min": float(model_state[key].detach().float().min().item()),
                "max": float(model_state[key].detach().float().max().item()),
            }
    return restored, skipped


def _filter_mismatched_stgasa_weights(state_dict, config):
    if not state_dict:
        return state_dict
    time_bucket = int(getattr(config, "stgasa_time_max_bucket", 6))
    relation_hidden = int(getattr(config, "stgasa_relation_hidden_size", 128))
    num_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)
    mtr_hidden = int(getattr(config, "mtr_relation_hidden_size", 128))
    use_stgasa = bool(getattr(config, "stgasa", False))
    use_mtr = bool(getattr(config, "mtr", False))
    use_dpft_temporal = bool(getattr(config, "dpft_temporal", False))
    expected_dpft_shapes = {
        'alpha': (),
        'birth_embedding.weight': (time_bucket + 1, relation_hidden),
        'observation_embedding.weight': (time_bucket + 1, relation_hidden),
        'projection.weight': (hidden_size, 2 * relation_hidden),
        'projection.bias': (hidden_size,),
        'norm.weight': (hidden_size,),
        'norm.bias': (hidden_size,),
    }
    expected_relation_shapes = {
        "gamma": (num_heads,),
        "birth_embedding.weight": (
            time_bucket * 2 + 2, relation_hidden,
        ),
        "observation_embedding.weight": (
            time_bucket * 2 + 2, relation_hidden,
        ),
        "type_pair_embedding.weight": (9, relation_hidden),
        "relation_mlp.0.weight": (relation_hidden, 3 * relation_hidden),
        "relation_mlp.0.bias": (relation_hidden,),
        "relation_mlp.2.weight": (relation_hidden,),
        "relation_mlp.2.bias": (relation_hidden,),
        "head_projection.weight": (num_heads, relation_hidden),
        "head_projection.bias": (num_heads,),
    }
    mtr_classes = 2 * time_bucket + 1
    expected_mtr_shapes = {
        "pair_encoder.0.weight": (mtr_hidden, hidden_size * 4),
        "pair_encoder.0.bias": (mtr_hidden,),
        "pair_encoder.2.weight": (mtr_hidden,),
        "pair_encoder.2.bias": (mtr_hidden,),
        "instruction_projection.0.weight": (mtr_hidden, hidden_size),
        "instruction_projection.0.bias": (mtr_hidden,),
        "fusion_norm.weight": (mtr_hidden,),
        "fusion_norm.bias": (mtr_hidden,),
        "birth_head.weight": (mtr_classes, mtr_hidden),
        "birth_head.bias": (mtr_classes,),
        "observation_head.weight": (mtr_classes, mtr_hidden),
        "observation_head.bias": (mtr_classes,),
    }
    if use_mtr and bool(getattr(config, 'mtr_spatial_recovery', False)):
        from .stgasa import MTRPredictionHead
        # Preserve RNG state: shape checking must not affect model initialization.
        with torch.random.fork_rng(devices=[]):
            expected_mtr_shapes.update({
                name: tuple(value.shape)
                for name, value in MTRPredictionHead(config).state_dict().items()
            })
    removed = []
    old_fragments = (
        "st_dist_embed", "st_mlp", "st_rel_mlp", "st_phi",
        "st_psi", "st_lambda_v", "st_time_bucket",
        "st_time_embedding", "st_time_key_proj", "st_dir_embedding",
        "st_dir_bias_proj", "st_value_proj", "st_gate_t", "st_gate_d", "st_gate_v",
        "instruction_gate", "age_embedding", "spatial_mask_bias",
        "st_planner", "st_progress_head", "st_progress_residual", "st_progress_beta",
        "mstr_head.age_head", "mstr_head.spatial_head",
    )
    for key in list(state_dict.keys()):
        if 'dpft_temporal.' in key:
            suffix = key.split('dpft_temporal.', 1)[1]
            expected_shape = expected_dpft_shapes.get(suffix)
            actual_shape = tuple(state_dict[key].shape)
            if not use_dpft_temporal or actual_shape != expected_shape:
                removed.append((key, actual_shape, expected_shape if use_dpft_temporal else 'dpft_temporal_disabled'))
                state_dict.pop(key)
            continue
        if any(fragment in key for fragment in old_fragments):
            removed.append((key, "old_stgasa_v1", None))
            state_dict.pop(key)
            continue
        relation_marker = "stgasa_relation."
        if relation_marker in key:
            if not use_stgasa:
                removed.append((key, tuple(state_dict[key].shape), "stgasa_disabled"))
                state_dict.pop(key)
                continue
            suffix = key.split(relation_marker, 1)[1]
            expected_shape = expected_relation_shapes.get(suffix)
            actual_shape = tuple(state_dict[key].shape)
            if suffix == "type_pair_embedding.weight" and actual_shape == (10, relation_hidden):
                state_dict[key] = state_dict[key][:9].clone()
                logger.warning("Migrated legacy 10-row type-pair embedding to the fixed 9-row table: %s", key)
                continue
            if expected_shape is None or actual_shape != expected_shape:
                removed.append((key, actual_shape, expected_shape or "disabled_or_unknown"))
                state_dict.pop(key)
                continue
        mtr_marker = "mtr_head."
        if mtr_marker in key:
            if not use_mtr:
                removed.append((key, tuple(state_dict[key].shape), "mtr_disabled"))
                state_dict.pop(key)
                continue
            suffix = key.split(mtr_marker, 1)[1]
            expected_shape = expected_mtr_shapes.get(suffix)
            actual_shape = tuple(state_dict[key].shape)
            if expected_shape is None or actual_shape != expected_shape:
                removed.append((key, actual_shape, expected_shape or "unknown_mtr_parameter"))
                state_dict.pop(key)
                continue
    if removed:
        logger.warning(
            "ST-GASA/MTR/DPFT or legacy checkpoint keys ignored or reinitialized: %s",
            removed[:24],
        )
    return state_dict


def _migrate_legacy_mstr_head(state_dict):
    """Map legacy MSTR names so compatibility checks can identify the old head."""
    migrated = []
    dropped = []
    for key in list(state_dict.keys()):
        marker = "mstr_head."
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        if suffix.startswith(("pair_encoder.", "birth_head.", "observation_head.")):
            new_key = key.replace(marker, "mtr_head.", 1)
            state_dict.setdefault(new_key, state_dict[key])
            migrated.append((key, new_key))
        else:
            dropped.append(key)
        state_dict.pop(key)
    if migrated:
        logger.warning(
            "Mapped legacy MSTR parameter names before MTR compatibility "
            "handling: %s", migrated[:12]
        )
    if dropped:
        logger.warning("Dropped legacy MSTR age/spatial parameters: %s", dropped[:12])
    return state_dict


def _drop_incompatible_recovery_mtr_weights(state_dict):
    """Drop an old post-ST-GASA MTR head while preserving the model trunk."""
    mtr_keys = [key for key in state_dict if "mtr_head." in key]
    if not mtr_keys:
        return state_dict
    if any("mtr_head.instruction_projection." in key for key in mtr_keys):
        return state_dict
    for key in mtr_keys:
        state_dict.pop(key)
    logger.warning(
        "Dropped %d legacy post-ST-GASA MTR tensors; the new "
        "recover-then-reason MTR head will be initialized from scratch.",
        len(mtr_keys),
    )
    return state_dict


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'bert_config/xlm-roberta-base'
    else:
        cfg_name = 'bert_config/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None, dropout_rate=0.1):
    
    from transformers import PretrainedConfig
    from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import GlocalTextPathNavCMT

    model_class = GlocalTextPathNavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    keywords = ['graph_query_text', 'graph_attentioned_txt_embeds_transform', 'global_sap_head']
    if model_name_or_path:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        for k, v in ckpt_weights.items():
            clean_k = k[7:] if k.startswith('module.') else k
            if clean_k.startswith('bert.'):
                new_ckpt_weights[clean_k[5:]] = v
            new_ckpt_weights[clean_k] = v
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            if any(key in k for key in keywords):
                new_ckpt_weights['bert.' + k] = v
        new_ckpt_weights = _migrate_legacy_mstr_head(new_ckpt_weights)
    
    cfg_name = 'bert_config/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = 512
    vis_config.use_depth_embedding = config.use_depth_embedding
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4

    vis_config.num_l_layers = 12
    vis_config.num_pano_layers = 2
    vis_config.num_x_layers = 4
    vis_config.graph_sprels = config.use_sprels
    vis_config.glocal_fuse = 'global'

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_pano_embedding = config.fix_pano_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True

    vis_config.pred_head_dropout_prob = dropout_rate
    vis_config.hidden_dropout_prob = dropout_rate
    vis_config.attention_probs_dropout_prob = dropout_rate

    vis_config.use_lang2visn_attn = True

    vis_config.max_txt_task_embeddings = 4
    vis_config.max_gmap_task_embeddings = 3
    vis_config.stgasa = getattr(config, "stgasa", False)
    vis_config.dpft_temporal = bool(getattr(config, "dpft_temporal", False))
    st_cfg = getattr(config, "STGASA", None)
    vis_config.stgasa_relation_hidden_size = getattr(st_cfg, "relation_hidden_size", 128) if st_cfg is not None else 128
    vis_config.stgasa_time_max_bucket = getattr(st_cfg, "time_max_bucket", 6) if st_cfg is not None else 6
    vis_config.mtr = getattr(config, "mtr", False)
    vis_config.stgasa_eval_gamma = getattr(st_cfg, "eval_gamma", None)
    mtr_cfg = getattr(config, "MTR", None)
    vis_config.mtr_eval_predict = getattr(mtr_cfg, "eval_predict", False)
    vis_config.mtr_eval_mask_ratio = getattr(mtr_cfg, "eval_mask_ratio", 0.15)
    vis_config.mtr_relation_hidden_size = getattr(mtr_cfg, "relation_hidden_size", 128) if mtr_cfg is not None else 128
    vis_config.mtr_spatial_recovery = bool(getattr(mtr_cfg, 'spatial_recovery', False))
    new_ckpt_weights = _filter_mismatched_stgasa_weights(new_ckpt_weights, vis_config)
    new_ckpt_weights = _drop_incompatible_recovery_mtr_weights(new_ckpt_weights)

    # HuggingFace's legacy LayerNorm conversion would rename these genuine
    # ST-GASA parameters (gamma->weight). Remove them from the
    # generic loader and restore them explicitly after model construction.
    residual_scale_weights = _extract_residual_scale_weights(
        new_ckpt_weights, remove=True
    )

    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
    # Shape-compatible old MTR heads were trained on post-ST-GASA features.
    # Detect the new instruction-conditioned recovery marker and otherwise
    # reset only MTR while retaining every compatible ETP-R1/ST-GASA weight.
    visual_model.reset_missing_st_gasa_parameters(new_ckpt_weights.keys())
    logger.info(
        'Evaluation overrides: STGASA.eval_gamma=%s, MTR.eval_predict=%s, '
        'MTR.eval_mask_ratio=%s (deterministic pair selection)',
        vis_config.stgasa_eval_gamma, vis_config.mtr_eval_predict,
        vis_config.mtr_eval_mask_ratio,
    )
    restored_scales, skipped_scales = _restore_residual_scale_weights(
        visual_model, residual_scale_weights
    )
    if restored_scales:
        logger.info(
            "Restored ST-GASA residual scales after legacy HF conversion: %s",
            restored_scales,
        )
    if skipped_scales:
        logger.warning(
            "ST-GASA residual scales found in checkpoint but not restored: %s",
            skipped_scales,
        )
    return visual_model
