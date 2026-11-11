"""Training-only relative-distance regularization; no model or Habitat state."""
import math

import torch


def rp_dagger_options(il_config):
    cfg = getattr(il_config, 'RP_DAGGER', None)
    enabled = bool(getattr(cfg, 'enabled', False))
    weight = float(getattr(cfg, 'lambda_reg', 0.1))
    tau = float(getattr(cfg, 'tau', 3.0))
    if enabled:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('IL.RP_DAGGER.lambda_reg must be finite and nonnegative')
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('IL.RP_DAGGER.tau must be finite and positive (meters)')
    return enabled, weight, tau


def relative_distance_regularizer(logits, teacher_actions, distances, ghost_mask, tau):
    """Sum per-state conditional ghost risks, matching the existing summed CE.

    Distances are detached metric GT labels, not normalized GASA inputs.
    NaN marks STOP/visited/padding. An unreachable (+inf) ghost has cost 1
    if the state has a reachable teacher. STOP/ignored states contribute zero.
    """
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError('RP-DAgger tau must be finite and positive')
    if distances.shape != logits.shape or ghost_mask.shape != logits.shape:
        raise ValueError('RP-DAgger labels/mask must match [batch, graph_nodes] logits')
    if teacher_actions.shape != logits.shape[:1]:
        raise ValueError('RP-DAgger teacher_actions must have shape [batch]')
    scores = logits.float()
    distances = distances.detach().to(device=scores.device, dtype=torch.float32)
    ghosts = ghost_mask.bool() & torch.isfinite(scores)
    ghosts = ghosts & ~torch.isnan(distances) & (distances >= 0)
    ghosts = ghosts.clone()
    ghosts[:, 0] = False
    teacher_index = teacher_actions.clamp(min=0, max=scores.size(1) - 1).unsqueeze(1)
    teacher_distance = distances.gather(1, teacher_index)
    valid = ((teacher_actions > 0) & (teacher_actions < scores.size(1)) &
             ghosts.gather(1, teacher_index).squeeze(1) &
             torch.isfinite(teacher_distance.squeeze(1)))

    # Avoid all--inf softmax for STOP/ignored/empty states. Their loss and
    # gradient are subsequently exactly zero, including the STOP gradient.
    masked_scores = scores.masked_fill(~ghosts, -float('inf'))
    masked_scores = torch.where(valid.unsqueeze(1), masked_scores, torch.zeros_like(scores))
    probabilities = torch.softmax(masked_scores, dim=1)
    reference = torch.where(valid.unsqueeze(1), teacher_distance, torch.zeros_like(teacher_distance))
    safe_distances = torch.where(ghosts, distances, torch.zeros_like(distances))
    excess = (safe_distances - reference).clamp(min=0)
    costs = -torch.expm1(-excess / float(tau))
    costs = costs.masked_fill(~(ghosts & valid.unsqueeze(1)), 0)
    risk = (probabilities * costs).sum(dim=1)
    return risk.sum(), valid.sum().detach()
