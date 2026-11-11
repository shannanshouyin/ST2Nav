import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


STOP_TYPE_ID = 0
CURRENT_TYPE_ID = 1
VISITED_TYPE_ID = 2
GHOST_TYPE_ID = 3


def nonnegative_log_bucket(values, max_bucket):
    """Shared time magnitude buckets: 0, 1-2, 3-6, 7-14, ... ."""
    values = values.long().clamp(min=0)
    return torch.floor(torch.log2(values.float() + 1.0)).long().clamp(max=int(max_bucket))


def signed_log_bucket(values, max_bucket):
    """Map signed integer offsets to ``[0, 2 * max_bucket]``."""
    values = values.long()
    magnitude = nonnegative_log_bucket(values.abs(), max_bucket)
    signed = torch.sign(values).long() * magnitude
    return signed + int(max_bucket)


def directed_type_pair_ids(node_type_ids):
    """Return ids for the nine directed current/visited/ghost type pairs."""
    src = node_type_ids.unsqueeze(2)
    dst = node_type_ids.unsqueeze(1)
    return ((src - 1).clamp(min=0, max=2) * 3 + (dst - 1).clamp(min=0, max=2)).long()


def temporal_pair_valid_mask(node_valid_mask):
    node_valid_mask = node_valid_mask.bool()
    return node_valid_mask.unsqueeze(2) & node_valid_mask.unsqueeze(1)


def sample_symmetric_pair_mask(node_valid_mask, ratio, training=True):
    """Sample a fixed ratio of unordered relations and mask both directions.

    Only the selected entries in the pairwise relation matrices are masked.
    Node features and every unselected relation remain unchanged.  A node may
    therefore participate in multiple masked relations without being masked
    itself.
    """
    valid = temporal_pair_valid_mask(node_valid_mask)
    batch_size, node_count, _ = valid.shape
    diagonal = torch.eye(node_count, dtype=torch.bool, device=valid.device).unsqueeze(0)
    upper_valid = valid & torch.triu(torch.ones_like(valid, dtype=torch.bool), diagonal=1)
    pair_mask = torch.zeros_like(valid)
    ratio = float(ratio)
    if ratio <= 0:
        return pair_mask

    for batch_idx in range(batch_size):
        pairs = upper_valid[batch_idx].nonzero(as_tuple=False)
        if pairs.numel() == 0:
            continue
        num_selected = max(1, int(math.ceil(pairs.size(0) * ratio)))
        num_selected = min(num_selected, pairs.size(0))
        if training:
            order = torch.randperm(pairs.size(0), device=pairs.device)[:num_selected]
        else:
            scores = pairs[:, 0] * max(node_count, 1) + pairs[:, 1]
            order = torch.argsort(scores)[:num_selected]
        selected = pairs[order]
        pair_mask[batch_idx, selected[:, 0], selected[:, 1]] = True
        pair_mask[batch_idx, selected[:, 1], selected[:, 0]] = True
    return pair_mask & valid & ~diagonal


class STGASARelationEncoder(nn.Module):
    """Fixed temporal relation bias shared by pretraining and online models.

    Spatial shortest-path bias remains in the original GASA implementation.
    This module adds only birth delta, observation delta and directed type-pair
    relations, scaled by a zero-initialized head-wise residual ``gamma``.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = int(config.num_attention_heads)
        self.max_time_bucket = int(getattr(config, "stgasa_time_max_bucket", 6))
        self.relation_hidden = int(getattr(config, "stgasa_relation_hidden_size", 128))
        self.initializer_range = float(getattr(config, "initializer_range", 0.02))
        self.mask_bucket_id = 2 * self.max_time_bucket + 1

        # The final entry in each temporal table is a distinct learnable MASK.
        self.birth_embedding = nn.Embedding(self.mask_bucket_id + 1, self.relation_hidden)
        self.observation_embedding = nn.Embedding(self.mask_bucket_id + 1, self.relation_hidden)
        self.type_pair_embedding = nn.Embedding(9, self.relation_hidden)
        self.relation_mlp = nn.Sequential(
            nn.Linear(3 * self.relation_hidden, self.relation_hidden),
            nn.GELU(),
            nn.LayerNorm(self.relation_hidden),
        )
        self.head_projection = nn.Linear(self.relation_hidden, self.num_heads)
        self.gamma = nn.Parameter(torch.zeros(self.num_heads))
        self.eval_gamma = getattr(config, "stgasa_eval_gamma", None)
        if self.eval_gamma is not None and not math.isfinite(float(self.eval_gamma)):
            raise ValueError("STGASA.eval_gamma must be null or a finite number")

        self.last_stats = {}
        self.reset_parameters(config)

    def reset_parameters(self, config=None):
        initializer_range = (
            float(getattr(config, "initializer_range", self.initializer_range))
            if config is not None else self.initializer_range
        )
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gamma)

    def _relation_embedding(self, ids, mask, probabilities, embedding):
        relation = embedding(ids)
        if mask is None:
            return relation
        mask = mask.bool()
        if probabilities is None:
            # Backward-compatible fallback. The recover-then-reason pipeline
            # always supplies probabilities for masked relations.
            masked_ids = torch.where(
                mask, torch.full_like(ids, self.mask_bucket_id), ids
            )
            return embedding(masked_ids)

        num_classes = 2 * self.max_time_bucket + 1
        probabilities = probabilities.to(dtype=embedding.weight.dtype)
        recovered = torch.matmul(
            probabilities, embedding.weight[:num_classes]
        ).to(dtype=relation.dtype)
        return torch.where(mask.unsqueeze(-1), recovered, relation)

    def forward(
        self,
        birth_steps,
        observation_steps,
        node_type_ids,
        node_valid_mask,
        birth_relation_mask=None,
        observation_relation_mask=None,
        birth_relation_probabilities=None,
        observation_relation_probabilities=None,
    ):
        pair_valid = temporal_pair_valid_mask(node_valid_mask)
        birth_delta = birth_steps.unsqueeze(2) - birth_steps.unsqueeze(1)
        observation_delta = observation_steps.unsqueeze(2) - observation_steps.unsqueeze(1)

        birth_ids = signed_log_bucket(birth_delta, self.max_time_bucket)
        observation_ids = signed_log_bucket(
            observation_delta, self.max_time_bucket
        )
        type_pair_ids = directed_type_pair_ids(node_type_ids)

        relation = torch.cat(
            [
                self._relation_embedding(
                    birth_ids, birth_relation_mask,
                    birth_relation_probabilities, self.birth_embedding,
                ),
                self._relation_embedding(
                    observation_ids, observation_relation_mask,
                    observation_relation_probabilities,
                    self.observation_embedding,
                ),
                self.type_pair_embedding(type_pair_ids),
            ],
            dim=-1,
        )
        encoded = self.relation_mlp(relation)
        raw_bias = self.head_projection(encoded).permute(0, 3, 1, 2).contiguous()
        scale = torch.tanh(self.gamma)
        if not self.training and self.eval_gamma is not None:
            scale = torch.tanh(torch.full_like(self.gamma, float(self.eval_gamma)))
        temporal_bias = raw_bias * scale[None, :, None, None]
        temporal_bias = temporal_bias * pair_valid[:, None].to(temporal_bias.dtype)

        with torch.no_grad():
            self.last_stats = {
                "scale": scale.detach(),
                "raw_bias_abs_mean": raw_bias.detach().abs().mean(),
                "bias_abs_mean": temporal_bias.detach().abs().mean(),
                "bias_abs_max": temporal_bias.detach().abs().max(),
                "bias_finite": torch.isfinite(temporal_bias.detach()).all(),
            }
        return temporal_bias, self.last_stats


def shortest_path_distances(cost):
    """Differentiable Floyd-Warshall, with no in-place updates of autograd inputs."""
    distance = cost.float()
    for k in range(cost.size(-1)):
        distance = torch.minimum(distance, distance[:, :, k:k+1] + distance[:, k:k+1, :])
    return distance


class VisualTopologyRecovery(nn.Module):
    """Predict the observed graph from visual features and XYZ only.

    This interface deliberately cannot receive teacher edges, shortest distances,
    step IDs, temporal relations or topology-derived position channels.
    """

    def __init__(self, config):
        super().__init__()
        self.width = int(getattr(config, "mtr_relation_hidden_size", 128))
        self.heads = 4
        if self.width % self.heads:
            raise ValueError('MTR relation_hidden_size must be divisible by 4')
        self.input_projection = nn.Linear(config.hidden_size, self.width)
        self.position_projection = nn.Linear(3, self.width)
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.attn_output = nn.Linear(self.width, self.width)
        self.norm1 = nn.LayerNorm(self.width)
        self.ffn = nn.Sequential(nn.Linear(self.width, 2 * self.width), nn.GELU(),
                                 nn.Linear(2 * self.width, self.width))
        self.norm2 = nn.LayerNorm(self.width)
        self.edge_head = nn.Sequential(
            nn.Linear(3 * self.width + 1, self.width), nn.GELU(),
            nn.LayerNorm(self.width), nn.Linear(self.width, 1),
        )

    def forward(self, visual_features, positions, valid):
        if positions is None or valid is None:
            raise ValueError('Topology recovery requires raw XYZ positions and node validity')
        valid = valid.bool().clone()
        valid[:, 0] = False  # STOP never represents a physical node.
        weights = valid.unsqueeze(-1).float()
        with autocast(enabled=False):
            xyz = positions.float().masked_fill(~valid.unsqueeze(-1), 0)
            center = (xyz * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True).clamp_min(1)
            xyz = ((xyz - center) / 30.0).masked_fill(~valid.unsqueeze(-1), 0)
            euclidean = (xyz.unsqueeze(2) - xyz.unsqueeze(1)).norm(dim=-1)
        x = self.input_projection(visual_features.masked_fill(~valid.unsqueeze(-1), 0))
        x = x + self.position_projection(xyz.to(visual_features.dtype))
        batch, count, _ = x.shape
        q, k, v = self.qkv(x).view(
            batch, count, 3, self.heads, self.width // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        # Do softmax and attention score arithmetic in FP32 under AMP.
        with autocast(enabled=False):
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
            scores = scores / math.sqrt(self.width // self.heads)
        # STOP is a harmless fallback key even for a graph with no valid nodes.
        key_valid = valid.bool().clone()
        key_valid[:, 0] = True
        scores = scores.masked_fill(~key_valid[:, None, None, :], -float('inf'))
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(attention, v).transpose(1, 2).reshape(batch, count, self.width)
        x = self.norm1(x + self.attn_output(out))
        x = self.norm2(x + self.ffn(x))
        src, dst = x.unsqueeze(2), x.unsqueeze(1)
        pairs = torch.cat([src + dst, (src - dst).abs(), src * dst,
                           euclidean.unsqueeze(-1).to(x.dtype)], dim=-1)
        edge_logits = self.edge_head(pairs).squeeze(-1)
        with autocast(enabled=False):
            # Continuous relaxation: unlikely edges become expensive. Shortest
            # paths, rather than a separate distance regressor, produce distances.
            cost = euclidean.clamp_min(1e-6) / edge_logits.float().sigmoid().clamp_min(1e-4)
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
            diagonal = torch.eye(count, dtype=torch.bool, device=x.device).unsqueeze(0)
            cost = cost.masked_fill(~pair_valid, float('inf')).masked_fill(diagonal, 0)
            distance = shortest_path_distances(cost).masked_fill(~pair_valid, 0)
        return {'edge_logits': edge_logits, 'distance': distance}


class MTRPredictionHead(nn.Module):
    """Recover masked temporal relations before ST-GASA graph reasoning."""

    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)
        relation_hidden = int(getattr(config, "mtr_relation_hidden_size", 128))
        self.initializer_range = float(getattr(config, "initializer_range", 0.02))
        self.max_time_bucket = int(getattr(config, "stgasa_time_max_bucket", 6))
        self.pair_encoder = nn.Sequential(
            nn.Linear(hidden_size * 4, relation_hidden),
            nn.GELU(),
            nn.LayerNorm(relation_hidden),
        )
        self.instruction_projection = nn.Sequential(
            nn.Linear(hidden_size, relation_hidden),
            nn.GELU(),
        )
        self.fusion_norm = nn.LayerNorm(relation_hidden)
        num_classes = 2 * self.max_time_bucket + 1
        self.birth_head = nn.Linear(relation_hidden, num_classes)
        self.observation_head = nn.Linear(relation_hidden, num_classes)
        self.spatial_recovery = bool(getattr(config, 'mtr_spatial_recovery', False))
        if self.spatial_recovery:
            self.topology = VisualTopologyRecovery(config)
            # Negative normalized distance/hop values mark missing position channels.
            self.position_mask_value = nn.Parameter(torch.full((2,), -1.0))

    def mask_position_features(self, position, pair_mask, node_types):
        """Remove current-to-node target duplicates, not entire node features."""
        if not self.spatial_recovery or pair_mask is None:
            return position
        if node_types is None or position.size(-1) < 7:
            raise ValueError('Spatial MTR requires node types and 7-D position features')
        hidden = (pair_mask.bool() & (node_types == CURRENT_TYPE_ID).unsqueeze(2)).any(dim=1)
        result = position.clone()
        result[..., -2:] = torch.where(
            hidden.unsqueeze(-1), self.position_mask_value.to(position.dtype), position[..., -2:]
        )
        return result

    @staticmethod
    def restore_distances(distances, prediction, pair_mask):
        return torch.where(pair_mask.bool(), prediction.to(distances.dtype), distances)

    @staticmethod
    def distance_loss(prediction, target, pair_mask):
        """Mean over unordered pairs per sample, then over samples with targets."""
        mask = torch.triu(pair_mask.bool(), diagonal=1)
        counts = mask.sum(dim=(1, 2))
        prediction = prediction.float().masked_fill(~mask, 0)
        target = target.detach().float().masked_fill(~mask, 0)
        error = prediction - target
        loss = F.smooth_l1_loss(prediction, target, reduction='none')
        weight = mask.to(loss.dtype)
        valid = (counts > 0).to(loss.dtype)
        denom = counts.clamp_min(1).to(loss.dtype)
        average = lambda values: (((values * weight).sum(dim=(1, 2)) / denom) * valid).sum() / valid.sum().clamp_min(1)
        return average(loss), average(error.abs()).detach()

    def topology_loss(self, logits, adjacency, positions, valid, pair_mask):
        """Teacher topology is consulted only here, after prediction.

        Edge BCE is class-balanced and sample-balanced; distance supervision uses
        the same snapshot graph (never full-environment hidden connectivity).
        """
        if adjacency is None or positions is None:
            raise ValueError('Spatial MTR loss requires snapshot adjacency and XYZ targets')
        valid = valid.bool().clone()
        valid[:, 0] = False
        pairs = torch.triu(valid.unsqueeze(2) & valid.unsqueeze(1), diagonal=1)
        target = adjacency.bool()
        scores = logits['edge_logits'].float()
        bce = F.binary_cross_entropy_with_logits(scores, target.float(), reduction='none')
        positive, negative = pairs & target, pairs & ~target
        means, accuracies, active = [], [], []
        for mask, correct in ((positive, scores >= 0), (negative, scores < 0)):
            counts = mask.sum((1, 2))
            means.append(bce.masked_fill(~mask, 0).sum((1, 2)) / counts.clamp_min(1))
            accuracies.append((correct & mask).sum((1, 2)).float() / counts.clamp_min(1))
            active.append((counts > 0).float())
        classes = (active[0] + active[1]).clamp_min(1)
        samples = pairs.any(dim=2).any(dim=1).float()
        average = lambda values: ((values / classes) * samples).sum() / samples.sum().clamp_min(1)
        edge_loss = average(means[0] + means[1])
        edge_acc = average(accuracies[0] + accuracies[1]).detach()
        with torch.no_grad(), autocast(enabled=False):
            lengths = (positions.float().unsqueeze(2) - positions.float().unsqueeze(1)).norm(dim=-1) / 30.0
            connected = target & valid.unsqueeze(2) & valid.unsqueeze(1)
            diagonal = torch.eye(scores.size(1), dtype=torch.bool, device=scores.device).unsqueeze(0)
            target_distances = shortest_path_distances(
                lengths.masked_fill(~connected, float('inf')).masked_fill(diagonal, 0)
            )
            distance_mask = pair_mask.bool() & torch.isfinite(target_distances) & (pairs | pairs.transpose(1, 2))
        distance_loss, distance_mae = self.distance_loss(logits['distance'], target_distances, distance_mask)
        predicted_edges = (scores >= 0) & pairs
        true_positive = (predicted_edges & target).sum().float()
        return {
            'mtr_edge_loss': edge_loss, 'mtr_edge_balanced_acc': edge_acc,
            'mtr_edge_precision': (true_positive / predicted_edges.sum().clamp_min(1)).detach(),
            'mtr_edge_recall': (true_positive / positive.sum().clamp_min(1)).detach(),
            'mtr_distance_loss': distance_loss, 'mtr_distance_mae': distance_mae,
        }

    def reset_parameters(self):
        """Reset the head when loading a semantically incompatible old MTR."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight, mean=0.0, std=self.initializer_range
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _pool_instruction(txt_embeds, txt_masks):
        weights = txt_masks.bool().unsqueeze(-1).to(txt_embeds.dtype)
        return (txt_embeds * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    def relation_logits(self, node_embeds, txt_embeds=None, txt_masks=None):
        src = node_embeds.unsqueeze(2).expand(-1, -1, node_embeds.size(1), -1)
        dst = node_embeds.unsqueeze(1).expand(-1, node_embeds.size(1), -1, -1)
        pair = self.pair_encoder(
            torch.cat([src, dst, src * dst, src - dst], dim=-1)
        )
        if txt_embeds is not None and txt_masks is not None:
            instruction = self.instruction_projection(
                self._pool_instruction(txt_embeds, txt_masks)
            )
            pair = self.fusion_norm(pair + instruction[:, None, None, :])
        logits = {
            "birth": self.birth_head(pair),
            "observation": self.observation_head(pair),
        }
        return logits

    @staticmethod
    def relation_probabilities(logits):
        return {
            name: F.softmax(scores, dim=-1)
            for name, scores in logits.items() if name in ('birth', 'observation')
        }

    def labels(self, birth_steps, observation_steps):
        return {
            "birth": signed_log_bucket(
                birth_steps.unsqueeze(2) - birth_steps.unsqueeze(1),
                self.max_time_bucket,
            ),
            "observation": signed_log_bucket(
                observation_steps.unsqueeze(2) - observation_steps.unsqueeze(1),
                self.max_time_bucket,
            ),
        }

    @staticmethod
    def compute_loss(logits, labels, pair_mask, sample_balanced=False):
        losses = {}
        accuracies = {}
        if not bool(pair_mask.any()):
            zero = next(iter(logits.values())).sum() * 0.0
            for name in ('birth', 'observation'):
                losses[name] = zero
                accuracies[name] = zero.detach()
            return losses, accuracies

        for name in ('birth', 'observation'):
            scores = logits[name]
            if sample_balanced:
                counts = pair_mask.sum(dim=(1, 2))
                valid = (counts > 0).to(scores.dtype)
                per_pair = F.cross_entropy(scores.float().permute(0, 3, 1, 2),
                                           labels[name], reduction='none')
                weights = pair_mask.to(per_pair.dtype)
                losses[name] = ((per_pair * weights).sum(dim=(1, 2)) /
                                counts.clamp_min(1) * valid).sum() / valid.sum().clamp_min(1)
                correct = (scores.argmax(dim=-1) == labels[name]).float()
                accuracies[name] = (((correct * weights).sum(dim=(1, 2)) /
                                     counts.clamp_min(1) * valid).sum() /
                                    valid.sum().clamp_min(1)).detach()
                continue
            selected_scores = scores[pair_mask]
            selected_labels = labels[name][pair_mask]
            losses[name] = F.cross_entropy(selected_scores, selected_labels)
            accuracies[name] = (
                selected_scores.argmax(dim=-1) == selected_labels
            ).float().mean().detach()
        return losses, accuracies

    def forward(
        self, node_embeds, pair_mask, birth_steps, observation_steps,
        txt_embeds=None, txt_masks=None,
    ):
        logits = self.relation_logits(
            node_embeds, txt_embeds, txt_masks,
        )
        labels = self.labels(birth_steps, observation_steps)
        losses, accuracies = self.compute_loss(
            logits, labels, pair_mask.bool(), sample_balanced=self.spatial_recovery,
        )
        return logits, labels, losses, accuracies
