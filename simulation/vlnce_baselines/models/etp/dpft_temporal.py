"""Node-relative time residual for the existing DPFT query (not a second branch)."""
import torch
from torch import nn

from .stgasa import nonnegative_log_bucket


class NodeTemporalQuery(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Reuse bucket/width configuration, never share relation embedding weights.
        self.max_bucket = int(getattr(config, 'stgasa_time_max_bucket', 6))
        width = int(getattr(config, 'stgasa_relation_hidden_size', 128))
        if self.max_bucket < 0 or width <= 0:
            raise ValueError('DPFT temporal requires nonnegative time_max_bucket and positive relation_hidden_size')
        self.birth_embedding = nn.Embedding(self.max_bucket + 1, width)
        self.observation_embedding = nn.Embedding(self.max_bucket + 1, width)
        self.projection = nn.Linear(2 * width, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_eps', 1e-12))
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, query, birth_steps, observation_steps, valid_mask):
        if birth_steps is None or observation_steps is None or valid_mask is None:
            raise ValueError('MODEL.dpft_temporal requires birth/observation steps and temporal validity')
        if any(x.shape != query.shape[:2] for x in (birth_steps, observation_steps, valid_mask)):
            raise ValueError('DPFT temporal states must have shape [batch, nodes]')
        valid_mask = valid_mask.bool()
        # GraphMap observes the current node on every high-level step. Therefore
        # the largest valid last-observation step is exactly the decision step,
        # even when legacy gmap_step_ids use inherited birth rather than arrival.
        current_step = observation_steps.masked_fill(~valid_mask, 0).max(dim=1, keepdim=True)[0]
        birth_ids = nonnegative_log_bucket(current_step - birth_steps, self.max_bucket)
        observation_ids = nonnegative_log_bucket(current_step - observation_steps, self.max_bucket)
        temporal = self.norm(self.projection(torch.cat([
            self.birth_embedding(birth_ids), self.observation_embedding(observation_ids),
        ], dim=-1)))
        temporal = temporal.to(dtype=query.dtype).masked_fill(~valid_mask.unsqueeze(-1), 0)
        # Normalize only the residual: alpha=0 leaves the original query intact.
        # Do not short-circuit at zero: alpha must receive navigation gradients.
        return query + torch.tanh(self.alpha).to(dtype=query.dtype) * temporal
