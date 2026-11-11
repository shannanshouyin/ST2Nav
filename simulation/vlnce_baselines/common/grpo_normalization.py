"""Per-trajectory weights for the GRPO policy and KL objectives."""

import torch


def trajectory_step_weights(step_indices, num_trajectories, device=None):
    """Return weights 1 / (B * T_i) aligned with each cached step's rows.

    ``step_indices[t]`` contains original environment slots, not their current
    positions after environments are paused. Every cached action counts once,
    including STOP and the final action at a rollout time limit. No padding is
    counted. Summing weighted per-action losses yields a mean of trajectory
    means. The caller separately averages over the G samples in each group.
    """
    if num_trajectories < 1:
        raise ValueError('GRPO requires at least one trajectory')
    lengths = [0] * num_trajectories
    for indices in step_indices:
        if len(set(indices)) != len(indices):
            raise ValueError('Duplicate environment slot in a GRPO step')
        for index in indices:
            if index < 0 or index >= num_trajectories:
                raise ValueError('Invalid original environment slot in GRPO buffer')
            lengths[index] += 1
    if any(length == 0 for length in lengths):
        raise ValueError('GRPO trajectory has no cached actions')
    return [
        torch.tensor(
            [1.0 / (num_trajectories * lengths[index]) for index in indices],
            dtype=torch.float32, device=device,
        )
        for indices in step_indices
    ]
