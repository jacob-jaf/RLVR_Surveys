from utils import calculate_policy_gradient_loss, mean_with_mask, normalize_with_mask
import torch
from typing import Literal


def grpo_microbatch_training_step(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    accumulation_steps: int,
    loss_variant: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"
    ],
    rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    previous_log_probs: torch.Tensor | None = None,
    clip_range: float | None = None,
    normalization_mode: Literal["mean", "constant", "microbatch"] = "mean",
    normalization_constant: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_gradient_loss, meta = calculate_policy_gradient_loss(
        log_probs,
        loss_variant,
        rewards,
        advantages,
        previous_log_probs,
        clip_range,
    )

    if normalization_mode == "mean":
        loss = (
            mean_with_mask(policy_gradient_loss, mask, dim=-1).mean()
            / accumulation_steps
        )
    elif normalization_mode in ["constant", "microbatch"]:
        assert normalization_constant is not None

        if normalization_mode == "constant":
            constant = normalization_constant
        elif normalization_mode == "microbatch":
            # Normalize by longest sequence in microbatch
            constant = mask.sum(dim=-1).max().item()

        loss = (
            normalize_with_mask(
                policy_gradient_loss, mask, constant, dim=-1
            ).mean()
            / accumulation_steps
        )

    loss.backward()
    detached_loss = loss.detach()

    # Trying to avoid OOM; haven't ablated this to confirm that it helps
    del loss
    del policy_gradient_loss

    return detached_loss, meta
