from collections.abc import Callable

import torch
from einops import rearrange
from typing import Literal


def calculate_groupwise_rewards(
    reward_function: Callable[[str, str], dict[str, float]],
    responses: list[str],
    ground_truths: list[str],
    group_size: int,
    epsilon: float,
    use_std_normalization: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards_list = [
        reward_function(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]
    raw_rewards = torch.tensor([reward["reward"] for reward in rewards_list])
    reward_groups = rearrange(
        raw_rewards,
        "(n_groups group_size) -> n_groups group_size",
        group_size=group_size,
    )
    group_means = reward_groups.mean(dim=-1)
    group_stds = reward_groups.std(dim=-1)
    divisors = (
        group_stds + epsilon if use_std_normalization else torch.ones_like(group_stds)
    )
    group_advantages = (reward_groups - group_means.unsqueeze(-1)) / divisors.unsqueeze(-1)
    advantages = rearrange(
        group_advantages, "n_groups group_size -> (n_groups group_size)"
    )
    return advantages, raw_rewards, {"rewards_list": rewards_list}


def basic_policy_gradient_loss(
    rewards_or_advantages: torch.Tensor, log_probs: torch.Tensor
) -> torch.Tensor:
    return -log_probs * rewards_or_advantages


def clipped_grpo_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    previous_log_probs: torch.Tensor,
    clip_range: float,
    apply_clip: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratios = torch.exp(log_probs - previous_log_probs)
    scores = ratios * advantages
    mean_ratio = ratios.mean()
    if not apply_clip:
        return -scores, {"clip_fraction": 0.0, "mean_ratio": mean_ratio}
    clipped_ratios = torch.clip(ratios, 1 - clip_range, 1 + clip_range)
    clipped_scores = clipped_ratios * advantages
    clip_fraction = (~torch.isclose(scores, clipped_scores)).float().mean()
    return -torch.minimum(scores, clipped_scores), {
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
    }


def calculate_policy_gradient_loss(
    log_probs: torch.Tensor,
    loss_variant: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"
    ],
    rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    previous_log_probs: torch.Tensor | None = None,
    clip_range: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert loss_variant in (
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        "grpo_no_clip",
    )
    if loss_variant == "no_baseline":
        assert rewards is not None
        return basic_policy_gradient_loss(rewards, log_probs), {}
    assert advantages is not None
    if loss_variant == "reinforce_with_baseline":
        return basic_policy_gradient_loss(advantages, log_probs), {}
    if loss_variant == "grpo_clip":
        assert previous_log_probs is not None
        assert clip_range is not None
        return clipped_grpo_loss(
            advantages, log_probs, previous_log_probs, clip_range
        )
    if loss_variant == "grpo_no_clip":
        assert previous_log_probs is not None
        return clipped_grpo_loss(
            advantages, log_probs, previous_log_probs, clip_range, apply_clip=False
        )


def mean_with_mask(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim) / mask.sum(dim)


def tokenize_prompt_and_response(
    prompts: list[str],
    responses: list[str],
    tokenizer: Tokenizer,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    assert len(prompts) == len(responses)
    prompt_tokens_list = []
    response_tokens_list = []
    concatenated_tokens_list = []
    for prompt, response in zip(prompts, responses):
        prompt_tokens = tokenizer.encode(prompt)
        response_tokens = tokenizer.encode(response)
        prompt_tokens_list.append(prompt_tokens)
        response_tokens_list.append(response_tokens)
        concatenated_tokens_list.append(prompt_tokens + response_tokens)
    max_len = max(len(t) for t in concatenated_tokens_list)
    input_ids = torch.full(
        (len(concatenated_tokens_list), max_len - 1), tokenizer.pad_token_id
    )
    labels = torch.full(
        (len(concatenated_tokens_list), max_len - 1), tokenizer.pad_token_id
    )
    response_mask = torch.zeros(
        (len(concatenated_tokens_list), max_len - 1), dtype=torch.bool
    )
    for i, tokens in enumerate(concatenated_tokens_list):
        n_tokens = min(len(tokens), max_len - 1)
        prompt_tokens = prompt_tokens_list[i]
        input_ids[i, :n_tokens] = torch.tensor(tokens[:n_tokens])
        labels[i, : len(tokens) - 1] = torch.tensor(tokens[1:])
        response_mask[i, len(prompt_tokens) - 1 : len(tokens) - 1] = True
    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        response_mask = response_mask.to(device)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)


def extract_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    all_log_probs = torch.log_softmax(logits, dim=-1)
    labels_idx = rearrange(labels, "b t -> b t 1")
    log_probs_of_labels = torch.gather(all_log_probs, dim=-1, index=labels_idx)
    log_probs_of_labels = log_probs_of_labels.squeeze(dim=-1)
    result = {"log_probs": log_probs_of_labels}
    if return_token_entropy:
        result["token_entropy"] = calculate_entropy(logits)
    return result


def normalize_with_mask(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalization_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked = tensor * mask
    return masked.sum(dim) / normalization_constant


def record_generations(
    vllm_model: LLM,
    hf_model: PreTrainedModel,
    tokenizer: Tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_function: Callable[[str, str], dict[str, float]] | None = None,
    sampling_parameters: SamplingParams | None = None,
    output_directory: str | None = "out",
    output_file: str | None = None,
) -> list[EvaluationResult]:
    if sampling_parameters is None:
        sampling_parameters = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
    outputs = vllm_model.generate(prompts, sampling_parameters)
    os.makedirs(output_directory, exist_ok=True)
    filename = output_file or f"{generate_ordered_filename('log_generations')}.jsonl"
    outpath = os.path.join(output_directory, filename)
    results = []
    total_response_length = 0
    correct_response_lengths = []
    incorrect_response_lengths = []
    total_entropy = 0
    n_tokens_total = 0
    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text
        ground_truth = ground_truths[i]
        rewards = {}
        if reward_function is not None:
            rewards = reward_function(completion, ground_truth)
            is_correct = rewards.get("reward", 0) > 0
        else:
            is_correct = False
        prompt_output_dict = tokenize_prompt_and_response(
            [prompt],
            [completion],
            tokenizer,
            device=hf_model.device if hasattr(hf_model, "device") else None,
        )
        input_ids = prompt_output_dict["input_ids"]
        labels = prompt_output_dict["labels"]
        response_mask = prompt_output_dict["response_mask"]
        with torch.no_grad():
            response_info = extract_response_log_probs(
                hf_model, input_ids, labels, return_token_entropy=True
            )
        token_entropy = response_info["token_entropy"]
        masked_entropy = token_entropy * response_mask
        total_entropy += masked_entropy.sum().item()
        n_response_tokens = response_mask.sum().item()
        response_length = n_response_tokens
        total_response_length += response_length
        if is_correct:
            correct_response_lengths.append(response_length)
        else:
            incorrect_response_lengths.append(response_length)
        n_tokens_total += n_response_tokens
        result = EvaluationResult(
            prompt=prompt,
            completion=completion,
            ground_truth=ground_truth,
            rewards=rewards,
        )
        results.append(result)
    n_examples = len(prompts)
    avg_response_length = total_response_length / n_examples if n_examples > 0 else 0
    avg_correct_length = (
        sum(correct_response_lengths) / len(correct_response_lengths)
        if correct_response_lengths
        else 0
    )
    avg_incorrect_length = (
        sum(incorrect_response_lengths) / len(incorrect_response_lengths)
        if incorrect_response_lengths
        else 0
    )
    avg_token_entropy = total_entropy / n_tokens_total if n_tokens_total > 0 else 0
    with open(outpath, "w") as f:
        summary = {
            "n_examples": n_examples,
            "avg_response_length": avg_response_length,
            "avg_correct_response_length": avg_correct_length,
            "avg_incorrect_response_length": avg_incorrect_length,
            "avg_token_entropy": avg_token_entropy,
        }
        f.write(json.dumps(summary) + "\n")
        for result in results:
            f.write(result.model_dump_json() + "\n")
    return results