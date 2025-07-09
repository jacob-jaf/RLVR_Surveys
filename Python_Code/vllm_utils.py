import os
import torch
from unittest.mock import patch
from transformers import PreTrainedModel
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from filename_utils import generate_ordered_filename
from collections.abc import Callable
from vllm import LLM, SamplingParams
from pydantic import BaseModel


class EvaluationMetrics(BaseModel):
    num_examples: int
    num_format_correct: int
    num_format_incorrect: int
    num_answer_correct: int
    num_answer_incorrect: int
    num_correct: int
    num_incorrect: int
    format_accuracy: float
    answer_accuracy: float
    accuracy: float


class EvaluationResult(BaseModel):
    prompt: str
    completion: str
    ground_truth: str
    rewards: dict[str, float]


def evaluate_with_vllm(
    vllm_model: LLM,
    reward_function: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    sampling_parameters: SamplingParams | None = None,
    output_directory: str | None = "out",
    output_file: str | None = None,
    write: bool = True,
    min_tokens: int = 0,
) -> tuple[list[EvaluationResult], EvaluationMetrics]:
    """
    Eval LM on prompts, compute eval metrics, optionally serialize to disk, return evaluation results.
    """
    sampling_params = sampling_parameters or SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_tokens=min_tokens,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    outputs = vllm_model.generate(
        prompts,
        sampling_params,
    )

    os.makedirs(output_directory, exist_ok=True)
    filename = output_file or f"{generate_ordered_filename('eval')}.jsonl"
    outpath = os.path.join(output_directory, filename)

    results = []

    metrics = EvaluationMetrics(
        num_examples=len(prompts),
        num_format_correct=0,
        num_format_incorrect=0,
        num_answer_correct=0,
        num_answer_incorrect=0,
        num_correct=0,
        num_incorrect=0,
        format_accuracy=0.0,
        answer_accuracy=0.0,
        accuracy=0.0,
    )

    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text
        ground_truth = ground_truths[i]
        rewards = reward_function(completion, ground_truth)

        format_correct = rewards["format_reward"]
        answer_correct = rewards["answer_reward"]
        correct = rewards["reward"]

        metrics.num_format_correct += int(format_correct)
        metrics.num_format_incorrect += int(not format_correct)
        metrics.num_answer_correct += int(answer_correct)
        metrics.num_answer_incorrect += int(not answer_correct)
        metrics.num_correct += int(correct)
        metrics.num_incorrect += int(not correct)

        result = EvaluationResult(
            prompt=prompt,
            completion=completion,
            ground_truth=ground_truth,
            rewards=rewards,
        )

        results.append(result)

    metrics.format_accuracy = metrics.num_format_correct / metrics.num_examples
    metrics.answer_accuracy = metrics.num_answer_correct / metrics.num_examples
    metrics.accuracy = metrics.num_correct / metrics.num_examples

    if write:
        with open(outpath, "w") as f:
            f.write(metrics.model_dump_json() + "\n")
            f.write("\n".join([result.model_dump_json() for result in results]) + "\n")

    return results, metrics


def initialize_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process; here we use vLLM to hold a model on a GPU separate from the policy.
    """
    set_random_seed = vllm_set_random_seed
    set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_to_vllm(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()

    # Handle compiled models by stripping _orig_mod prefix
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {
            key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()
        }

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
