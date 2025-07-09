from dataclasses import dataclass, field


@dataclass
class DataPathsConfig:
    training_examples_path: str = "/data/a5-alignment/MATH/train.jsonl"
    validation_examples_path: str = "/data/a5-alignment/MATH/validation.jsonl"
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    model_checkpoint_path: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    model_output_dir: str = "/data/c-sniderb/a5-alignment/grpo-experiments"


@dataclass
class ModelTrainingConfig:
    random_seed: int = 42
    tensor_dtype: str = "bfloat16"
    device_name: str = "cuda:0"
    vllm_device_name: str = "cuda:0"
    num_grpo_steps: int = 200
    learning_rate: float = 3e-5
    advantage_epsilon: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    epochs_per_rollout_batch: int = 1
    training_batch_size: int = 256
    grad_accumulation_steps: int = 128
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    sampling_top_p: float = 1.0
    gpu_memory_utilization: float = 0.2
    loss_variant: str = "reinforce_with_baseline"
    use_standard_normalization: bool = True
    normalization_mode: str = "constant"
    max_gradient_norm: float = 1.0
    clip_range: float = 0.2
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    wandb_entity_name: str = "brandon-snider-stanford-university"
    wandb_project_name: str = "cs336-a5"
    torch_compile_enabled: bool = True
    log_step_interval: int = 1
    wandb_tags: list[str] = field(default_factory=lambda: ["grpo"])
    evaluate_before_training: bool = False
    evaluation_step_interval: int = 20
    evaluation_num_examples: int = 1024
    old_log_probs_batch_size: int = 8
    use_question_only: bool = False


@dataclass
class ExperimentConfig:
    paths: DataPathsConfig = field(default_factory=DataPathsConfig)
    training: ModelTrainingConfig = field(default_factory=ModelTrainingConfig)
