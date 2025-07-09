from vllm import LLM, SamplingParams
from tokenizers import Tokenizer
import torch

def convert_prompt_for_training(questions: list[str], answers: list[str], tokenizer_model: Tokenizer) -> dict[str, torch.Tensor]:
    question_tokens = []
    answer_tokens = []
    for question, answer in zip(questions, answers):
        question_tokens.append(tokenizer_model.encode(question))
        answer_tokens.append(tokenizer_model.encode(answer))
