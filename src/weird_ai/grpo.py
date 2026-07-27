"""
Mini-GRPO utilities for Weird AI.

This advanced lesson adds the missing reinforcement learning step to the course.
It is educational, not production GRPO. It omits KL penalties, clipped ratios,
reference models, checkpointing, and distributed rollout collection.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol
import torch

GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]

class PolicyModelLike(Protocol):
    def parameters(self): ...

@dataclass
class Rollout:
    prompt: str
    text: str
    evaluation: dict
    reward: float
    advantage: float
    logprob: float
    index: int

@dataclass
class MiniGRPOResult:
    prompt: str
    rollouts: list[Rollout]
    loss: float
    reward_average: float
    advantage_average: float
    optimizer_step_completed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

def extract_reward(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a reward from an evaluation dictionary. Missing/invalid rewards become 0.0."""
    # TODO: read score_key, convert to float, return 0.0 on failure.
    raise NotImplementedError("Implement extract_reward.")

def compute_group_advantages(rewards: Iterable[float], *, normalize: bool=False, epsilon: float=1e-8) -> list[float]:
    """Compute group-relative advantages: reward - mean(rewards)."""
    # TODO: handle empty input, subtract group mean, optionally divide by population std + epsilon.
    raise NotImplementedError("Implement compute_group_advantages.")

def generate_rollouts(prompt: str, generation_func: GenerationFunction, *, num_rollouts: int=4,
                      temperature: float=1.0, top_p: Optional[float]=None, max_new_tokens: int=120,
                      **kwargs: Any) -> list[str]:
    """Generate multiple rollouts from the same prompt."""
    # TODO: validate num_rollouts >= 1, call generation_func repeatedly, forward sampling args.
    raise NotImplementedError("Implement generate_rollouts.")

def evaluate_rollouts(texts: list[str], evaluation_func: EvaluationFunction, *, score_key: str="overall_score") -> tuple[list[dict], list[float]]:
    """Evaluate rollout texts and extract rewards."""
    # TODO: evaluate each text and extract rewards.
    raise NotImplementedError("Implement evaluate_rollouts.")

def sequence_logprob_from_token_logprobs(token_logprobs: torch.Tensor, *, reduction: str="mean") -> torch.Tensor:
    """Convert token log probabilities into a scalar sequence log probability."""
    # TODO: support reduction='sum' and reduction='mean'; reject others.
    raise NotImplementedError("Implement sequence_logprob_from_token_logprobs.")

def selected_token_logprobs(logits: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities of selected target tokens."""
    # TODO: validate shapes, log_softmax logits, gather selected token logprobs.
    raise NotImplementedError("Implement selected_token_logprobs.")

def compute_policy_gradient_loss(logprobs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Compute educational policy-gradient loss: -(advantages * logprobs).mean()."""
    # TODO: validate matching shapes, move advantages to logprobs dtype/device, compute loss.
    raise NotImplementedError("Implement compute_policy_gradient_loss.")

def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute average entropy from logits."""
    # TODO: probs=softmax, log_probs=log_softmax, entropy=-(probs*log_probs).sum(dim=-1).mean().
    raise NotImplementedError("Implement compute_entropy_from_logits.")

def build_rollout_records(prompt: str, texts: list[str], evaluations: list[dict], rewards: list[float], advantages: list[float], logprobs: list[float]) -> list[Rollout]:
    """Package rollout information into Rollout dataclasses."""
    # TODO: validate list lengths, build Rollout objects indexed from 1.
    raise NotImplementedError("Implement build_rollout_records.")

def mini_grpo_update(*, prompt: str, model: PolicyModelLike, optimizer: torch.optim.Optimizer,
                     generation_func: GenerationFunction, evaluation_func: EvaluationFunction,
                     logprob_func: Callable[[str, str], torch.Tensor], num_rollouts: int=4,
                     score_key: str="overall_score", normalize_advantages: bool=False,
                     temperature: float=1.0, top_p: Optional[float]=None, max_new_tokens: int=120,
                     gradient_clip_norm: Optional[float]=None, **kwargs: Any) -> MiniGRPOResult:
    """Run one educational mini-GRPO update."""
    # TODO: generate, evaluate, compute advantages, compute differentiable logprobs, compute loss,
    # zero_grad/backward/optional clip/optimizer.step, return MiniGRPOResult.
    raise NotImplementedError("Implement mini_grpo_update.")

def summarize_mini_grpo_result(result: MiniGRPOResult) -> str:
    """Create a readable summary of one mini-GRPO update."""
    # TODO: include prompt, loss, avg reward/advantage, rollout details, optimizer status.
    raise NotImplementedError("Implement summarize_mini_grpo_result.")
