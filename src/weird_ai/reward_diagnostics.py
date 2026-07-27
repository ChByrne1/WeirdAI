"""
Reward diagnostics utilities for Weird AI.

Chapter 7 focuses on improving GRPO by tracking training metrics, detecting
instability, preventing reward exploitation, and adding format rewards.

This module does not train a model. It gives students tools to inspect whether
the reward signal from Weird AI is healthy enough to use in future training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Iterable, Optional


@dataclass
class DiagnosticWarning:
    """A warning produced by reward diagnostics."""

    code: str
    message: str
    severity: str = "warning"
    details: dict = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Summary of reward and output diagnostics."""

    reward_count: int
    reward_average: float
    reward_min: float
    reward_max: float
    reward_std: float
    advantage_average: float
    advantage_std: float
    warnings: list[DiagnosticWarning] = field(default_factory=list)


def moving_average(values: Iterable[float], window_size: int = 3) -> list[float]:
    """
    Compute a simple moving average.

    Example:
        [1, 2, 3, 4] with window_size=2 -> [1.0, 1.5, 2.5, 3.5]
    """

    # TODO:
    # 1. Convert values to a list.
    # 2. Validate that window_size is at least 1.
    # 3. For each position, average up to window_size recent values.
    # 4. Return the list of averages.

    raise NotImplementedError("Implement moving_average.")


def summarize_values(values: Iterable[float]) -> dict:
    """
    Compute count, average, min, max, and population standard deviation.

    Empty input should return count=0 and zeros for the numeric fields.
    """

    # TODO:
    # 1. Convert values to a list.
    # 2. Handle empty input.
    # 3. Compute count, average, min, max, and std.
    # 4. Return these values in a dictionary.

    raise NotImplementedError("Implement summarize_values.")


def advantage_statistics(advantages: Iterable[float]) -> dict:
    """Summarize advantage average and standard deviation."""

    # TODO:
    # 1. Reuse summarize_values.
    # 2. Return at least average and std.

    raise NotImplementedError("Implement advantage_statistics.")


def detect_reward_collapse(
    rewards: Iterable[float],
    *,
    min_std: float = 0.01,
) -> Optional[DiagnosticWarning]:
    """
    Detect whether rewards have collapsed to nearly identical values.

    Reward collapse matters because GRPO-style learning depends on differences
    within a group. If all rewards are identical, the advantage signal disappears.
    """

    # TODO:
    # 1. Return None if there are fewer than 2 rewards.
    # 2. Compute reward standard deviation.
    # 3. Return a DiagnosticWarning when std is below min_std.
    # 4. Otherwise return None.

    raise NotImplementedError("Implement detect_reward_collapse.")


def get_nonempty_lines(text: str) -> list[str]:
    """Return stripped non-empty lines."""

    # TODO:
    # 1. Split text into lines.
    # 2. Strip whitespace.
    # 3. Keep only non-empty lines.

    raise NotImplementedError("Implement get_nonempty_lines.")


def repeated_line_ratio(text: str) -> float:
    """
    Compute the fraction of non-empty lines that are repeated beyond first use.
    """

    # TODO:
    # 1. Get non-empty lines.
    # 2. Normalize lines for comparison.
    # 3. Count repeats beyond their first occurrence.
    # 4. Return repeated_count / total_lines.

    raise NotImplementedError("Implement repeated_line_ratio.")


def repeated_word_ratio(text: str) -> float:
    """
    Compute the fraction of words that are repeated beyond first use.

    This is a simple heuristic. Lowercase words and strip basic punctuation.
    """

    # TODO:
    # 1. Split text into words.
    # 2. Normalize words by lowercasing and stripping punctuation.
    # 3. Ignore empty tokens.
    # 4. Count repeats beyond their first occurrence.
    # 5. Return repeated_count / total_words.

    raise NotImplementedError("Implement repeated_word_ratio.")


def detect_short_high_reward(
    text: str,
    reward: float,
    *,
    min_lines: int = 4,
    high_reward: float = 0.85,
) -> Optional[DiagnosticWarning]:
    """Detect suspiciously short outputs with high reward."""

    # TODO:
    # 1. Count non-empty lines.
    # 2. If reward is high but line count is below min_lines, return a warning.
    # 3. Otherwise return None.

    raise NotImplementedError("Implement detect_short_high_reward.")


def detect_repetition_hacking(
    text: str,
    *,
    max_repeated_line_ratio: float = 0.25,
    max_repeated_word_ratio: float = 0.40,
) -> list[DiagnosticWarning]:
    """Detect possible repetition-based reward hacking."""

    # TODO:
    # 1. Compute repeated line ratio.
    # 2. Compute repeated word ratio.
    # 3. Add warnings when ratios exceed thresholds.
    # 4. Return the warnings.

    raise NotImplementedError("Implement detect_repetition_hacking.")


def format_reward(
    text: str,
    *,
    target_line_count: int = 8,
    line_tolerance: int = 2,
    min_line_length: int = 5,
    max_line_length: int = 100,
    explanation_markers: Optional[list[str]] = None,
) -> float:
    """
    Compute a simple format reward for lyric output.

    Suggested scoring:
    - 0.25 for non-empty output
    - 0.25 for line count within tolerance
    - 0.25 for most lines within length bounds
    - 0.25 for no explanation markers
    """

    # TODO:
    # 1. Get non-empty lines.
    # 2. Award partial credit using the suggested scoring above.
    # 3. Use default explanation markers such as "here are", "explanation:", and "revised lyrics:".
    # 4. Return a value clamped to [0.0, 1.0].

    raise NotImplementedError("Implement format_reward.")


def combine_rewards(
    creative_reward: float,
    format_reward_value: float,
    *,
    creative_weight: float = 0.75,
    format_weight: float = 0.25,
) -> float:
    """Combine creative reward and format reward as a weighted average."""

    # TODO:
    # 1. Reject negative weights.
    # 2. Return 0.0 if both weights are zero.
    # 3. Compute weighted average.
    # 4. Clamp the result to [0.0, 1.0].

    raise NotImplementedError("Implement combine_rewards.")


def diagnose_rollout(
    text: str,
    reward: float,
    *,
    min_lines: int = 4,
    high_reward: float = 0.85,
) -> list[DiagnosticWarning]:
    """Run output-level diagnostics for one generated rollout."""

    # TODO:
    # 1. Check for short high-reward output.
    # 2. Check for repetition hacking.
    # 3. Return all warnings.

    raise NotImplementedError("Implement diagnose_rollout.")


def build_diagnostic_report(
    rewards: Iterable[float],
    advantages: Iterable[float],
    texts: Optional[list[str]] = None,
) -> DiagnosticReport:
    """Build a diagnostic report for rewards, advantages, and optional rollout texts."""

    # TODO:
    # 1. Summarize rewards.
    # 2. Summarize advantages.
    # 3. Add reward collapse warning if needed.
    # 4. If texts are provided, run rollout diagnostics for each text/reward pair.
    # 5. Return DiagnosticReport.

    raise NotImplementedError("Implement build_diagnostic_report.")


def summarize_diagnostic_report(report: DiagnosticReport) -> str:
    """Convert a DiagnosticReport into readable text."""

    # TODO:
    # 1. Include reward count, average, min, max, and std.
    # 2. Include advantage average and std.
    # 3. Include all warnings.
    # 4. If there are no warnings, say so.

    raise NotImplementedError("Implement summarize_diagnostic_report.")
