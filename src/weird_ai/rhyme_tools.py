"""
Rhyme and syllable helper functions for Weird AI.

This module intentionally hides most of the pronunciation-dictionary complexity
so students can focus on model generation, evaluation, and refinement.
"""

import re
from typing import List, Optional

import pronouncing


def clean_word(word: str) -> str:
    """
    Normalize a word for pronunciation lookup.

    Args:
        word: A raw word, possibly with punctuation.

    Returns:
        A lowercase alphabetic word.
    """
    return re.sub(r"[^a-zA-Z']", "", word).lower()


def get_last_word(line: str) -> Optional[str]:
    """
    Get the final meaningful word from a line of lyrics.

    Args:
        line: A line of text.

    Returns:
        The final cleaned word, or None if no valid word exists.
    """
    words = [clean_word(word) for word in line.split()]
    words = [word for word in words if word]

    if not words:
        return None

    return words[-1]


def get_pronunciations(word: str) -> List[str]:
    """
    Look up pronunciations for a word.

    Args:
        word: The word to look up.

    Returns:
        A list of pronunciation strings.
    """
    cleaned = clean_word(word)

    if not cleaned:
        return []

    return pronouncing.phones_for_word(cleaned)


def count_syllables_in_word(word: str) -> int:
    """
    Estimate syllable count for a word.

    Args:
        word: The word to count.

    Returns:
        Estimated syllable count. Returns 1 as a fallback for unknown words.
    """
    pronunciations = get_pronunciations(word)

    if pronunciations:
        return pronouncing.syllable_count(pronunciations[0])

    # Simple fallback heuristic for unknown words.
    cleaned = clean_word(word)
    groups = re.findall(r"[aeiouy]+", cleaned)

    return max(1, len(groups))


def count_syllables_in_line(line: str) -> int:
    """
    Estimate the total syllable count for a line.

    Args:
        line: A line of lyrics.

    Returns:
        Estimated syllable count.
    """
    words = [clean_word(word) for word in line.split()]
    words = [word for word in words if word]

    return sum(count_syllables_in_word(word) for word in words)


def get_rhyming_part(word: str) -> Optional[str]:
    """
    Get the rhyming part of a word using its pronunciation.

    Args:
        word: The word to analyze.

    Returns:
        A rhyming-part string, or None if unavailable.
    """
    pronunciations = get_pronunciations(word)

    if not pronunciations:
        return None

    return pronouncing.rhyming_part(pronunciations[0])


def do_words_rhyme(word1: str, word2: str) -> bool:
    """
    Determine whether two words rhyme.

    Args:
        word1: First word.
        word2: Second word.

    Returns:
        True if the words appear to rhyme, otherwise False.
    """
    cleaned1 = clean_word(word1)
    cleaned2 = clean_word(word2)

    if not cleaned1 or not cleaned2:
        return False

    if cleaned1 == cleaned2:
        return True

    rhyming_part1 = get_rhyming_part(cleaned1)
    rhyming_part2 = get_rhyming_part(cleaned2)

    if rhyming_part1 and rhyming_part2:
        return rhyming_part1 == rhyming_part2

    rhymes = pronouncing.rhymes(cleaned1)

    return cleaned2 in rhymes


def do_lines_rhyme(line1: str, line2: str) -> bool:
    """
    Determine whether two lyric lines rhyme based on their final words.

    Args:
        line1: First lyric line.
        line2: Second lyric line.

    Returns:
        True if the final words rhyme, otherwise False.
    """
    word1 = get_last_word(line1)
    word2 = get_last_word(line2)

    if word1 is None or word2 is None:
        return False

    return do_words_rhyme(word1, word2)


def get_rhyme_label(line: str) -> Optional[str]:
    """
    Get a rhyme label key for a line based on its final word.

    This can be used to build simple rhyme schemes.

    Args:
        line: A line of lyrics.

    Returns:
        A rhyme key, or None if no key is available.
    """
    word = get_last_word(line)

    if word is None:
        return None

    rhyming_part = get_rhyming_part(word)

    if rhyming_part:
        return rhyming_part

    return word[-3:] if len(word) >= 3 else word


def get_rhyme_scheme(lines: List[str]) -> List[str]:
    """
    Estimate a rhyme scheme for a group of lines.

    Example:
        ["I saw the cat", "It wore a hat"] -> ["A", "A"]

    Args:
        lines: Lyric lines.

    Returns:
        A list of rhyme scheme labels such as ["A", "B", "A", "B"].
    """
    rhyme_keys = []
    scheme = []
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for line in lines:
        key = get_rhyme_label(line)

        if key is None:
            scheme.append("?")
            continue

        if key not in rhyme_keys:
            rhyme_keys.append(key)

        index = rhyme_keys.index(key)

        if index < len(labels):
            scheme.append(labels[index])
        else:
            scheme.append(f"R{index + 1}")

    return scheme


def calculate_pairwise_rhyme_score(lines: List[str]) -> float:
    """
    Calculate how often adjacent lines rhyme.

    Args:
        lines: Lyric lines.

    Returns:
        A score from 0.0 to 1.0.
    """
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    if len(cleaned_lines) < 2:
        return 0.0

    comparisons = 0
    matches = 0

    for index in range(len(cleaned_lines) - 1):
        comparisons += 1

        if do_lines_rhyme(cleaned_lines[index], cleaned_lines[index + 1]):
            matches += 1

    return matches / comparisons


def calculate_syllable_consistency_score(lines: List[str]) -> float:
    """
    Score how consistent line syllable counts are.

    A score near 1.0 means the lines have similar syllable counts.
    A score near 0.0 means the lines vary widely.

    Args:
        lines: Lyric lines.

    Returns:
        A score from 0.0 to 1.0.
    """
    cleaned_lines = [line.strip() for line in lines if line.strip()]

    if len(cleaned_lines) < 2:
        return 1.0

    counts = [count_syllables_in_line(line) for line in cleaned_lines]
    average = sum(counts) / len(counts)

    if average == 0:
        return 0.0

    average_deviation = sum(abs(count - average) for count in counts) / len(counts)

    score = 1.0 - min(1.0, average_deviation / average)

    return score


def analyze_lyrics(text: str) -> dict:
    """
    Analyze lyrics for rhyme and syllable information.

    Args:
        text: Full lyric text.

    Returns:
        A dictionary containing line-level analysis.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    syllable_counts = [count_syllables_in_line(line) for line in lines]
    rhyme_scheme = get_rhyme_scheme(lines)
    rhyme_score = calculate_pairwise_rhyme_score(lines)
    syllable_score = calculate_syllable_consistency_score(lines)

    return {
        "line_count": len(lines),
        "lines": lines,
        "syllable_counts": syllable_counts,
        "rhyme_scheme": rhyme_scheme,
        "pairwise_rhyme_score": rhyme_score,
        "syllable_consistency_score": syllable_score,
    }
