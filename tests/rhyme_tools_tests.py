from weird_ai.rhyme_tools import (
    count_syllables_in_line,
    do_words_rhyme,
    do_lines_rhyme,
    get_rhyme_scheme,
    analyze_lyrics,
)


def test_words_rhyme():
    assert do_words_rhyme("cat", "hat")


def test_lines_rhyme():
    assert do_lines_rhyme("I saw a cat", "It wore a hat")


def test_syllable_count():
    count = count_syllables_in_line("I saw a cat")

    assert count > 0


def test_rhyme_scheme():
    lines = [
        "I saw a cat",
        "It wore a hat",
        "I found a dog",
        "It chased a frog",
    ]

    scheme = get_rhyme_scheme(lines)

    assert scheme == ["A", "A", "B", "B"]


def test_analyze_lyrics():
    text = """
    I saw a cat
    It wore a hat
    I found a dog
    It chased a frog
    """

    result = analyze_lyrics(text)

    assert result["line_count"] == 4
    assert result["pairwise_rhyme_score"] > 0
