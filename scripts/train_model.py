from weird_ai.model import WeirdAIModel
from weird_ai.tokenizer import SimpleCharacterTokenizer
from weird_ai.config import SAMPLE_LYRICS_FILE


def main():
    print("Starting training...")

    text = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")
    tokenizer = SimpleCharacterTokenizer(text)

    vocab_size = len(tokenizer.stoi)
    model = WeirdAIModel(vocab_size=vocab_size)

    # TODO:
    # 1. Load training data
    # 2. Create optimizer
    # 3. Implement training loop
    # 4. Save checkpoints

    print("Training complete.")