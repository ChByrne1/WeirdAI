import torch
from torch.utils.data import DataLoader, RandomSampler
from weird_ai.trainer import train_model_simple
from weird_ai.dataset import LyricsDataset
from weird_ai.model import WeirdAIModel
from weird_ai.tokenizer import SimpleCharacterTokenizer
from weird_ai.config import SAMPLE_LYRICS_FILE, PROJECT_ROOT
from weird_ai.trainer import save_checkpoint

def main():
    print("Starting training...")

    text = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")
    tokenizer = SimpleCharacterTokenizer(text)
    context_len = 128
    training_ratio = 0.9
    num_epochs = 1
    batch_size = 12

    vocab_size = len(tokenizer.stoi)
    model = WeirdAIModel(vocab_size=vocab_size)
    device = torch.device("mps")
    model.to(device)

    # 1. Load training data
    split_index = int(len(text) * training_ratio)
    training_text = text[:split_index]
    validation_text = text[split_index:]

    training_tokens = tokenizer.encode(training_text)
    validation_tokens = tokenizer.encode(validation_text)

    training_dataset = LyricsDataset(training_tokens, context_len)
    validation_dataset = LyricsDataset(validation_tokens, context_len)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=RandomSampler(training_dataset, replacement=True, num_samples=len(training_dataset)),
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # 2. Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay=0.1)

    # 3. Implement training loop
    train_losses, val_losses, track_tokens_seen = train_model_simple(
            model, train_loader=training_loader, val_loader=validation_loader,
            optimizer=optimizer, device=device, num_epochs=num_epochs,
            eval_freq=500, eval_iter=5, start_context=training_text,
            tokenizer=tokenizer, context_size=context_len
    )

    # 4. Save checkpoints
    checkpoint_path = PROJECT_ROOT / "models" / "lesson-05-pretrained" / "checkpoint.pt"
    save_checkpoint(model, optimizer, num_epochs, train_losses, val_losses, track_tokens_seen, checkpoint_path)

    print("Training complete.")

if __name__ == "__main__":
    main()
