"""
Text generation utilities for Weird AI.

This module contains helper functions that convert between text and token IDs
and generate new text from a trained or partially trained model.
"""

import torch


def text_to_token_ids(text, tokenizer):
    """
    Convert text into a tensor of token IDs.

    Args:
        text: The input prompt as a string.
        tokenizer: The tokenizer object.

    Returns:
        A tensor of shape (1, num_tokens).
    """

    # TODO:
    # 1. Use the tokenizer to encode the text.
    token_ids = tokenizer.encode(text)
    # 2. Convert the encoded list into a torch tensor.
    token_ids = torch.tensor(token_ids)
    # 3. Add a batch dimension using unsqueeze(0).
    token_ids = token_ids.unsqueeze(0)
    return token_ids



def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back into text.

    Args:
        token_ids: Tensor of token IDs.
        tokenizer: The tokenizer object.

    Returns:
        The decoded text as a string.
    """

    # TODO:
    # 1. Remove the batch dimension if present.
    token_ids = token_ids.squeeze(0)
    # 2. Convert the tensor to a Python list.
    token_ids = token_ids.tolist()
    # 3. Use the tokenizer to decode the list.
    return tokenizer.decode(token_ids)


def generate_text_simple(model, input_ids, max_new_tokens, context_size):
    """
    Generate text one token at a time using greedy decoding.

    Args:
        model: The Weird AI model.
        input_ids: Tensor of shape (batch_size, num_tokens).
        max_new_tokens: Number of new tokens to generate.
        context_size: Maximum number of tokens the model can consider.

    Returns:
        Tensor containing the original input IDs plus generated token IDs.
    """

    # TODO:
    # Repeat max_new_tokens times:
    for _ in range(max_new_tokens):
        # 1. Crop input_ids to the most recent context_size tokens.
        input_cond = input_ids[:, -context_size:]
        # 2. Pass the cropped input into the model to get logits.
        with torch.no_grad():
            logits = model(input_cond)
        # 3. Select only the logits for the last time step.
        logits = logits[:, -1, :]
        # 4. Use argmax to choose the most likely next token.
        input_next = torch.argmax(logits, dim=-1)
        # 5. Append that token to input_ids.
        input_ids = torch.cat([input_ids, input_next], dim=1)

    return input_ids


def generate_and_print_sample(model, tokenizer, device, start_context, context_size, max_new_tokens=50):
    """
    Generate and print a sample text output.

    This is useful during training so students can visually inspect whether
    the model is improving.

    Args:
        model: The Weird AI model.
        tokenizer: The tokenizer object.
        device: CPU or CUDA device.
        start_context: Prompt text.
        context_size: Maximum number of tokens the model can consider.
        max_new_tokens: Number of tokens to generate.
    """

    model.eval()

    # TODO:
    # 1. Convert start_context to token IDs.
    encoded = text_to_token_ids(start_context, tokenizer)
    # 2. Move token IDs to the selected device.
    encoded = encoded.to(device)
    # 3. Generate new token IDs.
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, input_ids=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size
        )
    # 4. Convert generated token IDs back to text.
    decoded = token_ids_to_text(token_ids, tokenizer)
    # 5. Print the generated text.
    print(decoded.replace('\n', ''))
    return decoded