"""
Instruction fine-tuning helpers for Weird AI.
"""

import torch


def extract_response(generated_text, prompt_text):
    """
    Remove the prompt from generated text and return only the response.
    """

    # TODO:
    # Remove prompt_text from the beginning of generated_text.
    # Strip extra whitespace.

    raise NotImplementedError("Implement extract_response.")


def save_instruction_model(model, path):
    """
    Save instruction fine-tuned model weights.
    """

    # TODO:
    # Use torch.save with model.state_dict().

    raise NotImplementedError("Implement save_instruction_model.")


def load_instruction_model(model, path, device):
    """
    Load instruction fine-tuned model weights.
    """

    # TODO:
    # Use torch.load and model.load_state_dict.

    raise NotImplementedError("Implement load_instruction_model.")
