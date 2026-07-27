"""
Custom collate function for Weird AI instruction fine-tuning.
"""

import torch


def custom_collate_fn(
    batch,
    pad_token_id=0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Pad variable-length token ID lists, create shifted targets, and mask extra padding.
    """

    # TODO:
    # Find the maximum length in the batch.
    # Add 1 because we append a padding/end token before creating targets.
    batch_max_length = None

    inputs_lst = []
    targets_lst = []

    for item in batch:
        new_item = item.copy()

        # TODO: append one pad token to new_item.

        # TODO: pad new_item to batch_max_length.
        padded = None

        # TODO: inputs are padded[:-1], targets are padded[1:].
        inputs = None
        targets = None

        # TODO: replace all but the first pad_token_id in targets with ignore_index.

        # TODO: if allowed_max_length is not None, truncate inputs and targets.

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # TODO: stack input and target tensors, then move them to device.
    inputs_tensor = None
    targets_tensor = None

    return inputs_tensor, targets_tensor
