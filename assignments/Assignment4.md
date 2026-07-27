# Building the Weird AI Transformer Block

## Overview

This assignment will implement the remaining building blocks needed to create a transformer block:
- Layer normalization
- GELU activation
- Feed-forward network
- Residual (shortcut) connections
- Transformer block assembly

## Part 1: Notebook

Open the jupyter notebook titled ```04_transformer_building_blocks.ipynb```

Read the markdown explanation for each section and add the code implementation described to the notebook.

> Note: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application implementation

Read the application code and comments in the following files:
- layer_norm.py
- feed_forward.py
- transformer.py

Complete the TODO tasks in each.

Make sure all of the files pass the tests in *test_transformer.py*.  You can run the tests with the following command:

```bash
python -m pytest tests/test_transformer.py
```

## Part 3: Reflection Questions

1. Why is layer normalization important?
2. Why does GPT use GELU instead of ReLU?
3. Why does the feed-forward network expand to four times the embedding dimension?
4. What problem do shortcut connections solve?
5. Why must the transformer block return the same embedding size it receives?
