# Fine-Tuning Weird AI for Lyric Classification

## Overview

You will build a lyric classifier that can distinguish between different categories of lyrics.  We will classify lyrics as either serious or silly/funny.
The goal is to learn how a pretrained language model can be adapted for classification tasks.

> Note: If you haven't yet, make sure you sync your repository with the original that you forked to pull in any updates.

You may add additional helper methods or variables if needed, but do not remove or significantly alter the provided method signatures.

## Part 1: Notebook

Open the jupyter notebook titled ```06_classification_finetuning.ipynb```
Read the markdown explanation for each section and add the code implementation described to the notebook.

 > Note: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application implementation

Read the application code and comments in the following files:
- classification.py
- classification_dataset.py

Complete the TODO tasks in each.

Run all tests in test_classification.py. All tests must pass before submitting the assignment.
You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/test_classification.py
```

When all tests are passing be sure to check your code into your GitHub repository.

## Part 3: Reflection Questions
1. How is classification different from text generation?
2. Why does the output layer change from vocabulary size to number of classes?
3. Why might we freeze most of the model during fine-tuning?
4. How could a genre or mood classifier improve Weird AI’s parody generation?
5. What biases might exist in lyric genre or mood labels?
6. How is the classifier in this assignment reusing components from the Weird AI lyric generator?
