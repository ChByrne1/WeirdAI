# Coding the Weird AI Attention Mechanism

## Description
In this assignment, you will add the first major neural-network building block to the Weird AI project: an attention mechanism.  When you are done, you should understand how a model calculates relationships between tokens, and how causal attention restricts those relationships to previous tokens when predicting the next token.

## Getting started

Before beginning, make sure your virtual environment is activated and your project package is installed in editable mode:
```bash
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Part 1: Notebook

Open the jupyter notebook titled ```03_attention_mechanisms.ipynb```
Read the markdown explanation for each section and add the code implementation described to the notebook.
> Note: Run the notebook cells in order.  If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Complete the *attention.py* file

Open the *attention.py* file, read through the existing code, and complete the # TODO components

Make sure all of the *attention_tests.py* file's tests pass.   Run this with the command

```bash
python -m pytest tests/attention_tests.py
```

In a Word document with the course, the assignment, your name, and the date at the top of the page, answer these questions:
1. In Weird AI, what does it mean for one lyric token to “pay attention” to another lyric token?
2. Why should a token be prevented from looking at future tokens during training?
3. What is the difference between attention scores and attention weights?
4. Why do we divide attention scores by the square root of the key dimension?
5. Why might multi-head attention be useful for lyrics?

Take a screenshot of the *attention_tests.py* results after running them and put that in your document.  All provided tests must pass. You may add additional tests, but you may not remove or weaken the provided tests.

## Part 3: Submission
Submit your document and a zip file of your current project.