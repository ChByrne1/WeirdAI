# Lesson 15: Reward Signals for Weird AI

## Overview

This lesson introduces reinforcement learning for reasoning models. The key shift is from *inference-time scaling* to *training-time scaling*.
Previous lessons improved output by spending more compute during generation, using best-of-N generation and self-refinement.
This lesson prepares Weird AI for reinforcement learning by creating reward signals.

## Part 1: Notebook

Open the jupyter notebook titled ```15_rewards_and_group_advantages.ipynb```
Read the markdown explanation for each section and add the markdown answers to questions or the code implementation described to the notebook.

> **Note**: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application

>
> This assignment does **not** train the model weights yet. It prepares the data a future reinforcement learning loop would need.
>

Read the application code and comments in the file *rewards.py*.  
Complete the # TODO tasks listed.  
Suggested order:
- Complete clamp
- Complete extract_score
- Complete normalize_reward
- Complete reward_from_evaluation
- Complete compute_group_advantages
- Complete build_rollout_reward
- Complete prepare_reward_batch
- Complete rank_by_reward
- Complete get_best_rollout and get_worst_rollout
- Complete summarize_reward_batch

Make sure all of the files pass the tests in the *test_rewards.py* file.  You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/test_rewards.py
```

## Part 3: Reflection questions

1. What is the difference between inference-time scaling and training-time scaling?
2. How is RLHF different from RLVR?
3. Why can math use verifiable rewards more easily than parody lyrics?
4. How is Weird AI's reward signal different from the book's math verifier?
5. What does a positive group-relative advantage mean?
6. What does a negative group-relative advantage mean?
7. Why might a model learn to "reward hack" Weird AI's rhyme or syllable scorer?
8. Why is this assignment a preparation step rather than a full reinforcement learning training loop?
