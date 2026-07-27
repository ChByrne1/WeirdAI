# Lesson 14: Reward Diagnostics for Weird AI

## Overview

This lesson improves the GRPO reinforcement learning pipeline by analyzing training curves, tracking better metrics, preventing reward exploitation, and adding format rewards.

In the previous lesson, you created reward signals and group-relative advantages for Weird AI.  This lesson asks a new question: Are those rewards healthy?  

>
>A model can learn to increase a reward score while producing worse outputs for humans. This is called **reward hacking** or **reward exploitation**.
>

## Part 1: Notebook

Open the jupyter notebook titled ```14_reward_diagnostics_and_hacking.ipynb```
Read the markdown explanation for each section and add the markdown answers to questions or the code implementation described to the notebook.

> **Note**: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application

Read the application code and comments in the file *reward_diagnostics.py*.  
Complete the # TODO tasks listed.  
Suggested order:
- Complete `moving_average` to smooth noisy reward curves.
- Complete `summarize_values` and `advantage_statistics`.
- Complete `detect_reward_collapse`.
- Complete `get_nonempty_lines`.
- Complete repetition diagnostics:
  - `repeated_line_ratio`
  - `repeated_word_ratio`
  - `detect_repetition_hacking`
- Complete `detect_short_high_reward`.
- Complete `format_reward`.
- Complete `combine_rewards`.
- Complete `diagnose_rollout`.
- Complete `build_diagnostic_report` and `summarize_diagnostic_report`.

Make sure all of the files pass the tests in the *test_reward_diagnostics.py* file.  You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/test_reward_diagnostics.py
```

## Part 3: Reflection questions

1. Why can reward increase while human-perceived quality decreases?
2. What does reward collapse mean, and why is it a problem for GRPO-style training?
3. Why is advantage standard deviation more informative than advantage average?
4. Chapter 7 adds format rewards to improve GRPO behavior. In Weird AI, how could a format reward help, and how could it accidentally encourage worse lyrics?
5. What is one way Weird AI might exploit a rhyme reward?
6. What is one way Weird AI might exploit a syllable consistency reward?
7. Why should reward diagnostics be reviewed before using rewards in model training?
8. How could human review still be useful even when automated diagnostics look healthy?

Take a screenshot of your completed notebook activities and your tests passing.  Put these in your Word document along with your answers to the reflection questions.

Submit your document and a zip file of your current project as two separate attachments.