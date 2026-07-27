# Lesson 12: Self-Refinement for Weird AI

## Overview
In the previous lesson, Weird AI used **best-of-N generation**.
In this lesson, Weird AI will use **self-refinement**
This improves output quality without retraining the model.

The Weird AI project already has a project-specific scorer from the evaluation lesson. It can score rhyme, syllable consistency, structure, and overall quality. For this assignment, you will reuse the project scorer instead of recreating the book's math-oriented heuristic scorer.

>
> The book's simple heuristic scorer rewards answer format and brevity. That makes sense for math reasoning but isn't as useful for comparing lyrics.
> 

## Part 1: Notebook

Open the jupyter notebook titled ```12_self_refinement.ipynb```
Read the markdown explanation for each section and add the markdown answers to questions or the code implementation described to the notebook.

> **Note**: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application

Read the application code and comments in the file *refinement.py*.
Complete the # TODO tasks listed.  

Suggested order:
- Complete get_score
- Complete build_refinement_prompt
- Complete refine_once
- Complete choose_better_text
- Complete should_continue
- Complete iterative_refinement
- Complete summarize_refinement

Make sure all of the files pass the tests in the *test_refinement.py* file.  You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/test_refinement.py
```

## Part 3: Reflection Questions

1. How is self-refinement different from best-of-N generation?
2. Why does self-refinement increase inference-time compute?
3. The book uses a heuristic scorer that rewards answer format and brevity. How is that different from Weird AI's project scorer?
4. Why might a revision with a higher automated score still be a worse parody to a human reader?
5. What are the advantages and disadvantages of stopping after the first failed improvement?
6. What could happen if the model is repeatedly asked to improve a song for too many iterations?
7. How might best-of-N generation and self-refinement be combined in a larger AI application?
