# Teaching Weird AI to Write Lyrics

By the end of this assignment you should be able to:
- Generate text using Weird AI
- Calculate cross-entropy loss
- Measure training and validation performance
- Implement a training loop
- Observe loss reduction during training
- Save model checkpoints

## Part 1: Notebook

Open the jupyter notebook titled ```05_pretraining.ipynb```

Read the markdown explanation for each section and add the code implementation described to the notebook.

>  Note: Run the notebook cells in order.  If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application implementation

Read the application code and comments in the following files:
- losses.py
- generation.py
- trainer.py

Complete the TODO tasks in each.

Make sure all of the files pass the tests in *test_losses.py* and *test_trainer.py*.  There are tests for each of the files you will work on in this lesson.

You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/<testfile>.py
```

When all tests are passing be sure to check your code into your GitHub repository.

## Part 3: Reflection Questions
1. Why does an untrained model generate nonsense text?
2. What is the difference between logits and probabilities?
3. Why is cross-entropy loss useful?
4. What does perplexity measure?
5. Why do we need both training and validation datasets?
6. What role does backpropagation play in learning?
7. Why should model checkpoints be saved during training?

In a Word document, answer the reflection questions. 
Take a screenshot of your tests passing and include this in your Word document.

Submit your document and a zip file of your current project as two separate attachments.