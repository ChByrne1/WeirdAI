# Fine-Tuning Weird AI to Follow Parody Instructions

## Part 1: Notebook

Open the jupyter notebook titled ```07_instruction_finetuning.ipynb```

Read the markdown explanation for each section and add the code implementation described to the notebook.

> Note: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application implementation

Read the application code and comments in the following files in this order:
- instruction_data.py
- instruction_dataset.py
- instruction_collate.py
- instruction_tuning.py

Other files that have been added to help with this lesson include:
- data/instruction/weird_ai_instruction_data.json

Complete the TODO tasks in each.

Make sure all of the files pass the tests in the following test files:
- test_instruction_dataset.py
- test_instruction_data.py
- test_instruction_collate.py

There are tests for each of the files you will work on in this lesson.

You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/<testfile>.py
```

When all tests are passing be sure to check your code into your GitHub repository.

## Part 3: Reflection Questions
1. How is instruction fine-tuning different from pretraining?
2. Why do instruction datasets contain instruction, input, and output fields?
3. Why do we use -100 in target labels?
4. How does instruction fine-tuning change Weird AI from lyric generator to parody assistant?
5. What kinds of instruction examples would improve Weird AI?
6. Which components from earlier Weird AI assignments were reused in this lesson?
7. What concept from this lesson will be most important for building reasoning models?

Take a screenshot of your tests passing and include this in your Word document along with your answers to the reflection questions.

Submit your document and a zip file of your current project as two separate attachments.