# Lesson 11: Best-of-N Generation for Weird AI

Chapter 4 of *Build a Reasoning Model from Scratch* introduces inference-time scaling:
improving output quality by spending more compute during generation instead of retraining the model.

For Weird AI, we will adapt the chapter's self-consistency idea. Instead of generating several math answers and voting for the most common one, Weird AI will generate several parody candidates, evaluate each one, and select the highest-scoring result.

## Part 1: Notebook

Open the jupyter notebook titled ```11_inference_time_scaling.ipynb```
Read the markdown explanation for each section and add the markdown answers to questions or the code implementation described to the notebook.

> **Note**: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application

Read the application code and comments in the file *generator.py*.  

Complete the # TODO tasks listed.  

Make sure all of the files pass the tests in the *test_generator.py* file.  You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/test_generator.py
```

## Part 3: Reflection questions

1. How is best-of-N generation similar to self-consistency from the chapter?
2. Why does generating more candidates increase inference-time compute?
3. Why might a higher temperature produce better creative results but worse factual results?
4. Why does Weird AI use scoring instead of majority voting?
5. What are the risks of selecting a parody using only automated evaluation metrics?
6. How could this pipeline be combined with the evaluation tools from the previous lesson?

Take a screenshot of your tests passing and include this in your Word document along with your answers to the reflection questions.

Submit your document and a zip file of your current project as two separate attachments.