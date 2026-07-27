# Lesson 9: Running a Pretrained Language Model

## Part 1: Notebook

Open the jupyter notebook titled ```09_pretrained_generation.ipynb```
Read the markdown explanation for each section and add the markdown answers to questions or the code implementation described to the notebook.

> **Note**: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application

Read the application code and comments in the file *pretrained.py*.  

Complete the # TODO tasks listed.  

Make sure all of the files pass the tests in the *test_reasoning.py* file.  You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/test_pretrained.py
```

When all tests are passing be sure to check your code into your GitHub repository.f

## Part 3: Reflection
1. What is the difference between a tokenizer and a language model?
2. Why must text be converted into token IDs before entering the model?
3. What does a pretrained model already know before fine-tuning?
4. What is the purpose of temperature during generation?
5. How does top-k sampling affect creativity?
6. How did the reasoning prompt change the generated response?
7. Why might a reasoning prompt increase generation cost?
8. How could a pretrained model improve Weird AI compared to the small models built earlier in the course?

Take a screenshot of your tests passing and include this in your Word document along with your answers to the reflection questions.

Submit your document and a zip file of your current project as two separate attachments.