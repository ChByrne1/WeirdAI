# Coding the Weird AI Tokenizer
***Note***: This assignment depends on the completion of the Lesson 1 assignment.

## Part 1:
Before you begin, read this: [Syncing a Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)

Sync your repository with the original one you forked to get the latest updates.

### Instructions:
1. Open your project in Visual Studio Code and start the python virtual environment in the terminal
2. Create a new git branch, tokenization
3. Open the project file prepare_data.py and read through it, and in your own words explain in the word document what it does (do not use AI to do this)
4. Create a small training sample by running python scripts/prepare_data.py 
5. Open the project file ```tokenize_data.py``` and read through it, and in your own words explain in the word document what it does (do not use AI to do this).
6. When that is done, run python ```scripts/tokenize_data.py``` which uses the *SimpleCharacterTokenizer*
7. Finally, finish implementing ```LyricsDataset.__getitem__()``` in the *dataset.py* file

Use the comments to help assign the correct tokens to the x and y variables
Validate your implementation by running the ```lyrics_dataset_tests.py``` unit tests
Continue working until all tests pass
Put a screenshot of your passing tests in your Word document

## Part 2:

Experiment with what you've created so far by running the code blocks in the notebook named ```02_tokenization.ipynb```.
Add all changed files to the commit in git and merge your branch back into the main branch
Push your updated code to your GitHub repo

## Part 3: Reflection Questions

Answer the following questions:
1. What does a tokenization encoder do?
2. What does a tokenizer decoder do?
3. Why is it necessary to tokenize the data?
4. Why should we use a subset of data during early development?
5. When tokenizing data, explain why the target sequence shifts one token ahead of the input sequence?
