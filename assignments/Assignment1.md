## Part 1:
Review the class project in Beginning the Class


### Project preparation

- Fork the Weird AI Starter repo on GitHub
- Clone your forked repo locally to work with
- From your repository root folder on your local system, run the following commands:
```bash
python -m venv .venv
pip install -U huggingface_hub
pip install -U "huggingface_hub[cli]"
hf download theelderemo/genius-lyrics-cleaned --repo-type dataset --local-dir data/raw
```

> This should download the following dataset: Hugging Face Genius Lyrics Cleaned

Move all of the train files in your project's data/raw/data folder up one level to the data/raw folder, then delete the data/raw/data folder  


## Part 2:
Explore the legal and ethical issues around this project, focusing on:
- Using copyrighted lyrics for training data
- Parody works
- Copyright and derivative works
- Responsible dataset sourcing
- AI-generated content disclosure
- bias in training data
- safety filtering

Write a one to two page summary about the legal and ethical issues of this project and how to handle them.  

Submit your Word document.