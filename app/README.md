# Weird AI Gradio Application

The completed Gradio interface provides the presentation layer for the final
course project. Students do not need to implement the interface itself.

## Prerequisites

The application depends on the project modules completed in earlier lessons:

- `reasoning.py`
- `evaluate.py`
- `generator.py`
- `refinement.py`

Before using the interface, the corresponding unit tests should pass.

## Install the project

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the interface

```bash
python app/gradio_app.py
```

Open the local URL shown in the terminal, normally:

```text
http://127.0.0.1:7860
```

## Model field

The model field accepts either:

1. A Hugging Face model identifier:

   ```text
   sshleifer/tiny-gpt2
   ```

2. A local directory containing a model and tokenizer saved with:

   ```python
   model.save_pretrained("models/weird-ai-final")
   tokenizer.save_pretrained("models/weird-ai-final")
   ```

Then enter this directory in the interface:

```text
models/weird-ai-final
```

A raw `.pt` or `.pth` state-dictionary file is not enough by itself because the
application also needs the matching model configuration and tokenizer. Save the
final model in Hugging Face `save_pretrained` format for the simplest demo.

## Final-project demonstration

Students can use the interface to show:

- standard generation
- reasoning-oriented prompting
- best-of-N generation
- deterministic evaluation
- candidate ranking
- self-refinement
- a before/after comparison between pre-GRPO and post-GRPO checkpoints

Use the same prompt, seed, and generation settings when comparing checkpoints.
