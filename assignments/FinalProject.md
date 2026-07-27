# Final Project: 
- Implement a lightweight rewards based Training
- Deploy the Model

## Overview
To complete the Weird AI project, you will add the missing reinforcement learning step.  This will be done by implementing a small educational policy-gradient loop:
1. Generate rollouts
2. Evaluate rollouts
3. Compute rewards
4. Compute advantages
5. Compute log probabilities
6. Compute policy-gradient loss
7. Update model weights
Once completed, you will save the model and load it with a web application to run.

## Part 1: Notebook

Open the jupyter notebook titled ```16_mini_grpo_training.ipynb```
Read the markdown explanation for each section and add the markdown answers to questions or the code implementation described to the notebook.

> **Note**: Run the notebook cells in order. If something behaves strangely, restart the kernel and run all cells again.

## Part 2: Application

Read the application code and comments in the file *grpo.py*.  
Complete the # TODO tasks listed.  
Suggested order:
- Complete `extract_reward`
- Complete `compute_group_advantages`
- Complete `generate_rollouts`
- Complete `evaluate_rollouts`
- Complete `sequence_logprob_from_token_logprobs`
- Complete `selected_token_logprobs`
- Complete `compute_policy_gradient_loss`
- Complete `compute_entropy_from_logits`
- Complete `build_rollout_records`
- Complete `mini_grpo_update`
- Complete `summarize_mini_grpo_result`

Make sure all of the files pass the tests in the *test_grpo.py* file.  You can run the tests with the following command from the VS Code terminal:

```bash
python -m pytest tests/test_grpo.py
```

## Part 3: Model Deployment

One of the final steps in any machine learning project is preparing the model for deployment. During this step, you will save your completed Weird AI model to disk so that it can be loaded later without retraining.

Unlike the Hugging Face models used throughout the textbooks, the Weird AI model was built from scratch as a standard PyTorch nn.Module. Because of this, you will implement your own deployment package consisting of the model weights, model configuration, tokenizer, and training metadata.

After completing this section, you should be able to launch the provided Gradio web application and interact with your trained model.

### Save the Model Weights

Save the trained model parameters with this code:

```python
import torch

torch.save(
    model.state_dict(),
    output_dir / "model_state.pt"
)
```

This file contains the learned neural network weights.

### Save the Model Configuration

The model weights alone are not enough to recreate the model. Your application must also know how the model was originally constructed.

Create a JSON file named *models/weird-ai-final/model_config.json* containing all of the information required to reconstruct your model.

```python
import json

model_config = {
    "vocab_size": config.vocab_size,
    "context_length": config.context_length,
    "embedding_dim": config.embedding_dim,
    "num_heads": config.num_heads,
    "num_layers": config.num_layers,
    "dropout": config.dropout,
    "use_qkv_bias": config.use_qkv_bias
}

with open(output_dir / "model_config.json", "w") as file:
    json.dump(model_config, file, indent=4)
```

>Note: Your project may use different configuration property names. Save every value required to recreate your model using your constructor.

### Save the Tokenizer

For a Hugging Face tokenizer:

```python
tokenizer.save_pretrained(output_dir)
```

That should produce files like
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json

> The exact filenames depend on the tokenizer implementation.

### Save Training Metadata

Create a small metadata file describing the model, replacing the placeholder value with your name:

> This is not required to load the model, but it is valuable for the final demonstration

```python
training_metadata = {
    "model_name": "Weird AI",
    "training_stage": "Mini-GRPO",
    "developer": "Your Name",
    "course": "Artificial Intelligence Engineering"
}

with open(output_dir / "training_metadata.json", "w") as file:
    json.dump(training_metadata, file, indent=4)
``` 

At this point, you should have the following files in the deployment folder:

```
models/
    weird-ai-final/
        model_state.pt
        model_config.json
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json
        training_metadata.json
```

### Verify the deployment package

A deployment package is only useful if it can be loaded successfully.  Before submitting your project:

1. Start a new Python session.
2. Reconstruct your model using the values stored in *model_config.json*.
3. Load the weights from *model_state.pt*.
4. Load the tokenizer.
5. Generate text using your loaded model.

If your model can generate lyrics without retraining, your deployment package has been created successfully.

### Running the Web Interface

Once your deployment package has been created, launch the provided Gradio interface.

```bash
python app/gradio_app.py
```

The application will load your deployment package from ```models/weird-ai-final``` and then run a local webserver hosting the app.  You can access it by navigating to localhost or your computer's public IP on port 7860:

<a href="http://127.0.0.1:7860" target="_blank">http://127.0.0.1:7860</a>

If your model loads successfully, you should be able to:
- Generate parody lyrics
- Compare multiple candidate responses
- View evaluation scores
- Demonstrate self-refinement
- Showcase the completed Weird AI project during your final presentation

## Part 4: Reflection questions

1. What important pieces of production GRPO are missing from this mini version?
2. Why might this mini-GRPO trainer be unstable if used for long training runs?
3. How do reward diagnostics from Lesson 14 help make reinforcement learning safer?
4. How could distillation still be useful after reinforcement learning?
5. Why is saving only the model weights insufficient for deploying a neural network?
6. Why should a trained tokenizer be saved alongside the model?
7. What advantages does separating the model configuration from the model weights provide?
8. How does creating a deployment package make your AI system easier to distribute, test, and reuse?
9. How does this deployment process compare to packaging a traditional software application?
