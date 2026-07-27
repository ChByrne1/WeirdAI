"""Gradio interface for the completed Weird AI project.

This application loads a deployment package created by the final project:

models/weird-ai-final/
    model_state.pt
    model_config.json
    training_metadata.json          # optional
    tokenizer files                 # Hugging Face or character tokenizer

Run from the repository root with:

    python app/gradio_app.py

The app deliberately reconstructs ``WeirdAIModel`` as a normal ``torch.nn.Module``;
it does not require the model to inherit from Hugging Face ``PreTrainedModel``.
"""

from __future__ import annotations

import inspect
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch

# Allow ``python app/gradio_app.py`` to import the src-layout package even when
# the project has not yet been installed with ``pip install -e .``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from weird_ai.model import WeirdAIModel
from weird_ai.tokenizer import SimpleCharacterTokenizer

try:
    from transformers import AutoTokenizer
except ImportError:  # The custom character tokenizer does not require it.
    AutoTokenizer = None

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "weird-ai-final"


class DeploymentError(RuntimeError):
    """Raised when a deployment package is missing or incompatible."""


class WeirdAIRuntime:
    """Hold the currently loaded model, tokenizer, device, and metadata."""

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any | None = None
        self.device = torch.device("cpu")
        self.model_dir: Path | None = None
        self.model_config: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


RUNTIME = WeirdAIRuntime()


def choose_device(preference: str = "Auto") -> torch.device:
    """Resolve a user-facing device choice to a usable torch device."""

    normalized = preference.strip().lower()

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise DeploymentError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
        return torch.device("cuda")

    if normalized == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise DeploymentError("MPS was requested, but it is not available on this system.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    """Read a JSON object from disk with a useful error message."""

    if not path.exists():
        if required:
            raise DeploymentError(f"Required file not found: {path}")
        return {}

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentError(f"Could not read valid JSON from {path}: {exc}") from exc

    if not isinstance(value, dict):
        raise DeploymentError(f"Expected a JSON object in {path}.")

    return value


def _constructor_arguments(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the values accepted by the student's WeirdAIModel constructor.

    ``model_config.json`` may either store constructor arguments directly or place
    them under a ``model_kwargs`` key. Extra documentation fields are ignored when
    the constructor does not accept them.
    """

    configured_kwargs = config.get("model_kwargs", config)
    if not isinstance(configured_kwargs, dict):
        raise DeploymentError("model_config.json field 'model_kwargs' must be an object.")

    signature = inspect.signature(WeirdAIModel.__init__)
    parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    }

    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_arbitrary_kwargs:
        return dict(configured_kwargs)

    kwargs = {
        name: configured_kwargs[name]
        for name in parameters
        if name in configured_kwargs
    }

    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]

    if missing:
        raise DeploymentError(
            "model_config.json is missing constructor value(s): "
            + ", ".join(missing)
            + ". Save every argument required by WeirdAIModel.__init__()."
        )

    return kwargs


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    """Reconstruct WeirdAIModel from its saved configuration."""

    kwargs = _constructor_arguments(config)
    try:
        return WeirdAIModel(**kwargs)
    except Exception as exc:
        raise DeploymentError(
            "WeirdAIModel could not be reconstructed from model_config.json. "
            f"Constructor arguments were: {sorted(kwargs)}. Original error: {exc}"
        ) from exc


def _load_character_tokenizer(model_dir: Path) -> SimpleCharacterTokenizer | None:
    """Load the project's tokenizer from one of two simple JSON representations.

    Supported files:

    ``character_tokenizer.json`` or ``tokenizer.json`` containing either
    ``{"chars": [...]}`` or ``{"stoi": {"a": 0, ...}}``.
    """

    for filename in ("character_tokenizer.json", "tokenizer.json"):
        path = model_dir / filename
        if not path.exists():
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue

        chars: list[str] | None = None
        if isinstance(payload.get("chars"), list):
            chars = [str(character) for character in payload["chars"]]
        elif isinstance(payload.get("stoi"), dict):
            chars = [
                str(character)
                for character, _ in sorted(
                    payload["stoi"].items(), key=lambda item: int(item[1])
                )
            ]

        if chars is not None:
            # Passing the concatenated characters recreates the same sorted
            # vocabulary used by SimpleCharacterTokenizer.__init__().
            return SimpleCharacterTokenizer("".join(chars))

    return None


def load_tokenizer(model_dir: Path) -> Any:
    """Load a Hugging Face tokenizer or the project's character tokenizer."""

    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), local_files_only=True
            )
            if getattr(tokenizer, "pad_token_id", None) is None:
                eos_token = getattr(tokenizer, "eos_token", None)
                if eos_token is not None:
                    tokenizer.pad_token = eos_token
            return tokenizer
        except Exception:
            # The deployment may intentionally contain the custom tokenizer
            # rather than a Hugging Face tokenizer.
            pass

    tokenizer = _load_character_tokenizer(model_dir)
    if tokenizer is not None:
        return tokenizer

    raise DeploymentError(
        "No usable tokenizer was found. Save a Hugging Face tokenizer with "
        "tokenizer.save_pretrained(output_dir), or save character_tokenizer.json "
        "with a 'chars' list or 'stoi' mapping."
    )


def load_deployment(model_directory: str, device_preference: str) -> tuple[str, str]:
    """Load the selected deployment package and return status/metadata text."""

    raw_path = model_directory.strip() or str(DEFAULT_MODEL_DIR)
    model_dir = Path(raw_path).expanduser()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    if not model_dir.is_dir():
        raise gr.Error(f"Model directory does not exist: {model_dir}")

    try:
        config = read_json(model_dir / "model_config.json")
        metadata = read_json(model_dir / "training_metadata.json", required=False)
        device = choose_device(device_preference)
        model = build_model(config)

        weights_path = model_dir / "model_state.pt"
        if not weights_path.exists():
            raise DeploymentError(f"Required file not found: {weights_path}")

        try:
            state_dict = torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # Compatibility with older PyTorch versions.
            state_dict = torch.load(weights_path, map_location="cpu")

        # Permit checkpoints that wrap the state dictionary in a conventional key.
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        if not isinstance(state_dict, dict):
            raise DeploymentError("model_state.pt did not contain a model state dictionary.")

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        tokenizer = load_tokenizer(model_dir)

        RUNTIME.model = model
        RUNTIME.tokenizer = tokenizer
        RUNTIME.device = device
        RUNTIME.model_dir = model_dir
        RUNTIME.model_config = config
        RUNTIME.metadata = metadata

        metadata_text = json.dumps(metadata or {"message": "No metadata file supplied."}, indent=2)
        status = (
            f"Loaded **{model_dir.name}** on **{device}**.  "
            f"Parameters: **{sum(p.numel() for p in model.parameters()):,}**"
        )
        return status, metadata_text

    except DeploymentError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Unexpected model-loading error: {exc}") from exc


def encode_text(tokenizer: Any, text: str) -> torch.Tensor:
    """Encode text with either tokenizer interface."""

    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)

    if isinstance(encoded, torch.Tensor):
        token_ids = encoded.detach().clone().long()
    else:
        token_ids = torch.tensor(encoded, dtype=torch.long)

    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)

    if token_ids.ndim != 2:
        raise DeploymentError(
            f"Tokenizer returned token IDs with unsupported shape {tuple(token_ids.shape)}."
        )

    if token_ids.shape[1] == 0:
        raise DeploymentError(
            "The tokenizer could not encode the prompt. The prompt may contain only "
            "characters outside the saved vocabulary."
        )

    return token_ids


def decode_tokens(tokenizer: Any, token_ids: torch.Tensor) -> str:
    """Decode a one-dimensional token tensor with either tokenizer interface."""

    values = token_ids.detach().cpu().tolist()
    try:
        return tokenizer.decode(values, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(values)


def extract_logits(model_output: Any) -> torch.Tensor:
    """Extract logits from common plain-PyTorch and Hugging Face-style outputs."""

    if isinstance(model_output, torch.Tensor):
        return model_output

    logits = getattr(model_output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits

    if isinstance(model_output, (tuple, list)) and model_output:
        if isinstance(model_output[0], torch.Tensor):
            return model_output[0]

    if isinstance(model_output, dict) and isinstance(model_output.get("logits"), torch.Tensor):
        return model_output["logits"]

    raise DeploymentError(
        "The model forward method must return logits as a tensor, as .logits, "
        "as the first tuple value, or under a 'logits' dictionary key."
    )


def infer_context_size(model: torch.nn.Module, config: dict[str, Any], fallback: int) -> int:
    """Find the context-window value under common configuration names."""

    candidates = (
        "context_length",
        "context_size",
        "max_context_length",
        "max_seq_len",
        "max_sequence_length",
        "block_size",
    )

    model_kwargs = config.get("model_kwargs", {})
    sources = [config, model_kwargs if isinstance(model_kwargs, dict) else {}]

    for source in sources:
        for key in candidates:
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value

    for key in candidates:
        value = getattr(model, key, None)
        if isinstance(value, int) and value > 0:
            return value

    return fallback


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Sample one token from final-position logits."""

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        remove = cumulative_probabilities > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        logits = filtered

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def generate_completion(
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    """Generate a completion using the loaded plain PyTorch model."""

    if not RUNTIME.is_loaded:
        raise DeploymentError("Load a model deployment package before generating text.")

    assert RUNTIME.model is not None
    assert RUNTIME.tokenizer is not None

    input_ids = encode_text(RUNTIME.tokenizer, prompt).to(RUNTIME.device)
    context_size = infer_context_size(
        RUNTIME.model,
        RUNTIME.model_config,
        fallback=max(input_ids.shape[1], 256),
    )

    generated = input_ids
    prompt_length = input_ids.shape[1]

    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            model_input = generated[:, -context_size:]
            logits = extract_logits(RUNTIME.model(model_input))

            if logits.ndim != 3:
                raise DeploymentError(
                    "Expected model logits with shape (batch, sequence, vocabulary); "
                    f"received {tuple(logits.shape)}."
                )

            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
            )
            generated = torch.cat((generated, next_token), dim=1)

    completion_ids = generated[0, prompt_length:]
    return decode_tokens(RUNTIME.tokenizer, completion_ids).strip()


def build_parody_prompt(topic: str, style: str, line_count: int) -> str:
    """Create the final-project prompt used for the demonstration."""

    return (
        "Write a parody song with the following requirements:\n"
        f"Topic: {topic.strip()}\n"
        f"Style: {style.strip()}\n"
        f"Length: approximately {int(line_count)} lines\n"
        "Use clear line breaks and return only the finished lyrics.\n\n"
        "Final Parody:\n"
    )


def safe_evaluate(text: str, line_count: int) -> dict[str, Any]:
    """Use the student's Lesson 10 evaluator when it has been implemented."""

    try:
        from weird_ai.evaluate import evaluate_parody

        result = evaluate_parody(text, target_line_count=int(line_count))
        if not isinstance(result, dict):
            raise TypeError("evaluate_parody() did not return a dictionary")
        return result
    except (ImportError, NotImplementedError, TypeError, ValueError) as exc:
        return {
            "overall_score": 0.0,
            "evaluation_available": False,
            "message": (
                "Automated evaluation is unavailable. Complete Lesson 10's "
                f"evaluate.py implementation. Details: {exc}"
            ),
        }


def candidate_markdown(candidates: list[dict[str, Any]]) -> str:
    """Format candidate scores and lyrics for a Gradio Markdown component."""

    sections: list[str] = []
    for rank, candidate in enumerate(candidates, start=1):
        evaluation = candidate["evaluation"]
        score = float(evaluation.get("overall_score", 0.0) or 0.0)
        sections.append(
            f"### Rank {rank} — score {score:.3f}\n\n"
            f"```text\n{candidate['text']}\n```\n\n"
            f"<details><summary>Evaluation details</summary>\n\n"
            f"```json\n{json.dumps(evaluation, indent=2, default=str)}\n```\n"
            "</details>"
        )
    return "\n\n---\n\n".join(sections)


def generate_for_interface(
    topic: str,
    style: str,
    line_count: int,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[str, str, str]:
    """Generate, evaluate, rank, and return candidates for the UI."""

    if not topic.strip():
        raise gr.Error("Enter a topic for the parody.")

    try:
        prompt = build_parody_prompt(topic, style, int(line_count))
        candidates: list[dict[str, Any]] = []

        for index in range(int(num_candidates)):
            text = generate_completion(
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
            )
            evaluation = safe_evaluate(text, int(line_count))
            candidates.append(
                {"index": index + 1, "text": text, "evaluation": evaluation}
            )

        candidates.sort(
            key=lambda candidate: float(
                candidate["evaluation"].get("overall_score", 0.0) or 0.0
            ),
            reverse=True,
        )
        best = candidates[0]

        return (
            best["text"],
            json.dumps(best["evaluation"], indent=2, default=str),
            candidate_markdown(candidates),
        )
    except DeploymentError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc


def evaluate_for_interface(text: str, target_line_count: int) -> str:
    """Evaluate pasted lyrics without generating new text."""

    if not text.strip():
        raise gr.Error("Paste or generate lyrics before evaluating them.")
    return json.dumps(
        safe_evaluate(text, int(target_line_count)),
        indent=2,
        default=str,
    )


def runtime_summary() -> str:
    """Return diagnostic information suitable for the application's About tab."""

    if not RUNTIME.is_loaded:
        return "No model is currently loaded."

    assert RUNTIME.model is not None
    summary = {
        "model_directory": str(RUNTIME.model_dir),
        "device": str(RUNTIME.device),
        "model_class": type(RUNTIME.model).__name__,
        "tokenizer_class": type(RUNTIME.tokenizer).__name__,
        "parameter_count": sum(parameter.numel() for parameter in RUNTIME.model.parameters()),
        "model_config": RUNTIME.model_config,
        "training_metadata": RUNTIME.metadata,
    }
    return json.dumps(summary, indent=2, default=str)


with gr.Blocks(title="Weird AI") as demo:
    gr.Markdown(
        "# Weird AI\n"
        "Load the final project deployment package, generate several parody "
        "candidates, and compare their automated evaluation scores."
    )

    with gr.Tab("1. Load Model"):
        gr.Markdown(
            "The deployment directory must contain `model_state.pt`, "
            "`model_config.json`, and saved tokenizer files."
        )
        with gr.Row():
            model_directory = gr.Textbox(
                value=str(DEFAULT_MODEL_DIR.relative_to(PROJECT_ROOT)),
                label="Deployment directory",
                scale=3,
            )
            device = gr.Dropdown(
                choices=["Auto", "CPU", "CUDA", "MPS"],
                value="Auto",
                label="Device",
                scale=1,
            )
        load_button = gr.Button("Load Weird AI", variant="primary")
        load_status = gr.Markdown("No model loaded.")
        metadata_output = gr.Code(label="Training metadata", language="json")
        load_button.click(
            fn=load_deployment,
            inputs=[model_directory, device],
            outputs=[load_status, metadata_output],
        )

    with gr.Tab("2. Generate"):
        with gr.Row():
            with gr.Column(scale=2):
                topic = gr.Textbox(
                    label="Parody topic",
                    placeholder="A programmer debugging code at 2:00 a.m.",
                    lines=3,
                )
                style = gr.Textbox(label="Style", value="comic rock parody")
                with gr.Row():
                    line_count = gr.Slider(2, 24, value=8, step=1, label="Target lines")
                    num_candidates = gr.Slider(1, 8, value=3, step=1, label="Candidates")
                with gr.Accordion("Generation settings", open=False):
                    max_new_tokens = gr.Slider(
                        10, 500, value=160, step=10, label="Maximum new tokens"
                    )
                    temperature = gr.Slider(
                        0.0, 2.0, value=0.8, step=0.05, label="Temperature"
                    )
                    top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 disables)")
                    top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
                generate_button = gr.Button("Generate Parody", variant="primary")

            with gr.Column(scale=3):
                best_lyrics = gr.Textbox(label="Selected lyrics", lines=18)
                best_evaluation = gr.Code(label="Selected evaluation", language="json")

        candidate_report = gr.Markdown()
        generate_button.click(
            fn=generate_for_interface,
            inputs=[
                topic,
                style,
                line_count,
                num_candidates,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
            ],
            outputs=[best_lyrics, best_evaluation, candidate_report],
        )

    with gr.Tab("3. Evaluate Lyrics"):
        evaluation_text = gr.Textbox(
            label="Lyrics",
            lines=18,
            placeholder="Paste lyrics here, or copy the selected generation.",
        )
        evaluation_line_count = gr.Slider(
            2, 24, value=8, step=1, label="Target line count"
        )
        evaluation_button = gr.Button("Evaluate")
        evaluation_result = gr.Code(label="Evaluation result", language="json")
        evaluation_button.click(
            fn=evaluate_for_interface,
            inputs=[evaluation_text, evaluation_line_count],
            outputs=evaluation_result,
        )

    with gr.Tab("4. Presentation Guide"):
        gr.Markdown(
            "## Suggested live demonstration\n\n"
            "1. Load the final post-GRPO deployment package.\n"
            "2. Generate multiple candidates for one prompt.\n"
            "3. Explain how Weird AI evaluates and ranks the candidates.\n"
            "4. Compare the selected result with a saved pre-GRPO example.\n"
            "5. Discuss one limitation of the reward function and one future improvement.\n\n"
            "The demonstration should emphasize the complete engineering pipeline: "
            "prompt → tokenization → model inference → candidate generation → evaluation → selection."
        )
        diagnostics_button = gr.Button("Show Loaded Runtime Details")
        diagnostics = gr.Code(label="Runtime details", language="json")
        diagnostics_button.click(fn=runtime_summary, outputs=diagnostics)


if __name__ == "__main__":
    demo.launch(
       server_name="0.0.0.0",
       server_port=7860, 
    )
