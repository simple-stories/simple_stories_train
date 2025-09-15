"""
Usage:
```bash
python scripts/push_to_hf.py \
  --checkpoint-path /path/to/checkpoint.pt \
  --repo-id your-username/your-repo \
  --token $HF_TOKEN
```
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from tokenizers import Tokenizer
from transformers import PreTrainedModel

from simple_stories_train.models.gpt2 import (
    GPT2,
    GPT2Config,
    convert_gpt2_to_hf_gpt2,
)
from simple_stories_train.models.llama import (
    Llama,
    LlamaConfig,
    convert_llama_to_llama_for_causal_lm,
)
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from simple_stories_train.tokenizer import convert_to_hf_tokenizer


@dataclass
class PushArgs:
    checkpoint_path: Path
    repo_id: str
    token: str | None
    private: bool
    revision: str | None
    commit_message: str | None
    model_card_readme: Path | None


def parse_args() -> PushArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local custom Llama checkpoint, convert to Hugging Face format, and "
            "push to the Hub."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the local .pt checkpoint saved via torch.save(model.state_dict(), ...)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Destination repository in the form 'username/repo_name' or 'org/repo_name'",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, will use HF_TOKEN env var if present.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private (default: public).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch name on the Hub (e.g., 'main').",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--model-card-readme",
        type=str,
        default=None,
        help="Optional path to a README.md to upload as the model card.",
    )

    ns = parser.parse_args()
    token = ns.token or os.environ.get("HF_TOKEN")

    return PushArgs(
        checkpoint_path=Path(ns.checkpoint_path).expanduser().resolve(),
        repo_id=ns.repo_id,
        token=token,
        private=bool(ns.private),
        revision=ns.revision,
        commit_message=ns.commit_message,
        model_card_readme=(
            Path(ns.model_card_readme).expanduser().resolve() if ns.model_card_readme else None
        ),
    )


def load_config_from_checkpoint_dir(checkpoint_path: Path) -> tuple[str, LlamaConfig | GPT2Config]:
    """Load model config by reading model_id from final_config.yaml adjacent to the checkpoint.

    Returns (model_id, config) where config is one of LlamaConfig or GPT2Config.
    """
    final_cfg_path = checkpoint_path.parent / "final_config.yaml"
    if not final_cfg_path.exists():
        raise FileNotFoundError(
            f"Could not find 'final_config.yaml' next to checkpoint at {final_cfg_path}"
        )

    with final_cfg_path.open("r") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    model_id = data.get("model_id")
    if not isinstance(model_id, str):
        raise ValueError("'model_id' missing or invalid in final_config.yaml")

    preset = MODEL_CONFIGS.get(model_id)
    if preset is None:
        raise ValueError(
            f"Unknown model_id '{model_id}' in final_config.yaml."
            f" Available: {tuple(MODEL_CONFIGS.keys())}"
        )

    # Optionally override context length from training config if present
    train_ds_cfg = data.get("train_dataset_config", {}) or {}
    n_ctx_override = train_ds_cfg.get("n_ctx")
    if isinstance(n_ctx_override, int) and n_ctx_override > 0:
        if isinstance(preset, LlamaConfig):
            return model_id, preset.model_copy(
                update={"n_ctx": n_ctx_override, "block_size": n_ctx_override}
            )
        if isinstance(preset, GPT2Config):
            return model_id, preset.model_copy(update={"block_size": n_ctx_override})
    return model_id, preset


def load_custom_model(
    checkpoint_path: Path, model_id: str, config: LlamaConfig | GPT2Config
) -> Llama | GPT2:
    # Llama requires special loader to rebuild rotary buffers
    if isinstance(config, LlamaConfig):
        model = Llama.from_pretrained(str(checkpoint_path), config=config, strict=True)
    else:
        # GPT-2: regular state_dict load
        state_dict = torch.load(str(checkpoint_path), weights_only=True, map_location="cpu")
        # Strip DDP prefixes if present
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model = GPT2(config)
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def convert_to_hf_model(custom_model: Llama | GPT2) -> PreTrainedModel:
    if isinstance(custom_model, Llama):
        hf_model = convert_llama_to_llama_for_causal_lm(custom_model)
    else:
        hf_model = convert_gpt2_to_hf_gpt2(custom_model)
    hf_model.eval()
    return hf_model


def find_saved_tokenizer(output_dir: Path) -> Path | None:
    """Find the saved tokenizer in the training output directory."""
    tokenizer_path = output_dir / "tokenizer.json"
    if tokenizer_path.exists():
        return tokenizer_path

    known_default = Path("simple_stories_train/tokenizer/simplestories-tokenizer.json")
    if known_default.is_file():
        return known_default.resolve()

    return None


def convert_and_upload_tokenizer(
    repo_id: str,
    token: str | None,
    model_max_length: int,
    output_dir: Path,
) -> None:
    """Convert raw tokenizer to HF format and upload to Hub."""
    tokenizer_path = find_saved_tokenizer(output_dir)
    if tokenizer_path is None:
        print(f"No tokenizer found in {output_dir}, skipping tokenizer upload")
        return

    print(f"Found tokenizer at {tokenizer_path}")

    try:
        raw_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        return

    hf_tokenizer = convert_to_hf_tokenizer(raw_tokenizer, model_max_length)
    hf_tokenizer.push_to_hub(
        repo_id=repo_id,
        token=token,
        commit_message="Upload tokenizer",
    )
    print(f"Tokenizer uploaded to {repo_id}")


def push_model_to_hub(
    hf_model: PreTrainedModel,
    repo_id: str,
    token: str | None,
    private: bool,
    revision: str | None,
    commit_message: str | None,
) -> None:
    # Call via the class to satisfy certain linters complaining about 'self'
    hf_model.__class__.push_to_hub(
        hf_model,
        repo_id=repo_id,
        private=private,
        token=token,
        commit_message=commit_message,
        revision=revision,
    )


def optionally_upload_readme(repo_id: str, token: str | None, readme_path: Path | None) -> None:
    if readme_path is None:
        return
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path}")

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def main() -> None:
    args = parse_args()

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    model_id, config = load_config_from_checkpoint_dir(args.checkpoint_path)
    custom_model = load_custom_model(args.checkpoint_path, model_id, config)

    # Convert and push model
    hf_model = convert_to_hf_model(custom_model)
    push_model_to_hub(
        hf_model=hf_model,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        revision=args.revision,
        commit_message=args.commit_message,
    )

    # Get model max length
    model_max_len = 1024  # default
    if isinstance(config, LlamaConfig):
        model_max_len = config.n_ctx
    elif isinstance(config, GPT2Config):
        model_max_len = config.block_size

    # Convert and upload tokenizer
    # The models are stored inside checkpoints folder and tokenizer is saved outside
    output_dir = args.checkpoint_path.parent.parent
    convert_and_upload_tokenizer(
        repo_id=args.repo_id,
        token=args.token,
        model_max_length=model_max_len,
        output_dir=output_dir,
    )

    # Optional README
    optionally_upload_readme(args.repo_id, args.token, args.model_card_readme)

    print(f"Successfully uploaded model and tokenizer to {args.repo_id}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
