"""
Test script for custom to HuggingFace model conversion.
"""

import pytest
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from simple_stories_train.convert_to_hf import convert_llama_model_to_hf
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS


@pytest.mark.slow
@pytest.mark.parametrize("model_size", ["1.25M", "5M", "11M", "30M", "35M"])
@torch.inference_mode()
def test_convert_llama_model_to_hf(
    model_size: str, tokenizer_path: str = "simple_stories_train/tokenizer/simplestories-4096.json"
) -> None:
    """Test the conversion from custom llama model to a HuggingFace LlamaForCausalLM model.

    Args:
        model_size: Size of the model to test
        tokenizer_path: Path to the custom tokenizer
    """

    # Load the custom model
    model_config = MODEL_CONFIGS[model_size]
    custom_model = Llama.from_pretrained(f"SimpleStories/SimpleStories-{model_size}", model_config)
    custom_model.eval()

    # Convert the model
    hf_model = convert_llama_model_to_hf(custom_model)

    # Load custom tokenizer
    custom_tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load HF tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"SimpleStories/SimpleStories-{model_size}")
    prompt = "The curious cat looked at the"

    # Input using HF tokenizer
    hf_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    # Input using custom tokenizer
    encoding = custom_tokenizer.encode(prompt)
    custom_inputs = torch.tensor([encoding.ids], dtype=torch.long)  # Adding batch dimension with []

    # Test with HF tokenizer inputs
    hf_logits = hf_model.forward(input_ids=hf_inputs.input_ids).logits
    custom_logits = custom_model.forward(idx=hf_inputs.input_ids)[0]

    # Assert logits are the same
    torch.testing.assert_close(hf_logits, custom_logits, rtol=1e-4, atol=1e-4)

    # Test with custom tokenizer inputs
    hf_logits = hf_model.forward(input_ids=custom_inputs).logits  # pyright: ignore[reportArgumentType]
    custom_logits = custom_model.forward(idx=custom_inputs)[0]

    # Assert logits are the same
    torch.testing.assert_close(hf_logits, custom_logits, rtol=1e-4, atol=1e-4)
