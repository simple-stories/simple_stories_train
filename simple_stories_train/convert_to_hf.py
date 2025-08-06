"""
This script demonstrates how to convert our custom model to a HuggingFace-compatible model.
"""

from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM

from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS

# pyright: reportAttributeAccessIssue=false
# pyright: reportIndexIssue=false


def convert_llama_to_llama_for_causal_lm(custom_model: Llama) -> LlamaForCausalLM:
    """Convert Llama model to HuggingFace format.

    Args:
        custom_model: The custom Llama model to convert

    Returns:
        The converted HuggingFace model
    """
    model_config = custom_model.config

    # Create a matching HuggingFace configuration
    hf_config = HFLlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.n_embd,
        intermediate_size=model_config.n_intermediate,
        num_hidden_layers=model_config.n_layer,
        num_attention_heads=model_config.n_head,
        num_key_value_heads=model_config.n_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=model_config.rms_norm_eps,
        tie_word_embeddings=True,
    )

    hf_model = LlamaForCausalLM(hf_config)

    hf_model.model.embed_tokens.weight.data = custom_model.transformer.wte.weight.data

    for i in range(model_config.n_layer):
        # RMSNorm 1
        hf_model.model.layers[i].input_layernorm.weight.data = custom_model.transformer.h[
            i
        ].rms_1.weight.data

        # Attention weights
        # Query projection
        hf_model.model.layers[i].self_attn.q_proj.weight.data = custom_model.transformer.h[
            i
        ].attn.q_attn.weight.data

        # Key and Value are combined in your model but separate in HF model
        kv_weight = custom_model.transformer.h[i].attn.kv_attn.weight.data
        kv_dim = kv_weight.shape[0] // 2

        # Split KV weights for HF model
        hf_model.model.layers[i].self_attn.k_proj.weight.data = kv_weight[:kv_dim, :]
        hf_model.model.layers[i].self_attn.v_proj.weight.data = kv_weight[kv_dim:, :]

        # Output projection
        hf_model.model.layers[i].self_attn.o_proj.weight.data = custom_model.transformer.h[
            i
        ].attn.c_proj.weight.data

        # RMSNorm 2
        hf_model.model.layers[i].post_attention_layernorm.weight.data = custom_model.transformer.h[
            i
        ].rms_2.weight.data

        # MLP layers
        hf_model.model.layers[i].mlp.gate_proj.weight.data = custom_model.transformer.h[
            i
        ].mlp.gate_proj.weight.data
        hf_model.model.layers[i].mlp.up_proj.weight.data = custom_model.transformer.h[
            i
        ].mlp.up_proj.weight.data
        hf_model.model.layers[i].mlp.down_proj.weight.data = custom_model.transformer.h[
            i
        ].mlp.down_proj.weight.data

    # 3. Final layer norm
    hf_model.model.norm.weight.data = custom_model.transformer.rms_f.weight.data

    # 4. LM head
    hf_model.lm_head.weight.data = custom_model.lm_head.weight.data

    # Set model to eval mode
    hf_model.eval()

    return hf_model


if __name__ == "__main__":
    # Example usage: Load a custom model and convert it
    model_size = "1.25M"  # Change this to convert different model sizes
    model_config = MODEL_CONFIGS[model_size]
    custom_model = Llama.from_pretrained(f"SimpleStories/SimpleStories-{model_size}", model_config)
    custom_model.eval()

    # Convert the model
    hf_model = convert_llama_to_llama_for_causal_lm(custom_model)

    # Uncomment to save the converted model
    # hf_model.save_pretrained(f"converted_hf_model_{model_size}")
