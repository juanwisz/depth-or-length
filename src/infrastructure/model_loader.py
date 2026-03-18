"""Model loading utilities for depth-or-length experiments.

Supports all target models:
- DeepSeek-R1-Distill-Qwen-7B (primary reasoning)
- DeepSeek-R1-Distill-Llama-8B (secondary reasoning)
- Qwen3-8B (thinking mode, extension)
- Qwen2.5-7B-Instruct (instruct baseline)
- DeepSeek-R1-Distill-Qwen-14B (scale test, 4-bit)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple


# Model registry with known configurations
MODEL_REGISTRY = {
    "deepseek-r1-qwen-7b": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "num_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "family": "qwen2",
    },
    "deepseek-r1-llama-8b": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "family": "llama",
    },
    "qwen3-8b": {
        "hf_name": "Qwen/Qwen3-8B",
        "num_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "family": "qwen3",
    },
    "qwen2.5-7b-instruct": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "num_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "family": "qwen2",
    },
    "deepseek-r1-qwen-14b": {
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "num_layers": 48,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "family": "qwen2",
    },
}


def resolve_model_name(name: str) -> str:
    """Resolve short name or HF name to HF name."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]["hf_name"]
    # Check if it's already an HF name
    for key, info in MODEL_REGISTRY.items():
        if info["hf_name"] == name:
            return name
    # Assume it's an HF name we don't have registered
    return name


def get_model_info(hf_name: str) -> dict:
    """Get model info from registry."""
    for key, info in MODEL_REGISTRY.items():
        if info["hf_name"] == hf_name:
            return info
    return None


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "auto",
    quantize_4bit: bool = False,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from HuggingFace.

    Args:
        model_name: Short name or HF model path.
        dtype: torch dtype ("auto", "float16", "bfloat16").
        quantize_4bit: Use 4-bit quantization (for 14B model).
        device_map: Device placement strategy.
        attn_implementation: Attention implementation ("sdpa", "eager", "flash_attention_2").

    Returns:
        (model, tokenizer) tuple.
    """
    hf_name = resolve_model_name(model_name)

    # Set dtype
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }

    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(hf_name, **model_kwargs)
    model.eval()

    return model, tokenizer


def get_layer_modules(model) -> list:
    """Get the list of transformer layer modules from a model.

    Handles different model architectures (Qwen2, Llama, etc.)
    """
    # Try common paths
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)
    else:
        raise ValueError(f"Cannot find layer modules in model of type {type(model)}")


def count_parameters(model) -> dict:
    """Count parameters by component type."""
    total = 0
    ffn_params = 0
    attn_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if any(k in name for k in ['mlp', 'feed_forward', 'ffn']):
            ffn_params += n
        elif any(k in name for k in ['self_attn', 'attention', 'attn']):
            attn_params += n
        else:
            other_params += n

    return {
        "total": total,
        "ffn": ffn_params,
        "attention": attn_params,
        "other": other_params,
        "ffn_fraction": ffn_params / total if total > 0 else 0,
        "attn_fraction": attn_params / total if total > 0 else 0,
    }
