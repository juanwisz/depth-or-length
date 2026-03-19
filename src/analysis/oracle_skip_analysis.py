"""Oracle skip analysis: per-position, per-layer skip ceiling.

Luciano's method:
1. Run the model normally, save hidden states at every layer for every position.
2. For each position t (after warmup layers):
   - For each layer k (from warmup+1 to N):
     - Take the CORRECT hidden state from layer k-1 at position t
     - Skip layer k (pass hidden state through unchanged)
     - Run the remaining layers k+1..N
     - Get the logits and predicted token
     - Compare to the full-model prediction
3. Keep the CORRECT hidden state for position t before moving to t+1.
4. This gives: for each (position, layer), whether skipping that layer
   changes the prediction. The fraction of layers that can be skipped
   without changing the prediction is the theoretical ceiling.

Output: a matrix of shape (num_positions, num_layers) where each cell
is 1 if skipping that layer at that position preserves the correct prediction,
0 otherwise. Plus summary statistics.
"""

import torch
import json
import os
import time
import argparse
from typing import Dict, List, Any


def run_oracle_analysis(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    warmup_layers: int = 2,
    max_new_tokens: int = 100,
) -> Dict[str, Any]:
    """Run oracle skip analysis on a single input.

    For each generated token position, test skipping each layer individually
    and check if the prediction changes.

    Args:
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        input_ids: Input token IDs, shape (1, seq_len).
        warmup_layers: Number of initial layers to never skip.
        max_new_tokens: How many tokens to generate and analyze.

    Returns:
        Dict with:
        - skip_matrix: list of lists, [position][layer] = 1 if safe to skip
        - tokens: list of generated token strings
        - token_ids: list of generated token IDs
        - full_model_logits: logits from full model at each position
        - summary: per-layer skip rate, per-position skip rate
    """
    device = input_ids.device
    num_layers = model.config.num_hidden_layers
    skippable_layers = list(range(warmup_layers, num_layers))

    # Results
    skip_matrix = []  # [position_idx][layer_idx] = 1 if safe to skip
    generated_tokens = []
    generated_ids = []

    current_ids = input_ids.clone()
    past_kv = None  # KV cache for autoregressive generation

    for pos_idx in range(max_new_tokens):
        # Step 1: Full forward pass with KV cache
        with torch.no_grad():
            if past_kv is None:
                # First token: process full prompt
                outputs = model(
                    current_ids,
                    use_cache=True,
                )
            else:
                # Subsequent tokens: only process the new token
                outputs = model(
                    current_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                )

        full_logits = outputs.logits[0, -1, :]
        full_pred = full_logits.argmax().item()
        past_kv = outputs.past_key_values

        # Check for EOS
        if full_pred == tokenizer.eos_token_id:
            break

        # Step 2: For each skippable layer, test if skipping changes prediction
        # Use hook-based approach with KV cache: for each skip test, we forward
        # only the LAST token with past_key_values, but with one layer hooked
        # to passthrough. This is O(1) per token per layer, not O(seq_len).
        position_skip = []
        layers = model.model.layers

        for layer_idx in range(num_layers):
            if layer_idx < warmup_layers:
                position_skip.append(0)
                continue

            def skip_hook(module, args, output, **kwargs):
                h_in = args[0]
                if isinstance(output, tuple):
                    return (h_in,) + output[1:]
                return h_in

            hook = layers[layer_idx].register_forward_hook(skip_hook)

            # Forward only the last token with cached KV from full model
            with torch.no_grad():
                skip_outputs = model(
                    current_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=False,
                )

            hook.remove()

            skip_logits = skip_outputs.logits[0, -1, :]
            skip_pred = skip_logits.argmax().item()

            position_skip.append(1 if skip_pred == full_pred else 0)

        skip_matrix.append(position_skip)
        generated_tokens.append(tokenizer.decode([full_pred]))
        generated_ids.append(full_pred)

        # Step 3: Append the correct token and continue
        current_ids = torch.cat([
            current_ids,
            torch.tensor([[full_pred]], device=device)
        ], dim=1)

        if pos_idx % 10 == 0:
            safe_count = sum(position_skip[warmup_layers:])
            total_skippable = num_layers - warmup_layers
            print(f"  pos {pos_idx}: token='{generated_tokens[-1]}' "
                  f"safe_to_skip={safe_count}/{total_skippable} "
                  f"({safe_count/total_skippable*100:.0f}%)")

    # Summary statistics
    if skip_matrix:
        import numpy as np
        matrix = np.array(skip_matrix)

        # Per-layer: what fraction of positions can this layer be safely skipped?
        per_layer_rate = matrix[:, warmup_layers:].mean(axis=0).tolist()

        # Per-position: what fraction of layers can be safely skipped?
        per_position_rate = matrix[:, warmup_layers:].mean(axis=1).tolist()

        # Overall skip rate
        overall_rate = matrix[:, warmup_layers:].mean()

        summary = {
            "overall_skip_rate": float(overall_rate),
            "per_layer_skip_rate": {
                f"layer_{i+warmup_layers}": float(r)
                for i, r in enumerate(per_layer_rate)
            },
            "per_position_skip_rate": per_position_rate,
            "num_positions": len(skip_matrix),
            "num_layers": num_layers,
            "warmup_layers": warmup_layers,
        }
    else:
        summary = {"error": "no tokens generated"}

    return {
        "skip_matrix": skip_matrix,
        "tokens": generated_tokens,
        "token_ids": generated_ids,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle skip analysis")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--prompt", default=None, help="Custom prompt")
    parser.add_argument("--problem", default=None, help="MATH problem text")
    parser.add_argument("--warmup_layers", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--output", default="oracle_skip_results.json")
    args = parser.parse_args()

    import transformers

    print(f"Loading model: {args.model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Prepare prompt
    if args.prompt:
        prompt = args.prompt
    elif args.problem:
        prompt = (
            f"{args.problem}\n\n"
            f"Please reason step by step, and put your final answer within \\boxed{{}}."
        )
    else:
        prompt = (
            "What is 2 + 3 * 4?\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    # Tokenize with chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    input_ids = input_ids.to(model.device)

    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Warmup layers: {args.warmup_layers}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()

    start = time.time()
    results = run_oracle_analysis(
        model, tokenizer, input_ids,
        warmup_layers=args.warmup_layers,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Generated {len(results['tokens'])} tokens")
    print(f"Overall skip rate: {results['summary'].get('overall_skip_rate', 0)*100:.1f}%")

    # Show per-layer skip rates
    if 'per_layer_skip_rate' in results['summary']:
        print("\nPer-layer skip rates:")
        for layer, rate in results['summary']['per_layer_skip_rate'].items():
            bar = '#' * int(rate * 50)
            print(f"  {layer}: {rate*100:5.1f}% {bar}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
