"""Test adjacent vs non-adjacent layer skipping.

Hypothesis: non-adjacent skipping degrades less because intermediate
layers can re-center the residual stream between perturbations.
"""

import torch
import transformers
import json
import sys


def test_skip(model, tokenizer, input_ids, full_tokens, skip_layers, label):
    """Generate with specified layers skipped, compare to full model."""
    hooks = []
    for idx in skip_layers:
        def make_hook():
            def hook(module, args, output, **kwargs):
                h_in = args[0]
                return (h_in,) + output[1:] if isinstance(output, tuple) else h_in
            return hook
        hooks.append(model.model.layers[idx].register_forward_hook(make_hook()))

    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=50, do_sample=False)

    for h in hooks:
        h.remove()

    skip_tokens = out[0][input_ids.shape[1]:].tolist()
    match = sum(1 for a, b in zip(full_tokens, skip_tokens) if a == b)
    total = min(len(full_tokens), len(skip_tokens))
    pct = match / total * 100 if total > 0 else 0
    print(f"{label:45s}: {match}/{total} ({pct:.0f}%)")
    return {"label": label, "layers": skip_layers, "match": match, "total": total, "pct": pct}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "nonadjacent_skip.json"

    print(f"Loading {model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompt = ("Find the number of positive integers n such that "
              "n^2 - 19n + 99 is a perfect square.\n\n"
              "Please reason step by step, and put your final answer "
              "within \\boxed{}.")
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

    # Full model baseline
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=50, do_sample=False)
    full_tokens = out[0][input_ids.shape[1]:].tolist()
    print(f"Full model: {tokenizer.decode(full_tokens[:20])}...\n")

    results = []
    t = lambda sl, l: results.append(test_skip(model, tokenizer, input_ids, full_tokens, sl, l))

    print("=== 2 LAYERS: ADJACENT vs NON-ADJACENT ===")
    t([13, 14],  "adjacent [13,14]")
    t([10, 17],  "spread-7 [10,17]")
    t([5, 22],   "spread-17 [5,22]")
    t([8, 20],   "spread-12 [8,20]")
    t([6, 14],   "spread-8 [6,14]")
    t([3, 25],   "spread-22 [3,25]")

    print("\n=== 3 LAYERS: ADJACENT vs NON-ADJACENT ===")
    t([13, 14, 15],  "adjacent [13,14,15]")
    t([8, 14, 20],   "spread [8,14,20]")
    t([5, 13, 22],   "spread [5,13,22]")
    t([4, 14, 24],   "spread [4,14,24]")

    print("\n=== 4 LAYERS: ADJACENT vs NON-ADJACENT ===")
    t([12, 13, 14, 15],  "adjacent [12-15]")
    t([6, 10, 18, 22],   "every-4ish [6,10,18,22]")
    t([4, 10, 18, 24],   "spread [4,10,18,24]")

    print("\n=== 6 LAYERS: ADJACENT vs NON-ADJACENT ===")
    t([11, 12, 13, 14, 15, 16],  "adjacent [11-16]")
    t([4, 8, 12, 16, 20, 24],    "every-4th [4,8,12,16,20,24]")
    t([3, 7, 11, 15, 19, 23],    "every-4th [3,7,11,15,19,23]")

    print("\n=== 8 LAYERS: ADJACENT vs NON-ADJACENT ===")
    t(list(range(10, 18)),         "adjacent [10-17]")
    t([3, 6, 9, 12, 15, 18, 21, 24], "every-3rd")
    t([2, 5, 8, 11, 14, 17, 20, 23], "every-3rd offset")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
