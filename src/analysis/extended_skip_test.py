"""Extended skip test: multiple MATH problems, many tokens, key configs."""

import torch
import transformers
import json
import sys
import time


PROBLEMS = [
    "Find the number of positive integers n such that n^2 - 19n + 99 is a perfect square.",
    "How many integers between 1 and 200 are multiples of both 3 and 5 but not of either 4 or 7?",
    "What is the largest prime factor of 2^{12} - 1?",
    "Let f(x) = x^3 - 3x + 1. Find the sum of all real roots of f(f(x)) = 0.",
    "Find the remainder when 3^{2024} is divided by 17.",
    "In triangle ABC, AB=13, BC=14, CA=15. Find the length of the altitude from A to BC.",
    "How many 4-digit numbers have their digits in non-decreasing order?",
    "Find the sum of all positive integers n for which n^2 + 19n + 92 is a perfect square.",
    "What is the value of the sum 1/1*2 + 1/2*3 + 1/3*4 + ... + 1/99*100?",
    "A bag contains 4 red, 3 blue, and 2 green marbles. Three marbles are drawn without replacement. What is the probability that all three are different colors?",
]

SKIP_CONFIGS = [
    ([13], "1-layer [13]"),
    ([13, 14], "2-adj [13,14]"),
    ([10, 17], "2-spread [10,17]"),
    ([5, 22], "2-spread [5,22]"),
    ([13, 14, 15], "3-adj [13-15]"),
    ([5, 13, 22], "3-spread [5,13,22]"),
    ([8, 14, 20], "3-spread [8,14,20]"),
    ([4, 14, 24], "3-spread [4,14,24]"),
]


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "extended_skip.json"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    print(f"Loading {model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    all_results = []

    for prob_idx, problem in enumerate(PROBLEMS):
        prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

        # Full model generation
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=0.6, top_p=0.95, seed=42)
        full_tokens = out[0][input_ids.shape[1]:].tolist()
        full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        t_full = time.time() - t0

        print(f"\n{'='*60}")
        print(f"Problem {prob_idx}: {problem[:60]}...")
        print(f"Full model: {len(full_tokens)} tokens in {t_full:.1f}s")
        print(f"{'='*60}")

        for skip_layers, label in SKIP_CONFIGS:
            t1 = time.time()
            hooks = []
            for idx in skip_layers:
                def make_hook():
                    def h(m, args, output, **kw):
                        return (args[0],) + output[1:] if isinstance(output, tuple) else args[0]
                    return h
                hooks.append(model.model.layers[idx].register_forward_hook(make_hook()))

            with torch.no_grad():
                out = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=True, temperature=0.6, top_p=0.95, seed=42)

            for h in hooks:
                h.remove()

            skip_tokens = out[0][input_ids.shape[1]:].tolist()
            match = sum(1 for a, b in zip(full_tokens, skip_tokens) if a == b)
            total = min(len(full_tokens), len(skip_tokens))
            pct = match / total * 100 if total > 0 else 0

            skip_text = tokenizer.decode(skip_tokens, skip_special_tokens=True)
            # Per-position match vector
            per_pos_match = [int(a == b) for a, b in zip(full_tokens, skip_tokens)]
            first_div = next((i for i, m in enumerate(per_pos_match) if m == 0), total)

            result = {
                "problem_idx": prob_idx,
                "problem": problem,
                "config": label,
                "skip_layers": skip_layers,
                "match": match,
                "total": total,
                "pct": pct,
                "first_divergence": first_div,
                "per_position_match": per_pos_match,
                "full_token_ids": full_tokens,
                "skip_token_ids": skip_tokens,
                "num_full_tokens": len(full_tokens),
                "num_skip_tokens": len(skip_tokens),
                "full_text": full_text,
                "skip_text": skip_text,
                "full_time_s": t_full,
                "skip_time_s": time.time() - t1,
                "input_tokens": input_ids.shape[1],
                "model": model_name,
                "max_new_tokens": max_tokens,
            }
            all_results.append(result)
            print(f"  {label:30s}: {match}/{total} ({pct:.0f}%)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (avg across problems)")
    print(f"{'='*60}")
    for _, label in SKIP_CONFIGS:
        rows = [r for r in all_results if r["config"] == label]
        avg_pct = sum(r["pct"] for r in rows) / len(rows)
        print(f"  {label:30s}: {avg_pct:.1f}% avg match")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
