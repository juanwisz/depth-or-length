"""Skip accuracy benchmark: does skipping layers change the ANSWER?

The only metric that matters: does the model get \boxed{correct_answer}
with and without layer skipping? Token-level matching of reasoning
traces is meaningless — different reasoning paths can reach the same answer.
"""

import torch
import transformers
import json
import sys
import time
import re


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} answer from generated text."""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    return ""


def generate_with_hooks(model, tokenizer, input_ids, skip_layers, max_new_tokens=8192):
    """Generate with specified layers skipped via hooks."""
    hooks = []
    for idx in skip_layers:
        def make_hook():
            def h(m, args, output, **kw):
                return (args[0],) + output[1:] if isinstance(output, tuple) else args[0]
            return h
        hooks.append(model.model.layers[idx].register_forward_hook(make_hook()))

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
        )
    elapsed = time.time() - t0

    for h in hooks:
        h.remove()

    gen_ids = out[0][input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    answer = extract_boxed_answer(text)

    return {
        "text": text,
        "answer": answer,
        "num_tokens": len(gen_ids),
        "time_s": elapsed,
        "token_ids": gen_ids.tolist(),
    }


SKIP_CONFIGS = [
    ([], "full_model"),
    ([13], "skip_1_layer"),
    ([13, 14], "skip_2_adj"),
    ([10, 17], "skip_2_spread"),
    ([13, 14, 15], "skip_3_adj"),
    ([5, 13, 22], "skip_3_spread"),
    ([4, 8, 12, 16, 20, 24], "skip_6_spread"),
]


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "skip_accuracy.jsonl"
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    n_problems = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    print(f"Loading {model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Load MATH-500
    from datasets import load_dataset
    ds = load_dataset("lighteval/MATH-Hard", split="train")
    problems = list(ds)[:n_problems]

    print(f"Loaded {len(problems)} problems, max_tokens={max_tokens}")
    print(f"Configs: {[c[1] for c in SKIP_CONFIGS]}")

    # Resume support
    completed = set()
    try:
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                completed.add((r["problem_idx"], r["config"]))
        print(f"Resuming: {len(completed)} results already done")
    except FileNotFoundError:
        pass

    for prob_idx, problem in enumerate(problems):
        question = problem.get("problem", problem.get("question", ""))
        ground_truth = problem.get("solution", problem.get("answer", ""))
        gt_answer = extract_boxed_answer(ground_truth) if "\\boxed" in ground_truth else ground_truth

        prompt = f"{question}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

        for skip_layers, config_name in SKIP_CONFIGS:
            if (prob_idx, config_name) in completed:
                continue

            result = generate_with_hooks(model, tokenizer, input_ids, skip_layers, max_tokens)
            correct = int(result["answer"] == gt_answer) if gt_answer else -1

            record = {
                "problem_idx": prob_idx,
                "config": config_name,
                "skip_layers": skip_layers,
                "correct": correct,
                "extracted_answer": result["answer"],
                "ground_truth": gt_answer,
                "num_tokens": result["num_tokens"],
                "time_s": result["time_s"],
                "generation_text": result["text"],
                "token_ids": result["token_ids"],
                "model": model_name,
                "max_new_tokens": max_tokens,
                "question": question,
                "input_tokens": input_ids.shape[1],
            }

            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            marker = "Y" if correct == 1 else ("N" if correct == 0 else "?")
            print(f"  [{prob_idx:3d}] {config_name:20s}: {marker} | "
                  f"ans={result['answer'][:20]:20s} gt={gt_answer[:20]:20s} | "
                  f"tok={result['num_tokens']:5d} t={result['time_s']:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("ACCURACY SUMMARY")
    print(f"{'='*60}")
    results = []
    with open(output_path) as f:
        for line in f:
            results.append(json.loads(line))

    for _, config_name in SKIP_CONFIGS:
        rows = [r for r in results if r["config"] == config_name]
        if not rows:
            continue
        correct = sum(r["correct"] for r in rows if r["correct"] >= 0)
        total = sum(1 for r in rows if r["correct"] >= 0)
        avg_tok = sum(r["num_tokens"] for r in rows) / len(rows)
        print(f"  {config_name:20s}: {correct}/{total} = {correct/total*100:.0f}%  avg_tok={avg_tok:.0f}")


if __name__ == "__main__":
    main()
