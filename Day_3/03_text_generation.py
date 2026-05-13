"""
Day 3 – Script 03: Text Generation Strategies
===============================================
Explores different decoding / generation strategies side-by-side:
  1. Greedy Decoding
  2. Beam Search
  3. Pure Sampling
  4. Top-k Sampling
  5. Top-p (Nucleus) Sampling
  6. Temperature Effect

Model: GPT-2 (small, ~500MB, runs on CPU)

Run:
    python 03_text_generation.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("=" * 60)
print("Text Generation Strategy Comparison (GPT-2)")
print("=" * 60)

MODEL_NAME = "gpt2"
PROMPT     = "The key to understanding artificial intelligence is"

print(f"\nLoading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# Encode the prompt
input_ids = tokenizer.encode(PROMPT, return_tensors="pt")

def generate_and_print(label: str, **generate_kwargs) -> str:
    """Generate text and print the result with a label."""
    with torch.no_grad():
        output_ids = model.generate(input_ids, **generate_kwargs)
    # Decode only the newly generated tokens (skip the prompt)
    new_tokens  = output_ids[0][input_ids.shape[-1]:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\n── {label} ──")
    print(f"  Prompt   : {PROMPT}")
    print(f"  Generated: {continuation}")
    return continuation


MAX_NEW = 60

# ─────────────────────────────────────────────────────────────────────────────
# 1. Greedy Decoding – always pick the most probable next token
# ─────────────────────────────────────────────────────────────────────────────
generate_and_print(
    "1. Greedy Decoding",
    max_new_tokens=MAX_NEW,
    do_sample=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Beam Search – keep top-4 candidate sequences at each step
# ─────────────────────────────────────────────────────────────────────────────
generate_and_print(
    "2. Beam Search (num_beams=4)",
    max_new_tokens=MAX_NEW,
    num_beams=4,
    early_stopping=True,
    do_sample=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Pure Random Sampling
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
generate_and_print(
    "3. Pure Sampling (do_sample=True, temperature=1.0)",
    max_new_tokens=MAX_NEW,
    do_sample=True,
    temperature=1.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Top-k Sampling – only sample from the top-k most likely tokens
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
generate_and_print(
    "4. Top-k Sampling (top_k=50)",
    max_new_tokens=MAX_NEW,
    do_sample=True,
    top_k=50,
    temperature=0.8,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Top-p (Nucleus) Sampling – sample from smallest set summing to p
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
generate_and_print(
    "5. Top-p / Nucleus Sampling (top_p=0.92)",
    max_new_tokens=MAX_NEW,
    do_sample=True,
    top_p=0.92,
    top_k=0,            # Disable top-k so only top-p applies
    temperature=0.8,
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Temperature Effect – compare low vs high temperature
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 6. Temperature Effect ──")
for temp in [0.3, 0.7, 1.2, 1.8]:
    torch.manual_seed(42)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=temp,
            top_k=50,
        )
    new_text = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    label = "focused" if temp < 0.7 else ("creative" if temp > 1.0 else "balanced")
    print(f"  temp={temp:.1f} ({label:8s}): {new_text}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Repetition Penalty demo
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 7. Repetition Penalty ──")
for penalty in [1.0, 1.3, 1.8]:
    torch.manual_seed(42)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.9,
            repetition_penalty=penalty,
        )
    text = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"\n  penalty={penalty}: {text}")

print("\n✅  Text generation strategies demo complete.")
