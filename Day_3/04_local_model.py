"""
Day 3 – Script 04: Loading Models Locally with AutoClasses
===========================================================
Demonstrates the manual (non-pipeline) workflow:
  - AutoTokenizer + AutoModelForCausalLM for text generation
  - AutoTokenizer + AutoModelForSequenceClassification for sentiment
  - Saving & reloading models from disk

Run:
    python 04_local_model.py
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Part A: Text Generation with AutoModelForCausalLM
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Part A: Text Generation (GPT-2) ──")

GEN_MODEL = "gpt2"
tokenizer_gen = AutoTokenizer.from_pretrained(GEN_MODEL)
model_gen     = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(device)
model_gen.eval()

prompts = [
    "Machine learning is a subset of artificial intelligence that",
    "In the year 2050, robots will",
    "The most important skill a programmer needs is",
]

for prompt in prompts:
    inputs = tokenizer_gen(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model_gen.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.75,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.2,
            pad_token_id=tokenizer_gen.eos_token_id,
        )

    # Decode only new tokens
    new_ids  = output_ids[0][inputs["input_ids"].shape[-1]:]
    new_text = tokenizer_gen.decode(new_ids, skip_special_tokens=True)
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {new_text}")

# ─────────────────────────────────────────────────────────────────────────────
# Part B: Sentiment with AutoModelForSequenceClassification
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Part B: Sentiment Classification (DistilBERT) ──")

SENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer_sent = AutoTokenizer.from_pretrained(SENT_MODEL)
model_sent     = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL).to(device)
model_sent.eval()

# Class labels from the model config
id2label = model_sent.config.id2label  # {0: 'NEGATIVE', 1: 'POSITIVE'}

sentences = [
    "The film was a masterpiece of storytelling and cinematography.",
    "I waited an hour and the food was cold and tasteless.",
    "The product is average – nothing special but does what it says.",
]

for sent in sentences:
    inputs = tokenizer_sent(sent, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        logits = model_sent(**inputs).logits

    probs  = F.softmax(logits, dim=-1)[0]
    label  = id2label[probs.argmax().item()]
    conf   = probs.max().item()

    print(f"\n  Text      : {sent}")
    print(f"  Prediction: {label} ({conf:.2%} confidence)")
    print(f"  All probs : " + " | ".join(
        f"{id2label[i]}: {p:.2%}" for i, p in enumerate(probs.tolist())
    ))

# ─────────────────────────────────────────────────────────────────────────────
# Part C: Save & Reload from Disk
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Part C: Save & Reload Model from Disk ──")

SAVE_DIR = "./saved_gpt2"

# Save
print(f"  Saving model to '{SAVE_DIR}'…")
tokenizer_gen.save_pretrained(SAVE_DIR)
model_gen.save_pretrained(SAVE_DIR)
print(f"  Files saved: {os.listdir(SAVE_DIR)}")

# Reload
print("  Reloading from disk…")
tokenizer_reload = AutoTokenizer.from_pretrained(SAVE_DIR)
model_reload     = AutoModelForCausalLM.from_pretrained(SAVE_DIR).to(device)
model_reload.eval()

test_prompt = "Loading models from disk allows offline usage:"
test_input  = tokenizer_reload(test_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    out = model_reload.generate(
        **test_input,
        max_new_tokens=30,
        do_sample=False,  # Greedy for deterministic result
        pad_token_id=tokenizer_reload.eos_token_id,
    )

new_text = tokenizer_reload.decode(out[0][test_input["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
print(f"\n  Reload test prompt: {test_prompt}")
print(f"  Generated         : '{new_text}'")

print("\n✅  Local model demo complete.")
