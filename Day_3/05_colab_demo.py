"""
Day 3 – Script 05: Google Colab Demo
======================================
This script is designed to be copied into a Google Colab notebook.
It demonstrates:
  1. GPU availability check
  2. Half-precision (float16) inference on GPU
  3. Mounting Google Drive to cache models
  4. Running a larger model (microsoft/phi-2) on a free Colab T4 GPU

HOW TO USE:
  1. Open https://colab.research.google.com
  2. Runtime → Change runtime type → GPU (T4)
  3. Paste this script into a code cell and run

NOTE: This script will NOT run well on CPU-only machines because
      TinyLlama is a 1.1B parameter model (~2.5GB in float16).
      GPT-2 is used as a fallback if no GPU is detected.
"""

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Environment check
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Colab Environment Check")
print("=" * 60)

cuda_available = torch.cuda.is_available()
print(f"CUDA available : {cuda_available}")

if cuda_available:
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    DTYPE  = torch.float16
    DEVICE = 0            # GPU index
    MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
else:
    print("No GPU found — falling back to CPU with GPT-2")
    DTYPE  = torch.float32
    DEVICE = -1           # CPU
    MODEL  = "gpt2"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 (Colab only): Install libraries
# ─────────────────────────────────────────────────────────────────────────────
# Uncomment the lines below when running in Colab:
# !pip install transformers accelerate -q

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 (Optional): Mount Google Drive for model caching
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = None   # Set to "/content/drive/MyDrive/hf_cache" after mounting

# Uncomment to mount Drive:
# from google.colab import drive
# drive.mount("/content/drive")
# CACHE_DIR = "/content/drive/MyDrive/hf_cache"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Load model with pipeline
# ─────────────────────────────────────────────────────────────────────────────
from transformers import pipeline

print(f"\nLoading model: {MODEL}")
print("(First run downloads the model — this may take a few minutes on Colab)")

pipe = pipeline(
    "text-generation",
    model=MODEL,
    device=DEVICE,
    torch_dtype=DTYPE,
    model_kwargs={"cache_dir": CACHE_DIR} if CACHE_DIR else {},
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Run inference
# ─────────────────────────────────────────────────────────────────────────────
prompts = [
    "Explain the concept of a neural network to a 10-year-old:",
    "Write a Python function to find the factorial of a number:",
    "The three most important things about prompt engineering are:",
]

print("\n── Inference Results ──")
for prompt in prompts:
    result = pipe(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False,   # Only return new tokens, not the prompt
    )
    print(f"\nPrompt : {prompt}")
    print(f"Output : {result[0]['generated_text'].strip()}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Check GPU memory usage (Colab only)
# ─────────────────────────────────────────────────────────────────────────────
if cuda_available:
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    print(f"\n── GPU Memory ──")
    print(f"  Allocated : {allocated:.2f} GB")
    print(f"  Reserved  : {reserved:.2f} GB")

print("\n✅  Colab demo complete.")
