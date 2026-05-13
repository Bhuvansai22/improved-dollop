"""
Day 3 – Script 01: Tokenization Deep Dive
==========================================
This script demonstrates how HuggingFace tokenizers work:
  - Basic encoding / decoding
  - Inspecting individual tokens and special tokens
  - Comparing different tokenizer vocabularies (BERT vs GPT-2)
  - Padding & truncation for batches

Run:
    python 01_tokenization.py
"""

from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load two different tokenizers for comparison
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading tokenizers (downloads on first run)…")
print("=" * 60)

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# GPT-2 has no padding token by default; add one for batch demos
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

TEXT = "HuggingFace Transformers make NLP surprisingly easy!"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Basic encode / decode
# ─────────────────────────────────────────────────────────────────────────────
print("\n── BERT Tokenizer ──")
bert_encoded = bert_tokenizer(TEXT)
bert_tokens  = bert_tokenizer.convert_ids_to_tokens(bert_encoded["input_ids"])

print(f"Input text : {TEXT}")
print(f"Token IDs  : {bert_encoded['input_ids']}")
print(f"Tokens     : {bert_tokens}")
print(f"Decoded    : {bert_tokenizer.decode(bert_encoded['input_ids'])}")

print("\n── GPT-2 Tokenizer ──")
gpt2_encoded = gpt2_tokenizer(TEXT)
gpt2_tokens  = gpt2_tokenizer.convert_ids_to_tokens(gpt2_encoded["input_ids"])

print(f"Input text : {TEXT}")
print(f"Token IDs  : {gpt2_encoded['input_ids']}")
print(f"Tokens     : {gpt2_tokens}")
print(f"Decoded    : {gpt2_tokenizer.decode(gpt2_encoded['input_ids'])}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Sub-word tokenization – rare / long words
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Sub-word Tokenization (BERT) ──")
words = ["transformers", "unhappiness", "antidisestablishmentarianism", "AI"]
for word in words:
    tokens = bert_tokenizer.tokenize(word)
    print(f"  '{word}' → {tokens}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Special tokens
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Special Tokens ──")
print(f"BERT special tokens : {bert_tokenizer.all_special_tokens}")
print(f"GPT-2 special tokens: {gpt2_tokenizer.all_special_tokens}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Batch encoding with padding & truncation
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Batch Encoding (Padding & Truncation) ──")
sentences = [
    "I love deep learning.",
    "Transformers changed the NLP landscape forever.",
    "Short sentence.",
]

batch = bert_tokenizer(
    sentences,
    padding=True,        # Pad to longest sequence in batch
    truncation=True,     # Truncate to model max length
    max_length=20,
    return_tensors="pt", # Return PyTorch tensors
)

print(f"Input IDs shape    : {batch['input_ids'].shape}")   # (3, max_len)
print(f"Attention mask     :\n{batch['attention_mask']}")   # 1=real token, 0=pad

# ─────────────────────────────────────────────────────────────────────────────
# 6. Vocabulary size comparison
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Vocabulary Sizes ──")
print(f"BERT-base-uncased vocab size : {bert_tokenizer.vocab_size:,}")
print(f"GPT-2 vocab size             : {gpt2_tokenizer.vocab_size:,}")

print("\n✅  Tokenization demo complete.")
