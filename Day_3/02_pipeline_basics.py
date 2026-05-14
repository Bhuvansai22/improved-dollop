"""
Day 3 – Script 02: Pipeline Basics
====================================
Demonstrates the HuggingFace pipeline() API across four common tasks:
  1. Text Generation
  2. Sentiment Classification
  3. Named Entity Recognition (NER)
  4. Summarisation (via AutoModelForSeq2SeqLM — pipeline-free)

All models are small and run well on CPU.

Run:
    python 02_pipeline_basics.py
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

print("=" * 60)
print("HuggingFace pipeline() API – 4 Task Showcase")
print("(Models download on first run – be patient!)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Text Generation
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1. Text Generation (gpt2) ──")

generator = pipeline("text-generation", model="gpt2")

prompt = "Artificial intelligence will transform the world by"
results = generator(
    prompt,
    max_new_tokens=60,
    num_return_sequences=2,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)

for i, r in enumerate(results, 1):
    print(f"\n  Variation {i}:")
    print(f"  {r['generated_text']}")

# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Sentiment Classification
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. Sentiment Classification ──")

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

reviews = [
    "This product exceeded all my expectations. Absolutely love it!",
    "Terrible quality. Broke after one day. Do not buy.",
    "It's okay. Not great, not terrible.",
]

for review in reviews:
    result = classifier(review)[0]
    print(f"  '{review[:55]}…'")
    print(f"   → Label: {result['label']}, Score: {result['score']:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Named Entity Recognition (NER)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. Named Entity Recognition (NER) ──")

ner = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",  # Merge sub-word tokens into full entities
)

text = "Elon Musk founded SpaceX in Hawthorne, California in 2002."
entities = ner(text)

print(f"  Text: {text}\n")
for ent in entities:
    print(f"  [{ent['entity_group']:4s}]  '{ent['word']}'  (score: {ent['score']:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Summarisation  (direct model call — pipeline-free)
# NOTE: The pipeline task strings "summarization" and "text2text-generation"
#       are not registered in all transformers versions. Loading the model and
#       tokenizer directly is equivalent and always works.
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Summarisation (facebook/bart-large-cnn) ──")

SUMMARISER_MODEL = "facebook/bart-large-cnn"
print(f"  Loading {SUMMARISER_MODEL} …")

sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARISER_MODEL)
sum_model     = AutoModelForSeq2SeqLM.from_pretrained(SUMMARISER_MODEL)
sum_model.eval()

article = """
The transformer architecture, introduced in the 2017 paper "Attention is All You Need"
by researchers at Google, revolutionised natural language processing. Before transformers,
recurrent neural networks (RNNs) and long short-term memory (LSTM) networks were the
dominant architectures for sequential data. However, these models struggled to handle
long-range dependencies and were slow to train because they processed tokens sequentially.
Transformers replaced recurrence with a self-attention mechanism, allowing every token
to directly attend to every other token in the sequence. This enabled massive parallelisation
during training and led to models like BERT, GPT, and T5, which set new benchmarks on
virtually every NLP task and formed the foundation of modern large language models (LLMs).
"""

inputs = sum_tokenizer(
    article,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)

with torch.no_grad():
    summary_ids = sum_model.generate(
        **inputs,
        max_length=80,
        min_length=30,
        num_beams=4,
        early_stopping=True,
    )

summary_text = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"\n  Original length : {len(article.split())} words")
print(f"  Summary length  : {len(summary_text.split())} words")
print(f"\n  Summary:\n  {summary_text}")

print("\n✅  Pipeline basics demo complete.")

