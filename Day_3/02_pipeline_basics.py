"""
Day 3 – Script 02: Pipeline Basics
====================================
Demonstrates the HuggingFace pipeline() API across four common tasks:
  1. Text Generation
  2. Sentiment Classification
  3. Named Entity Recognition (NER)
  4. Summarisation

All models are small and run well on CPU.

Run:
    python 02_pipeline_basics.py
"""

from transformers import pipeline

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
# Task 4: Summarisation
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Summarisation (facebook/bart-large-cnn) ──")

summariser = pipeline("summarization", model="facebook/bart-large-cnn")

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

summary = summariser(article, max_length=80, min_length=30, do_sample=False)
print(f"\n  Original length : {len(article.split())} words")
print(f"  Summary length  : {len(summary[0]['summary_text'].split())} words")
print(f"\n  Summary:\n  {summary[0]['summary_text']}")

print("\n✅  Pipeline basics demo complete.")
