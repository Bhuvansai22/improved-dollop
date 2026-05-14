# Day 3 – HuggingFace Transformers & Running Models Locally

HuggingFace is the central hub for open-source AI. It hosts thousands of pre-trained models for NLP, vision, audio, and multimodal tasks. Today we explore the `transformers` library — the primary SDK for loading and running these models — along with the concepts of **pipelines**, **tokenization**, and **text generation**.

---

## 1. What is HuggingFace?

HuggingFace (🤗) is an AI company and open-source platform that provides:

| Component | Description |
|---|---|
| **Model Hub** | 500,000+ pre-trained models (BERT, GPT-2, LLaMA, Mistral, etc.) |
| **`transformers` library** | Python SDK to load, run, and fine-tune models |
| **`datasets` library** | Standardised datasets for training and evaluation |
| **`tokenizers` library** | Ultra-fast tokenisation implementations |
| **Spaces** | Free hosting for ML demos (Gradio / Streamlit apps) |
| **Inference API** | Cloud-hosted model inference via REST or Python SDK |

### Why HuggingFace matters for GenAI

- Most open-source LLMs (Mistral, LLaMA, Falcon, Phi) are released on the HuggingFace Hub first.
- The `pipeline()` abstraction lets you run state-of-the-art models in **3 lines of code**.
- Models can run **locally** (CPU or GPU) or on **Google Colab** for free.

---

## 2. The Transformer Architecture (Quick Recap)

Before running models, it helps to understand the three main model families on HuggingFace:

```
Input Text → Tokenizer → Token IDs → Model → Logits / Embeddings → Output
```

| Architecture | Examples | Best For |
|---|---|---|
| **Encoder-only** | BERT, RoBERTa, DistilBERT | Classification, NER, embeddings |
| **Decoder-only** | GPT-2, LLaMA, Mistral, Falcon | Text generation, chat |
| **Encoder-Decoder** | T5, BART, mT5 | Translation, summarisation, Q&A |

---

## 3. Tokenization – How Text Becomes Numbers

A **tokenizer** converts raw text into numerical token IDs that the model can process. Each model has its own tokenizer trained alongside it — you must always use the matching tokenizer.

### Key Concepts

| Concept | Explanation |
|---|---|
| **Token** | A chunk of text (word, sub-word, or character). "unhappiness" → ["un", "happiness"] |
| **Vocabulary** | The full set of known tokens for a model |
| **Special Tokens** | `[CLS]`, `[SEP]`, `<s>`, `</s>`, `<pad>` — added automatically |
| **Attention Mask** | `1` for real tokens, `0` for padding tokens |
| **Encoding** | Text → token IDs |
| **Decoding** | Token IDs → text |

### Tokenization Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "HuggingFace makes NLP easy!"
encoded = tokenizer(text)

print("Token IDs:", encoded["input_ids"])
# → [101, 17662, 12172, 2515, 17953, 2361, 4154, 999, 102]

print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
# → ['[CLS]', 'hugging', '##face', 'makes', 'nl', '##p', 'easy', '!', '[SEP]']

print("Decoded:", tokenizer.decode(encoded["input_ids"]))
# → [CLS] huggingface makes nlp easy! [SEP]
```

### Why Sub-word Tokenization?

- **Problem:** Vocabulary can't contain every word → out-of-vocabulary (OOV) words break models.
- **Solution:** Break rare words into known sub-pieces: `"transformers"` → `["transform", "##ers"]`
- Algorithms: **BPE** (GPT-2), **WordPiece** (BERT), **SentencePiece** (T5, LLaMA)

---

## 4. The `pipeline()` API – 3-Line Inference

The `pipeline()` function is the easiest way to use any HuggingFace model. It:
1. Downloads the model and tokenizer automatically
2. Handles preprocessing (tokenization, batching)
3. Handles postprocessing (decoding, score formatting)

### Syntax

```python
from transformers import pipeline

pipe = pipeline(task="<task-name>", model="<model-name>")
result = pipe("your input text")
```

### Common Pipeline Tasks

| Task String | Description | Example Model |
|---|---|---|
| `"text-generation"` | Generate text completions | `gpt2`, `microsoft/phi-2` |
| `"text-classification"` | Classify text (e.g., sentiment) | `distilbert-base-uncased-finetuned-sst-2-english` |
| `"fill-mask"` | Fill in a `[MASK]` token | `bert-base-uncased` |
| `"ner"` | Named Entity Recognition | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| `"text2text-generation"` | Summarise / translate (seq2seq) | `facebook/bart-large-cnn` |
| `"translation"` | Translate between languages | `Helsinki-NLP/opus-mt-en-fr` |
| `"question-answering"` | Extract answers from context | `deepset/roberta-base-squad2` |
| `"zero-shot-classification"` | Classify without training | `facebook/bart-large-mnli` |

---

## 5. Text Generation in Depth

Text generation uses **decoder-only** models. At each step, the model predicts the most likely next token, which is appended to the sequence (autoregressive generation).

### Generation Strategies

| Strategy | What it does | Use When |
|---|---|---|
| **Greedy Search** | Always picks the highest-probability next token | Deterministic, short outputs |
| **Beam Search** | Keeps top-`k` candidate sequences at each step | Translations, summarisation |
| **Sampling** | Randomly samples from the probability distribution | Creative, diverse text |
| **Top-k Sampling** | Samples from the top `k` most likely tokens | Balance quality + diversity |
| **Top-p (Nucleus)** | Samples from tokens whose cumulative prob ≥ `p` | Most natural-sounding text |
| **Temperature** | Scales the logit distribution (higher = more random) | Tune creativity |

### Key Generation Parameters

```python
pipe = pipeline("text-generation", model="gpt2")

result = pipe(
    "The future of artificial intelligence is",
    max_new_tokens=100,       # How many new tokens to generate
    do_sample=True,           # Enable sampling (vs greedy)
    temperature=0.7,          # Lower = more focused, Higher = more random
    top_k=50,                 # Sample from top-50 tokens only
    top_p=0.92,               # Nucleus sampling threshold
    repetition_penalty=1.2,   # Penalise repeated tokens
    num_return_sequences=2,   # Generate 2 variations
)
```

---

## 6. Running Models Locally

### Installation

```bash
pip install transformers torch accelerate
# For faster tokenization:
pip install tokenizers
# Optional: datasets, evaluate
pip install datasets evaluate
```

### Full Local Pipeline Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"  # ~500MB, good for CPU

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time in a kingdom far away,"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )

generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated)
```

### Device Selection (CPU vs GPU)

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
```

### Memory-Efficient Loading (Large Models)

```python
# For large models (>7B parameters) — requires bitsandbytes
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=quantization_config,
    device_map="auto",
)
```

---

## 7. Running Models on Google Colab

Google Colab provides **free GPU access** (T4, ~16GB VRAM) — ideal for running larger HuggingFace models.

### Colab Setup Steps

```python
# Step 1: Check GPU availability
import torch
print(torch.cuda.is_available())           # Should print True
print(torch.cuda.get_device_name(0))       # e.g. "Tesla T4"

# Step 2: Install dependencies
# !pip install transformers accelerate bitsandbytes -q

# Step 3: Load model on GPU
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    device=0,            # GPU index (0 = first GPU)
    torch_dtype=torch.float16,   # Half precision to save VRAM
)

result = pipe("Explain quantum entanglement simply:", max_new_tokens=200)
print(result[0]["generated_text"])
```

### Colab Tips

| Tip | Details |
|---|---|
| **Runtime → Change runtime type** | Select GPU (T4) before running |
| **Use `float16`** | Halves VRAM usage with minimal quality loss |
| **Cache models in Drive** | Mount Google Drive and set `cache_dir="/content/drive/MyDrive/hf_cache"` |
| **Free Colab limits** | ~12h sessions, ~12GB RAM, ~16GB GPU VRAM |
| **Colab Pro** | Longer sessions, A100 GPU access |

### Mounting Google Drive (to persist downloaded models)

```python
from google.colab import drive
drive.mount("/content/drive")

from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="gpt2",
    model_kwargs={"cache_dir": "/content/drive/MyDrive/hf_models"}
)
```

---

## 8. The `AutoClass` Pattern

HuggingFace's `Auto` classes automatically select the correct model class based on the model name:

```python
from transformers import (
    AutoTokenizer,
    AutoModel,                  # Base model, returns hidden states
    AutoModelForCausalLM,       # For text generation (GPT-style)
    AutoModelForSeq2SeqLM,      # For T5/BART (encoder-decoder)
    AutoModelForSequenceClassification,   # For classification
    AutoModelForTokenClassification,      # For NER
    AutoModelForQuestionAnswering,        # For extractive QA
)

# Just change the model name — same API for all
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```

---

## 9. Saving & Loading Models Locally

Download once, reuse offline — no internet required after the first download.

```python
# Save to disk
tokenizer.save_pretrained("./my_model/")
model.save_pretrained("./my_model/")

# Load from disk
tokenizer = AutoTokenizer.from_pretrained("./my_model/")
model     = AutoModelForCausalLM.from_pretrained("./my_model/")
```

> **Tip:** HuggingFace also caches models automatically in `~/.cache/huggingface/hub/` after the first download.

---

## 10. Hands-On Scripts

The following Python files are provided in `Day_3/`:

| File | What It Demonstrates |
|---|---|
| `01_tokenization.py` | Tokenize text, inspect tokens, decode back, compare tokenizers |
| `02_pipeline_basics.py` | Run 4 different pipeline tasks (generation, sentiment, NER, summarization) |
| `03_text_generation.py` | Deep-dive into generation strategies (greedy, beam, sampling, top-k/p) |
| `04_local_model.py` | Load model + tokenizer manually, generate with `AutoModelForCausalLM` |
| `05_colab_demo.py` | Colab-ready script: GPU check, float16 inference, Drive caching |

Run any script:
```bash
cd Day_3
python 01_tokenization.py
python 02_pipeline_basics.py
```

---

## Summary

| Concept | Key Takeaway |
|---|---|
| **Tokenization** | Converts text → token IDs; every model has its own tokenizer |
| **Sub-word Tokenization** | Handles unknown words by splitting into sub-pieces (BPE, WordPiece) |
| **`pipeline()`** | Easiest API — 3 lines to run any HuggingFace model |
| **Text Generation** | Decoder-only (GPT-style); use sampling + temperature for natural output |
| **AutoClasses** | `AutoTokenizer`, `AutoModelForCausalLM` — model-agnostic loading |
| **Local Execution** | Works on CPU; GPU needed for large models (>3B params) |
| **Google Colab** | Free T4 GPU; use `device=0` and `torch_dtype=torch.float16` |
| **Model Caching** | Downloaded once, cached in `~/.cache/huggingface/hub/` |

---

## Further Reading

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [HuggingFace Course (free)](https://huggingface.co/learn/nlp-course)
- [Text Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)