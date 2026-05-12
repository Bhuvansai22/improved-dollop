# Day 2 – Prompting Techniques

Effective prompting is the backbone of working with Large Language Models (LLMs). Understanding different prompting strategies helps you get better, more accurate, and more structured outputs from models like Gemini, GPT, Claude, etc.

---

## 1. Zero-Shot Prompting

### What is it?
Zero-shot prompting means giving the model a task **without any examples**. You simply describe what you want, and the model uses its pre-trained knowledge to respond.

### When to use?
- Simple, well-understood tasks
- When you want a quick response without setup
- Tasks the model has been extensively trained on (e.g., translation, summarization)

### Example

**Prompt:**
```
Classify the sentiment of the following sentence as Positive, Negative, or Neutral.

Sentence: "The movie was absolutely fantastic!"
```

**Response:**
```
Positive
```

### Key Insight
The model relies entirely on what it learned during training. No task-specific examples are provided.

---

## 2. Few-Shot Prompting

### What is it?
Few-shot prompting provides the model with **a small number of input-output examples** (typically 2–5) before asking it to perform the actual task. This helps the model understand the expected format and behavior.

### When to use?
- When zero-shot gives inconsistent or incorrect results
- When you need a specific output format
- For domain-specific or nuanced tasks

### Example

**Prompt:**
```
Classify the sentiment of the following sentences.

Sentence: "I love this product!" → Positive
Sentence: "This is the worst experience ever." → Negative
Sentence: "The package arrived on time." → Neutral

Sentence: "I'm not sure how I feel about this update."
```

**Response:**
```
Neutral
```

### Key Insight
The examples act as **in-context learning** — the model adapts its behavior based on the patterns shown in the prompt itself, without any weight updates.

---

## 3. Chain-of-Thought (CoT) Prompting

### What is it?
Chain-of-Thought prompting encourages the model to **reason step by step** before arriving at a final answer. Instead of jumping to a conclusion, the model "thinks aloud" through intermediate reasoning steps.

### When to use?
- Math problems and logical reasoning
- Multi-step decision making
- Complex tasks where intermediate steps matter
- Reducing hallucinations on hard questions

### Types

| Type | Description |
|---|---|
| **Zero-Shot CoT** | Add "Let's think step by step." to the prompt |
| **Few-Shot CoT** | Provide examples that include step-by-step reasoning |

### Example – Zero-Shot CoT

**Prompt:**
```
A train travels 60 km/h for 2.5 hours. How far does it travel?
Let's think step by step.
```

**Response:**
```
Step 1: Speed = 60 km/h
Step 2: Time = 2.5 hours
Step 3: Distance = Speed × Time = 60 × 2.5 = 150 km

Answer: 150 km
```

### Example – Few-Shot CoT

**Prompt:**
```
Q: If there are 5 apples and you take away 3, how many do you have?
A: You took away 3 apples, so you personally have 3 apples.

Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?
A:
```

**Response:**
```
Roger starts with 5 balls. He buys 2 cans × 3 balls = 6 more balls.
Total = 5 + 6 = 11 tennis balls.
```

### Key Insight
CoT dramatically improves accuracy on reasoning-heavy tasks. The phrase **"Let's think step by step"** is a powerful zero-shot trigger.

---

## 4. Role Prompting

### What is it?
Role prompting (also called **system prompting** or **persona prompting**) assigns the model a specific **identity, role, or persona** before giving it a task. This shapes the model's tone, style, expertise level, and perspective.

### When to use?
- When you need domain-specific expertise (e.g., act as a doctor, lawyer, teacher)
- To control tone and formality (e.g., professional, friendly, concise)
- To simulate conversations or interviews
- Building chatbots with a specific personality

### Example

**Prompt:**
```
You are an experienced Python developer and software architect.
Explain the difference between a list and a tuple in Python to a beginner.
```

**Response:**
```
Great question! Think of a list like a shopping cart — you can add, remove, or change items freely. A tuple, on the other hand, is like a receipt — once it's printed, you can't change it.

- List: mutable (changeable), uses []
- Tuple: immutable (fixed), uses ()

Use a list when data might change, and a tuple when it should stay constant.
```

### Combining Role + Task

```
You are a strict JSON API. Respond only with valid JSON and no extra text.

Extract the name and age from this sentence:
"My name is Alice and I am 30 years old."
```

**Response:**
```json
{"name": "Alice", "age": 30}
```

### Key Insight
Role prompting is especially powerful in **system messages** (e.g., in OpenAI's `system` role or Gemini's system instructions). It sets the context for the entire conversation.

---

## Summary Comparison

| Technique | Examples Needed | Best For | Complexity |
|---|---|---|---|
| **Zero-Shot** | None | Simple, general tasks | Low |
| **Few-Shot** | 2–5 | Format control, niche tasks | Medium |
| **Chain-of-Thought** | Optional | Reasoning, math, logic | Medium–High |
| **Role Prompting** | None | Persona, tone, domain expertise | Low–Medium |

---

## Tips for Combining Techniques

You can combine these techniques for powerful results:

```
You are an expert data scientist. (Role)

Classify the following customer reviews as Positive, Negative, or Neutral.

Review: "Great product, fast delivery!" → Positive  (Few-Shot example)
Review: "Terrible quality, broke in a week." → Negative

Now classify this review step by step: (Chain-of-Thought)
Review: "It's okay, nothing special but does the job."
```

This approach — **Role + Few-Shot + CoT** — gives you maximum control over the model's output quality and format.
