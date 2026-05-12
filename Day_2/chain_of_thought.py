"""
Chain-of-Thought (CoT) Prompting Examples
==========================================
Chain-of-Thought prompting encourages the model to reason step by step
before arriving at a final answer. Instead of jumping to a conclusion,
the model "thinks aloud" through intermediate reasoning steps.

Types:
  - Zero-Shot CoT : Add "Let's think step by step." to the prompt
  - Few-Shot CoT  : Provide examples that include step-by-step reasoning

Best for:
- Math problems and logical reasoning
- Multi-step decision making
- Reducing hallucinations on hard questions
"""

import os
from dotenv import load_dotenv
load_dotenv()
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"
model = "deepseek/DeepSeek-R1-0528"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# ─────────────────────────────────────────────────────────────────────────────
# Example 1: Zero-Shot CoT — Math problem
# Magic trigger: "Let's think step by step."
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 1: Zero-Shot CoT — Distance Calculation")
print("=" * 60)

response = client.complete(
    messages=[
        UserMessage(
            "A train travels 60 km/h for 2.5 hours. How far does it travel?\n"
            "Let's think step by step."
        ),
    ],
    max_tokens=300,
    model=model,
)
print("Prompt: Train distance problem with 'Let's think step by step.'")
print("Response:\n", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 2: Zero-Shot CoT — Word problem
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 2: Zero-Shot CoT — Word Problem")
print("=" * 60)

response = client.complete(
    messages=[
        UserMessage(
            "If a shirt costs $25 and is on a 20% discount, what is the final price?\n"
            "Let's think step by step."
        ),
    ],
    max_tokens=300,
    model=model,
)
print("Prompt: Discount calculation with 'Let's think step by step.'")
print("Response:\n", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 3: Few-Shot CoT — Provide example with reasoning, then ask new Q
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 3: Few-Shot CoT — Tennis Balls Problem")
print("=" * 60)

few_shot_cot_prompt = """Q: If there are 5 apples and you take away 3, how many do you have?
A: You took away 3 apples, so you personally have 3 apples.

Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?
A:"""

response = client.complete(
    messages=[
        UserMessage(few_shot_cot_prompt),
    ],
    max_tokens=200,
    model=model,
)
print("Prompt: Few-shot CoT with one example of step-by-step reasoning.")
print("Response:\n", response.choices[0].message.content)
print()

# ─────────────────────────────────────────────────────────────────────────────
# Example 4: Few-Shot CoT — Logical Reasoning
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Example 4: Few-Shot CoT — Logical Reasoning")
print("=" * 60)

few_shot_logic_prompt = """Q: All mammals are warm-blooded. Dogs are mammals. Is a dog warm-blooded?
A: Step 1 – All mammals are warm-blooded (given).
   Step 2 – A dog is a mammal (given).
   Step 3 – Therefore, a dog is warm-blooded.
   Answer: Yes.

Q: All fruits contain seeds. An apple is a fruit. Does an apple contain seeds?
A:"""

response = client.complete(
    messages=[
        UserMessage(few_shot_logic_prompt),
    ],
    max_tokens=200,
    model=model,
)
print("Prompt: Few-shot CoT with logical syllogism example.")
print("Response:\n", response.choices[0].message.content)
