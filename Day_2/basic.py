"""
Day 2 – Prompting Techniques (Overview)
========================================
This file demonstrates the base client setup for Azure AI Inference.
Each prompting technique has its own dedicated file:

    zero_shot.py        – Zero-Shot Prompting examples
    few_shot.py         – Few-Shot Prompting examples
    chain_of_thought.py – Chain-of-Thought (CoT) Prompting examples
    role_prompting.py   – Role / Persona Prompting examples

Run any of them directly:
    python zero_shot.py
    python few_shot.py
    python chain_of_thought.py
    python role_prompting.py
"""

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
load_dotenv()

# ── Client Setup (shared pattern across all technique files) ──────────────────
endpoint = "https://models.github.ai/inference"
model    = "deepseek/DeepSeek-R1-0528"
token    = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# ── Quick sanity-check call ───────────────────────────────────────────────────
response = client.complete(
    messages=[
        UserMessage("What is the capital of France?"),
    ],
    max_tokens=50,
    model=model,
)

print("Sanity check – Capital of France:")
print(response.choices[0].message.content)

