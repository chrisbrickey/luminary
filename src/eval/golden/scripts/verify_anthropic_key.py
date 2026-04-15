"""Minimal verification script to check Anthropic API key works.

Usage:
    uv run python src/eval/golden/scripts/verify_anthropic_key.py
"""

import os

from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types import TextBlock

# Load environment variables from .env file
load_dotenv()

# Get API key
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("❌ ANTHROPIC_API_KEY not found in environment")
    print("Make sure you have a .env file with ANTHROPIC_API_KEY=your-key-here")
    exit(1)

print(f"✓ API key found (starts with: {api_key[:20]}...)")
print("\nTesting Anthropic API connection...")
print("-" * 50)

# Try multiple models to find one that works
models_to_try = [
    "claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5
    "claude-3-7-sonnet-20250219",  # Sonnet 3.7
    "claude-3-5-sonnet-20250219",  # Sonnet 3.5 Feb 2025
    "claude-3-opus-20240229",      # Older Opus
    "claude-3-sonnet-20240229",    # Older Sonnet
]

client = Anthropic(api_key=api_key)

for model in models_to_try:
    try:
        print(f"\nTrying model: {model}")
        message = client.messages.create(
            model=model,
            max_tokens=20,
            messages=[
                {"role": "user", "content": "Say 'hello' in one word"}
            ]
        )

        # Print success
        print(f"\n✅ SUCCESS! Model {model} works!")

        # Extract text from response (type narrowing for mypy)
        response_text = "N/A"
        if message.content and isinstance(message.content[0], TextBlock):
            response_text = message.content[0].text

        print(f"Response: {response_text}")
        print(f"\nUsage:")
        print(f"  - Input tokens: {message.usage.input_tokens}")
        print(f"  - Output tokens: {message.usage.output_tokens}")
        print(f"\n*** Use this model: {model} ***")
        exit(0)

    except Exception as e:
        print(f"  ❌ {model} failed: {e}")
        continue

print("\n❌ All models failed. Check API key permissions or account status.")
exit(1)
