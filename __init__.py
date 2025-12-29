"""
Benchmarking suite for vLLM servers.

Supports:
- STT (Speech-to-Text) benchmarks for audio transcription models
- LLM (Large Language Model) benchmarks for chat completion models

Usage:
    python -m bechmarks stt --url https://... --model voxtral-mini --endpoint chat
    python -m bechmarks llm --url https://... --model llama-3.1-8b
"""

__version__ = "0.1.0"

