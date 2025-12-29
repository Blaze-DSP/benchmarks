#!/usr/bin/env python3
"""
Common entrypoint for benchmark scripts.

Usage:
    python -m bechmarks stt --url https://... --model voxtral-mini --endpoint chat
    python -m bechmarks llm --url https://... --model llama-3.1-8b
"""

import argparse
import asyncio


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared between all benchmark types."""
    parser.add_argument("--url", type=str, required=True, help="Base URL for the API")
    parser.add_argument(
        "--api-key", type=str, default="DUMMY", help="API key for authentication"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name to use")
    parser.add_argument(
        "--max-concurrent", type=int, default=10, help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--total-requests", type=int, default=100, help="Total requests to run"
    )
    parser.add_argument(
        "--ramp-start", type=int, default=0, help="Starting concurrency for ramp-up"
    )
    parser.add_argument(
        "--ramp-step", type=int, default=0, help="Step size for ramp-up"
    )
    parser.add_argument(
        "--num-warmups", type=int, default=5, help="Number of warmup requests"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--dataset", type=str, required=True, help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )


def add_stt_args(parser: argparse.ArgumentParser) -> None:
    """Add STT-specific arguments."""
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        choices=["transcriptions", "chat"],
        help="Endpoint type: 'transcriptions' for /v1/audio/transcriptions, 'chat' for /v1/chat/completions",
    )
    parser.add_argument(
        "--prompt", type=str, default="Transcribe the given audio in appropriate language.", help="Optional prompt for transcription"
    )


def add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Add LLM-specific arguments."""
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens per response"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m bechmarks",
        description="Benchmarking suite for vLLM servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark STT with transcriptions endpoint
  python -m bechmarks stt \\
      --url https://voxtral-mini.models.example.com/v1 \\
      --model voxtral-mini \\
      --endpoint transcriptions \\
      --max-concurrent 10 \\
      --total-requests 100

  # Benchmark STT with chat endpoint (audio input)
  python -m bechmarks stt \\
      --url https://qwen3-omni.models.example.com/v1 \\
      --model qwen3-omni \\
      --endpoint chat \\
      --max-concurrent 20 \\
      --total-requests 100 \\
      --ramp-start 5 \\
      --ramp-step 5

  # Benchmark LLM with multi-turn conversations
  python -m bechmarks llm \\
      --url https://llama.models.example.com/v1 \\
      --model llama-3.1-8b \\
      --max-concurrent 10 \\
      --total-requests 100
        """,
    )

    subparsers = parser.add_subparsers(
        dest="benchmark_type",
        title="benchmark types",
        description="Available benchmark types",
        required=True,
    )

    # STT subcommand
    stt_parser = subparsers.add_parser(
        "stt",
        help="Benchmark Audio STT endpoints (transcriptions or chat with audio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(stt_parser)
    add_stt_args(stt_parser)

    # LLM subcommand
    llm_parser = subparsers.add_parser(
        "llm",
        help="Benchmark LLM chat completions with multi-turn conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(llm_parser)
    add_llm_args(llm_parser)

    return parser


async def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.benchmark_type == "stt":
        from .benchmarks.stt import run_stt_benchmark

        await run_stt_benchmark(args)
    elif args.benchmark_type == "llm":
        from .benchmarks.llm import run_llm_benchmark

        await run_llm_benchmark(args)


if __name__ == "__main__":
    asyncio.run(main())
