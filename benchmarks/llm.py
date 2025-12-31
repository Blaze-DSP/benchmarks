"""
LLM Chat Completions Benchmark Implementation.

Benchmarks /v1/chat/completions endpoint with multi-turn conversations.
Uses semaphore-based concurrency control with conversation chains to ensure:
1. Exact concurrency is maintained at all times
2. No two requests from the same conversation run concurrently
3. Turns within a conversation execute sequentially
"""

import asyncio
import json
import random
import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ..common import (
    PercentileStats,
    TokenStats,
    compute_tpot_array,
    print_footer,
    print_header,
)


# =============================================================================
# Data Models
# =============================================================================


class PrebuiltRequest(BaseModel):
    """A pre-built request with full context baked in."""

    conversation_id: str
    turn_index: int
    messages: list[dict]  # Full context + current turn (pre-built)


class TurnResult(BaseModel):
    """Result of a single turn in a conversation."""

    request_id: str
    conversation_id: str
    turn_index: int
    start_time: float
    end_time: float
    latency_ms: float
    ttft_ms: float
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    response: Optional[str] = None


class LLMSummaryStats(BaseModel):
    """LLM summary statistics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_conversations: int
    total_time_sec: float


class LLMThroughputStats(BaseModel):
    """LLM throughput statistics."""

    requests_per_sec: float
    total_input_tokens: int
    total_output_tokens: int
    input_tokens_per_sec: float
    output_tokens_per_sec: float


class LLMBenchmarkStatsOutput(BaseModel):
    """Structured output for LLM benchmark stats."""

    summary: LLMSummaryStats
    throughput: LLMThroughputStats
    latency_ms: PercentileStats
    ttft_ms: PercentileStats
    tpot_ms: PercentileStats
    errors: dict = Field(default_factory=dict)


class LLMBenchmarkStats:
    """LLM benchmark statistics with conversation count."""

    def __init__(
        self,
        total_requests: int,
        successful_requests: int,
        failed_requests: int,
        total_conversations: int,
        total_time_sec: float,
        throughput_rps: float,
        tokens: TokenStats,
        latency: PercentileStats,
        ttft: PercentileStats,
        tpot: PercentileStats,
        errors: dict = None,
    ):
        self.total_requests = total_requests
        self.successful_requests = successful_requests
        self.failed_requests = failed_requests
        self.total_conversations = total_conversations
        self.total_time_sec = total_time_sec
        self.throughput_rps = throughput_rps
        self.tokens = tokens
        self.latency = latency
        self.ttft = ttft
        self.tpot = tpot
        self.errors = errors or {}

    def to_output(self) -> LLMBenchmarkStatsOutput:
        """Convert to structured output model."""
        return LLMBenchmarkStatsOutput(
            summary=LLMSummaryStats(
                total_requests=self.total_requests,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                total_conversations=self.total_conversations,
                total_time_sec=self.total_time_sec,
            ),
            throughput=LLMThroughputStats(
                requests_per_sec=self.throughput_rps,
                total_input_tokens=self.tokens.total_input_tokens,
                total_output_tokens=self.tokens.total_output_tokens,
                input_tokens_per_sec=self.tokens.input_tokens_per_sec,
                output_tokens_per_sec=self.tokens.output_tokens_per_sec,
            ),
            latency_ms=self.latency,
            ttft_ms=self.ttft,
            tpot_ms=self.tpot,
            errors=self.errors,
        )


# =============================================================================
# Dataset Loading and Preprocessing
# =============================================================================


def load_and_prepare_dataset(dataset_name: str, split: str = "train") -> pd.DataFrame:
    """Load and prepare the conversation dataset."""
    print(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split)
    return dataset.to_pandas()


def parse_messages(messages_data) -> list[dict]:
    """Parse messages from various formats into a standard list of dicts."""
    if isinstance(messages_data, list):
        return messages_data
    elif isinstance(messages_data, str):
        try:
            return json.loads(messages_data)
        except json.JSONDecodeError:
            return []
    return []


def count_turns(conversation: list[dict]) -> int:
    """Count the number of complete turns (assistant + users) in a conversation.

    A turn is one assistant message followed by one or more user messages.
    """
    turn_count = 0
    has_assistant = False
    has_users = False

    for msg in conversation:
        role = msg.get("role")
        if role == "assistant":
            if has_assistant and has_users:
                turn_count += 1
            has_assistant = True
            has_users = False
        elif role == "user" and has_assistant:
            has_users = True

    if has_assistant and has_users:
        turn_count += 1

    return turn_count


def extract_turns(conversation: list[dict]) -> tuple[Optional[dict], list[dict]]:
    """Extract system message and turns from a conversation.

    Returns:
        Tuple of (system_message, list_of_turns)
        where each turn is {"assistant": msg, "users": [msgs]}
    """
    system_msg = None
    turns = []
    current_turn = None

    for msg in conversation:
        role = msg.get("role")
        if role == "system":
            system_msg = msg
        elif role == "assistant":
            if current_turn and current_turn["users"]:
                turns.append(current_turn)
            current_turn = {"assistant": msg, "users": []}
        elif role == "user" and current_turn:
            current_turn["users"].append(msg)

    if current_turn and current_turn["users"]:
        turns.append(current_turn)

    return system_msg, turns


def build_prebuilt_requests(
    conversation: list[dict],
    conversation_id: str,
    max_turns: int,
) -> list[PrebuiltRequest]:
    """Build pre-built requests for a conversation with full context baked in.

    Context for turn N includes the original assistant content from turns 0..N-1
    as simulated responses. This allows deterministic pre-building while maintaining
    realistic context lengths.
    """
    system_msg, turns = extract_turns(conversation)

    # Truncate to max_turns
    turns = turns[:max_turns]

    requests = []
    context = []  # Accumulated context from previous turns

    for turn_index, turn in enumerate(turns):
        # Build messages: system + context + current turn
        messages = []
        if system_msg:
            messages.append(system_msg)
        messages.extend(context)
        messages.append(turn["assistant"])
        messages.extend(turn["users"])

        requests.append(
            PrebuiltRequest(
                conversation_id=conversation_id,
                turn_index=turn_index,
                messages=messages,
            )
        )

        # Add current turn to context for next turn
        # Use the original assistant content as "simulated response"
        context.append(turn["assistant"])
        context.extend(turn["users"])
        # Add assistant's content as the response (from dataset)
        assistant_content = turn["assistant"].get("content", "")
        if assistant_content:
            context.append({"role": "assistant", "content": assistant_content})

    return requests


def prepare_benchmark_requests(
    conversations: list[list[dict]],
    max_turns: int,
    max_concurrent: int,
    num_conversations: Optional[int] = None,
    total_requests: Optional[int] = None,
    seed: int = 42,
) -> tuple[list[list[PrebuiltRequest]], int, int]:
    """Prepare all benchmark requests grouped by conversation.

    Args:
        conversations: List of raw conversations from dataset
        max_turns: Maximum turns per conversation (all truncated to this)
        max_concurrent: Maximum concurrent requests (for validation)
        num_conversations: Target number of conversations (optional)
        total_requests: Target total requests (optional, for validation)
        seed: Random seed for shuffling

    Returns:
        Tuple of (grouped_requests, actual_num_conversations, actual_total_requests)
        where grouped_requests[i] is a list of PrebuiltRequests for conversation i

    Raises:
        ValueError: If constraints cannot be satisfied
    """
    random.seed(seed)

    # Filter conversations that have enough turns
    valid_conversations = []
    for conv in conversations:
        if count_turns(conv) >= max_turns:
            valid_conversations.append(conv)

    if len(valid_conversations) == 0:
        raise ValueError(
            f"No conversations found with >= {max_turns} turns. "
            f"Try lowering --max-turns."
        )

    print(f"Found {len(valid_conversations)} conversations with >= {max_turns} turns")

    # Shuffle for randomness
    random.shuffle(valid_conversations)

    # Determine number of conversations to use
    if num_conversations is not None:
        if num_conversations > len(valid_conversations):
            raise ValueError(
                f"Requested {num_conversations} conversations but only "
                f"{len(valid_conversations)} have >= {max_turns} turns. "
                f"Try lowering --max-turns or --num-conversations."
            )
        selected_conversations = valid_conversations[:num_conversations]
    else:
        selected_conversations = valid_conversations

    actual_num_convs = len(selected_conversations)
    actual_total_requests = actual_num_convs * max_turns

    # Validation
    if actual_num_convs < max_concurrent:
        raise ValueError(
            f"Need at least {max_concurrent} conversations to maintain "
            f"{max_concurrent} concurrency, but only have {actual_num_convs}. "
            f"Try lowering --max-concurrent or --max-turns."
        )

    if total_requests is not None and actual_total_requests != total_requests:
        raise ValueError(
            f"--total-requests={total_requests} doesn't match computed total: "
            f"{actual_num_convs} conversations × {max_turns} turns = {actual_total_requests}. "
            f"Either adjust --num-conversations or remove --total-requests."
        )

    # Warn if ratio is low
    recommended_convs = int(max_concurrent * 1.5)
    if actual_num_convs < recommended_convs:
        drop_point = (actual_num_convs - max_concurrent) * max_turns
        sustained_pct = (drop_point / actual_total_requests) * 100 if actual_total_requests > 0 else 0
        print(
            f"⚠️  Warning: {actual_num_convs} conversations may not sustain "
            f"{max_concurrent} concurrency until end.\n"
            f"   Full concurrency sustained for ~{sustained_pct:.0f}% of requests.\n"
            f"   Recommendation: Use >= {recommended_convs} conversations "
            f"(--max-turns={actual_total_requests // recommended_convs} with "
            f"--num-conversations={recommended_convs})."
        )

    # Build all requests
    grouped_requests = []
    for conv in selected_conversations:
        conv_id = str(uuid.uuid4())[:8]
        requests = build_prebuilt_requests(conv, conv_id, max_turns)
        grouped_requests.append(requests)

    return grouped_requests, actual_num_convs, actual_total_requests


# =============================================================================
# Benchmark Runner
# =============================================================================


class LLMBenchmark:
    """Benchmark runner for LLM chat completions with semaphore-based concurrency."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "DUMMY",
        max_tokens: int = 256,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens

        self.client = AsyncOpenAI(base_url=self.url, api_key=self.api_key)
        self.results: list[TurnResult] = []

    async def _stream_request(
        self,
        request: PrebuiltRequest,
    ) -> TurnResult:
        """Make a streaming request to measure TTFT and collect tokens."""
        start_time = time.perf_counter()
        ttft = 0.0
        first_token_received = False
        collected_content = ""
        input_tokens = 0
        output_tokens = 0

        request_id = f"{request.conversation_id}-t{request.turn_index}"

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=request.messages,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if not first_token_received and chunk.choices:
                    if chunk.choices[0].delta.content:
                        ttft = (time.perf_counter() - start_time) * 1000
                        first_token_received = True

                if chunk.choices and chunk.choices[0].delta.content:
                    collected_content += chunk.choices[0].delta.content

                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

            end_time = time.perf_counter()

            return TurnResult(
                request_id=request_id,
                conversation_id=request.conversation_id,
                turn_index=request.turn_index,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                ttft_ms=ttft,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response=collected_content,
            )

        except Exception as e:
            end_time = time.perf_counter()
            return TurnResult(
                request_id=request_id,
                conversation_id=request.conversation_id,
                turn_index=request.turn_index,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                ttft_ms=0,
                success=False,
                error=str(e),
            )

    async def warmup(
        self, grouped_requests: list[list[PrebuiltRequest]], num_warmups: int = 3
    ) -> None:
        """Run warmup requests (not counted in metrics)."""
        if num_warmups <= 0:
            return

        print(f"\nRunning {num_warmups} warmup requests...")

        # Pick random first-turn requests for warmup
        first_turn_requests = [chain[0] for chain in grouped_requests if chain]

        for i in range(min(num_warmups, len(first_turn_requests))):
            request = random.choice(first_turn_requests)
            result = await self._stream_request(request)
            status = "✓" if result.success else "✗"
            print(
                f"  Warmup {i + 1}/{num_warmups}: {status} "
                f"(latency: {result.latency_ms:.0f}ms, ttft: {result.ttft_ms:.0f}ms)"
            )

    async def run_benchmark(
        self,
        grouped_requests: list[list[PrebuiltRequest]],
        max_concurrent: int,
        ramp_start: int = 0,
        ramp_step: int = 0,
    ) -> LLMBenchmarkStats:
        """Run the benchmark with round-robin request scheduling.

        Uses a centralized queue to ensure requests are picked from different
        conversations, maintaining even distribution and preventing any single
        conversation from exhausting early.

        Supports optional ramp-up from ramp_start to max_concurrent.
        """
        self.results = []

        total_convs = len(grouped_requests)
        total_requests = sum(len(chain) for chain in grouped_requests)

        use_ramp_up = ramp_start > 0 and ramp_step > 0 and ramp_start < max_concurrent

        print(f"\nRunning benchmark:")
        print(f"  Conversations: {total_convs}")
        print(f"  Total requests: {total_requests}")
        print(f"  Max concurrent: {max_concurrent}")
        print(f"  Turns per conversation: {total_requests // total_convs}")
        if use_ramp_up:
            print(f"  Ramp-up: {ramp_start} -> {max_concurrent} (step: {ramp_step})")

        # Track state for each conversation
        # conv_state[i] = next turn index to process for conversation i
        conv_state = [0] * total_convs
        conv_complete = [False] * total_convs
        max_turns = len(grouped_requests[0]) if grouped_requests else 0

        # Concurrency control
        current_concurrency = ramp_start if use_ramp_up else max_concurrent
        completed_requests = 0
        completed_convs = 0
        requests_at_last_ramp = 0
        benchmark_start = time.perf_counter()

        # Build initial request queue: round-robin across conversations
        # Queue contains (conv_index, turn_index) pairs
        request_queue = []
        for turn_idx in range(max_turns):
            for conv_idx in range(total_convs):
                request_queue.append((conv_idx, turn_idx))

        # Track which conversations have in-flight requests
        conv_in_flight = [False] * total_convs
        active_tasks = {}  # task -> (conv_idx, turn_idx)

        def get_next_request():
            """Get next request ensuring no two from same conversation run concurrently."""
            for i, (conv_idx, turn_idx) in enumerate(request_queue):
                # Skip if this conversation already has a request in flight
                if conv_in_flight[conv_idx]:
                    continue
                # Skip if this turn isn't ready yet (previous turn not complete)
                if turn_idx > conv_state[conv_idx]:
                    continue
                # Found a valid request
                request_queue.pop(i)
                return conv_idx, turn_idx
            return None, None

        # Start initial batch of requests
        for _ in range(min(current_concurrency, len(request_queue))):
            conv_idx, turn_idx = get_next_request()
            if conv_idx is not None:
                conv_in_flight[conv_idx] = True
                request = grouped_requests[conv_idx][turn_idx]
                task = asyncio.create_task(self._stream_request(request))
                active_tasks[task] = (conv_idx, turn_idx)

        # Process requests as they complete
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                result = await task
                self.results.append(result)
                completed_requests += 1

                conv_idx, turn_idx = active_tasks.pop(task)
                conv_in_flight[conv_idx] = False
                conv_state[conv_idx] = turn_idx + 1  # Mark turn as complete

                # Check if conversation is fully complete
                if conv_state[conv_idx] >= max_turns and not conv_complete[conv_idx]:
                    conv_complete[conv_idx] = True
                    completed_convs += 1

                # Ramp-up logic
                if use_ramp_up and current_concurrency < max_concurrent:
                    requests_since_ramp = completed_requests - requests_at_last_ramp
                    if requests_since_ramp >= current_concurrency:
                        old_concurrency = current_concurrency
                        current_concurrency = min(current_concurrency + ramp_step, max_concurrent)
                        requests_at_last_ramp = completed_requests
                        print(f"  Ramping up: {old_concurrency} -> {current_concurrency} concurrent")

                        # Start additional requests up to new concurrency
                        while len(active_tasks) < current_concurrency:
                            next_conv, next_turn = get_next_request()
                            if next_conv is None:
                                break
                            conv_in_flight[next_conv] = True
                            request = grouped_requests[next_conv][next_turn]
                            new_task = asyncio.create_task(self._stream_request(request))
                            active_tasks[new_task] = (next_conv, next_turn)

                # Start next request if slot available
                while len(active_tasks) < current_concurrency:
                    next_conv, next_turn = get_next_request()
                    if next_conv is None:
                        break
                    conv_in_flight[next_conv] = True
                    request = grouped_requests[next_conv][next_turn]
                    new_task = asyncio.create_task(self._stream_request(request))
                    active_tasks[new_task] = (next_conv, next_turn)

                # Progress reporting
                if completed_requests % 10 == 0 or completed_requests == total_requests:
                    elapsed = time.perf_counter() - benchmark_start
                    print(
                        f"  Progress: {completed_requests}/{total_requests} requests, "
                        f"{completed_convs}/{total_convs} conversations done, "
                        f"{len(active_tasks)}/{current_concurrency} active ({elapsed:.1f}s)"
                    )

        total_time = time.perf_counter() - benchmark_start
        return self._compute_stats(total_time, total_convs)

    def _compute_stats(
        self, total_time: float, total_conversations: int
    ) -> LLMBenchmarkStats:
        """Compute aggregated statistics from results."""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        errors = {}
        for r in failed:
            error_key = r.error[:50] if r.error else "Unknown"
            errors[error_key] = errors.get(error_key, 0) + 1

        if not successful:
            return LLMBenchmarkStats(
                total_requests=len(self.results),
                successful_requests=0,
                failed_requests=len(failed),
                total_conversations=total_conversations,
                total_time_sec=total_time,
                throughput_rps=0,
                tokens=TokenStats(),
                latency=PercentileStats(),
                ttft=PercentileStats(),
                tpot=PercentileStats(),
                errors=errors,
            )

        total_input = sum(r.input_tokens for r in successful)
        total_output = sum(r.output_tokens for r in successful)

        latencies = np.array([r.latency_ms for r in successful])
        ttfts = np.array([r.ttft_ms for r in successful if r.ttft_ms > 0])
        tpots_arr = compute_tpot_array(successful)

        return LLMBenchmarkStats(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_conversations=total_conversations,
            total_time_sec=total_time,
            throughput_rps=len(successful) / total_time if total_time > 0 else 0,
            tokens=TokenStats.compute(total_input, total_output, total_time),
            latency=PercentileStats.from_array(latencies),
            ttft=PercentileStats.from_array(ttfts),
            tpot=PercentileStats.from_array(tpots_arr),
            errors=errors,
        )


# =============================================================================
# Output and Reporting
# =============================================================================


def print_stats(stats: LLMBenchmarkStats, args) -> None:
    """Pretty print benchmark statistics."""
    print_header("BENCHMARK RESULTS")

    print("\nConfiguration:")
    print(f"  URL:              {args.url}")
    print(f"  Model:            {args.model}")
    print(f"  Conversations:    {stats.total_conversations}")
    print(f"  Total Requests:   {stats.total_requests}")
    print(f"  Max Concurrent:   {args.max_concurrent}")
    print(f"  Turns/Conv:       {args.max_turns}")
    
    ramp_start = getattr(args, 'ramp_start', 0)
    ramp_step = getattr(args, 'ramp_step', 0)
    if ramp_start > 0 and ramp_step > 0:
        print(f"  Ramp-up:          {ramp_start} -> {args.max_concurrent} (step: {ramp_step})")

    success_rate = (
        100 * stats.successful_requests / stats.total_requests
        if stats.total_requests > 0
        else 0
    )

    print("\nResults:")
    print(
        f"  Successful:       {stats.successful_requests}/{stats.total_requests} "
        f"({success_rate:.1f}%)"
    )
    print(f"  Failed:           {stats.failed_requests}")
    print(f"  Total Time:       {stats.total_time_sec:.2f}s")

    print("\nThroughput:")
    print(f"  Requests/sec:     {stats.throughput_rps:.2f}")
    stats.tokens.print_stats()

    print("\nE2E Latency (ms):")
    stats.latency.print_stats()

    print("\nTTFT - Time To First Token (ms):")
    stats.ttft.print_stats()

    print("\nTPOT - Time Per Output Token (ms):")
    stats.tpot.print_stats()

    if stats.errors:
        print("\nErrors:")
        for error, count in sorted(stats.errors.items(), key=lambda x: -x[1]):
            print(f"  [{count}x] {error}")

    print_footer()


class LLMBenchmarkConfig(BaseModel):
    """LLM benchmark configuration."""

    url: str
    model: str
    max_concurrent: int
    total_requests: int
    num_conversations: int
    max_turns: int
    ramp_start: int
    ramp_step: int
    num_warmups: int
    max_tokens: int
    dataset: str
    split: str
    seed: int
    api_key: str


class LLMBenchmarkOutput(BaseModel):
    """Complete LLM benchmark output."""

    benchmark_type: str = "llm"
    timestamp: str
    config: LLMBenchmarkConfig
    stats: LLMBenchmarkStatsOutput
    requests: list[TurnResult]


def export_results(
    stats: LLMBenchmarkStats,
    benchmark: LLMBenchmark,
    args,
    num_conversations: int,
    total_requests: int,
) -> str:
    """Export benchmark results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"benchmark_llm_{args.model}_{timestamp}.json"

    output = LLMBenchmarkOutput(
        timestamp=datetime.now().isoformat(),
        config=LLMBenchmarkConfig(
            url=args.url,
            model=args.model,
            max_concurrent=args.max_concurrent,
            total_requests=total_requests,
            num_conversations=num_conversations,
            max_turns=args.max_turns,
            ramp_start=getattr(args, 'ramp_start', 0),
            ramp_step=getattr(args, 'ramp_step', 0),
            num_warmups=args.num_warmups,
            max_tokens=args.max_tokens,
            dataset=args.dataset,
            split=args.split,
            seed=args.seed,
            api_key="***" if args.api_key != "DUMMY" else "DUMMY",
        ),
        stats=stats.to_output(),
        requests=benchmark.results,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    print(f"\nResults exported to: {output_path}")
    return output_path


# =============================================================================
# Main Entry Point
# =============================================================================


async def run_llm_benchmark(args) -> None:
    """Main entry point for LLM benchmark."""
    random.seed(args.seed)

    # Validate arguments
    if args.max_turns is None and args.num_conversations is None:
        raise ValueError(
            "Either --max-turns or --num-conversations must be specified.\n"
            "Examples:\n"
            "  --max-turns 5                    # Use 5 turns per conversation\n"
            "  --num-conversations 40 --total-requests 200  # Auto-calculate turns (200/40=5)"
        )

    # Calculate max_turns if not specified
    if args.max_turns is None:
        if args.total_requests is None or args.num_conversations is None:
            raise ValueError(
                "When --max-turns is not specified, both --total-requests and "
                "--num-conversations must be provided to calculate turns."
            )
        args.max_turns = args.total_requests // args.num_conversations
        if args.max_turns < 1:
            raise ValueError(
                f"Calculated max_turns={args.max_turns} is too low. "
                f"Increase --total-requests or decrease --num-conversations."
            )
        print(f"Auto-calculated --max-turns={args.max_turns} "
              f"({args.total_requests} requests / {args.num_conversations} conversations)")

    # Load dataset
    df = load_and_prepare_dataset(args.dataset, args.split)
    print(f"Loaded {len(df)} conversations from dataset")

    # Parse conversations
    conversations = []
    for _, row in df.iterrows():
        if "messages" not in row:
            continue
        messages = parse_messages(row["messages"])
        if messages and len(messages) >= 2:
            conversations.append(messages)

    if not conversations:
        print("Error: No valid conversations found in dataset")
        return

    print(f"Extracted {len(conversations)} valid conversations")

    # Prepare benchmark requests
    grouped_requests, num_conversations, total_requests = prepare_benchmark_requests(
        conversations=conversations,
        max_turns=args.max_turns,
        max_concurrent=args.max_concurrent,
        num_conversations=args.num_conversations,
        total_requests=args.total_requests,
        seed=args.seed,
    )

    print(f"\nBenchmark configuration:")
    print(f"  Conversations: {num_conversations}")
    print(f"  Turns per conversation: {args.max_turns}")
    print(f"  Total requests: {total_requests}")
    print(f"  Max concurrent: {args.max_concurrent}")

    # Create benchmark runner
    benchmark = LLMBenchmark(
        url=args.url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )

    # Warmup
    await benchmark.warmup(grouped_requests, args.num_warmups)

    # Run benchmark
    stats = await benchmark.run_benchmark(
        grouped_requests=grouped_requests,
        max_concurrent=args.max_concurrent,
        ramp_start=getattr(args, 'ramp_start', 0),
        ramp_step=getattr(args, 'ramp_step', 0),
    )

    # Output results
    print_stats(stats, args)
    export_results(stats, benchmark, args, num_conversations, total_requests)
