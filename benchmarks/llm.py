"""
LLM Chat Completions Benchmark Implementation.

Benchmarks /v1/chat/completions endpoint with multi-turn conversations.
Each concurrent request processes a full conversation from the dataset,
building up context turn by turn.
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


class ConversationState(BaseModel):
    """State tracker for an ongoing multi-turn conversation."""

    conversation_id: str
    messages: list[dict]  # Original conversation messages
    turns: list[dict] = Field(
        default_factory=list
    )  # Parsed turns: {"assistant": msg, "users": [msgs]}
    current_turn_index: int = 0
    system_msg: Optional[dict] = None
    context: list[dict] = Field(
        default_factory=list
    )  # Accumulated context for next turn
    results: list[TurnResult] = Field(default_factory=list)

    def has_more_turns(self) -> bool:
        """Check if conversation has remaining turns to process."""
        return self.current_turn_index < len(self.turns)

    def get_next_turn(self) -> Optional[dict]:
        """Get the next turn to process."""
        if self.has_more_turns():
            return self.turns[self.current_turn_index]
        return None

    def advance_turn(self, result: TurnResult) -> None:
        """Add result and advance to next turn."""
        self.results.append(result)

        # Add current turn to context
        current_turn = self.turns[self.current_turn_index]
        self.context.append(current_turn["assistant"])
        self.context.extend(current_turn["users"])

        # Add model's response to context
        if result.success and result.response:
            self.context.append({"role": "assistant", "content": result.response})

        self.current_turn_index += 1

    @classmethod
    def from_conversation(
        cls, conversation: list[dict], conversation_id: str
    ) -> "ConversationState":
        """Initialize state from a conversation."""
        system_msg = None
        turns = []
        current_turn = None

        # Extract system message and group into turns
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

        return cls(
            conversation_id=conversation_id,
            messages=conversation,
            turns=turns,
            system_msg=system_msg,
        )


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
            # If we had a complete turn before this assistant, count it
            if has_assistant and has_users:
                turn_count += 1
            has_assistant = True
            has_users = False
        elif role == "user" and has_assistant:
            has_users = True

    # Count the last turn if complete
    if has_assistant and has_users:
        turn_count += 1

    return turn_count


def truncate_conversation(conversation: list[dict], max_turns: int) -> list[dict]:
    """Truncate a conversation to a maximum number of turns.

    A turn is one assistant message followed by one or more user messages.
    Expected format: [system, assistant, user, user, ..., assistant, user, ...]
    """
    if max_turns <= 0:
        return []

    truncated = []
    turn_count = 0
    current_assistant = None
    current_users = []

    for msg in conversation:
        role = msg.get("role")

        if role == "system":
            truncated.append(msg)
        elif role == "assistant":
            # Save previous turn if complete
            if current_assistant and current_users:
                truncated.append(current_assistant)
                truncated.extend(current_users)
                turn_count += 1
                if turn_count >= max_turns:
                    break
            # Start new turn
            current_assistant = msg
            current_users = []
        elif role == "user" and current_assistant:
            current_users.append(msg)

    # Add last turn if complete and under limit
    if turn_count < max_turns and current_assistant and current_users:
        truncated.append(current_assistant)
        truncated.extend(current_users)

    return truncated


class LLMBenchmark:
    """Benchmark runner for LLM chat completions."""

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
        messages: list[dict],
        request_id: str,
        conversation_id: str,
        turn_index: int,
    ) -> TurnResult:
        """Make a streaming request to measure TTFT and collect tokens."""
        start_time = time.perf_counter()
        ttft = 0.0
        first_token_received = False
        collected_content = ""
        input_tokens = 0
        output_tokens = 0

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
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

            # TTFT remains 0 if no tokens received (no meaningful "first token" time)

            return TurnResult(
                request_id=request_id,
                conversation_id=conversation_id,
                turn_index=turn_index,
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
                conversation_id=conversation_id,
                turn_index=turn_index,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                ttft_ms=0,
                success=False,
                error=str(e),
            )

    async def _process_single_turn_no_semaphore(
        self, conv_state: ConversationState
    ) -> ConversationState:
        """Process a single turn from a conversation.

        No semaphore needed - concurrency is controlled by number of active conversations.
        """
        turn = conv_state.get_next_turn()
        if not turn:
            return conv_state

        # Build messages: system + context + current turn
        messages = []
        if conv_state.system_msg:
            messages.append(conv_state.system_msg)
        messages.extend(conv_state.context)
        messages.append(turn["assistant"])
        messages.extend(turn["users"])

        # Make request
        request_id = f"{conv_state.conversation_id}-t{conv_state.current_turn_index}"
        result = await self._stream_request(
            messages=messages,
            request_id=request_id,
            conversation_id=conv_state.conversation_id,
            turn_index=conv_state.current_turn_index,
        )

        # Update state
        conv_state.advance_turn(result)

        return conv_state

    async def run_conversation(
        self, conversation: list[dict], conversation_id: str
    ) -> list[TurnResult]:
        """Run a full multi-turn conversation.

        Expected format: [system, assistant, user, user, ..., assistant, user, ...]
        Each turn is: one assistant message followed by one or more user messages.
        Each request sends: system + context + current turn (assistant + users)
        """
        results = []
        context = []

        system_msg = None
        turns = []  # List of {"assistant": msg, "users": [msg, ...]}

        # Extract system message and group remaining into turns
        current_turn = None
        for msg in conversation:
            role = msg.get("role")
            if role == "system":
                system_msg = msg
            elif role == "assistant":
                # Save previous turn if complete
                if current_turn and current_turn["users"]:
                    turns.append(current_turn)
                # Start new turn
                current_turn = {"assistant": msg, "users": []}
            elif role == "user" and current_turn:
                current_turn["users"].append(msg)

        # Don't forget the last turn
        if current_turn and current_turn["users"]:
            turns.append(current_turn)

        # Process each turn
        for turn_index, turn in enumerate(turns):
            # Build messages: system + context + assistant + all users
            messages = []
            if system_msg:
                messages.append(system_msg)
            messages.extend(context)
            messages.append(turn["assistant"])
            messages.extend(turn["users"])

            request_id = f"{conversation_id}-t{turn_index}"
            result = await self._stream_request(
                messages=messages,
                request_id=request_id,
                conversation_id=conversation_id,
                turn_index=turn_index,
            )
            results.append(result)

            # Add current turn to context for next turn
            context.append(turn["assistant"])
            context.extend(turn["users"])

            # Add model's response to context
            if result.success and result.response:
                context.append({"role": "assistant", "content": result.response})

        return results

    async def warmup(
        self, conversations: list[list[dict]], num_warmups: int = 3
    ) -> None:
        """Run warmup requests (not counted in metrics)."""
        if num_warmups <= 0:
            return

        print(f"\nRunning {num_warmups} warmup requests...")

        for i in range(num_warmups):
            conv_idx = random.randint(0, len(conversations) - 1)
            conv = conversations[conv_idx]

            # Build messages: system + first turn (assistant + users)
            messages = []
            first_assistant = None
            first_users = []

            for msg in conv:
                role = msg.get("role")
                if role == "system":
                    messages.append(msg)
                elif role == "assistant":
                    if first_assistant is None:
                        first_assistant = msg
                    else:
                        break  # Stop at second assistant
                elif role == "user" and first_assistant:
                    first_users.append(msg)

            if first_assistant:
                messages.append(first_assistant)
            messages.extend(first_users)

            if len(messages) < 2:
                continue

            result = await self._stream_request(
                messages=messages,
                request_id=f"warmup-{i}",
                conversation_id=f"warmup-{i}",
                turn_index=0,
            )
            status = "✓" if result.success else "✗"
            print(
                f"  Warmup {i + 1}/{num_warmups}: {status} "
                f"(latency: {result.latency_ms:.0f}ms, ttft: {result.ttft_ms:.0f}ms)"
            )

    async def run_benchmark(
        self,
        conversations: list[list[dict]],
        max_concurrent: int,
        ramp_start: int = 0,
        ramp_step: int = 0,
    ) -> LLMBenchmarkStats:
        """Run the benchmark with request-level concurrency control.

        Concurrency level = number of active conversations (simulating concurrent users).
        Each conversation processes turns sequentially (realistic user behavior).
        New conversations start only when previous ones complete all turns.
        """
        self.results = []
        benchmark_start = time.perf_counter()

        total_convs = len(conversations)
        total_turns = sum(count_turns(conv) for conv in conversations)

        use_ramp_up = ramp_start > 0 and ramp_step > 0 and ramp_start < max_concurrent

        print(f"\nRunning benchmark:")
        print(f"  Conversations: {total_convs}")
        print(f"  Total turns: {total_turns}")
        if use_ramp_up:
            print(
                f"  Ramp-up: {ramp_start} -> {max_concurrent} conversations (step: {ramp_step})"
            )
        else:
            print(f"  Max concurrent conversations: {max_concurrent}")

        # Initialize conversation states
        pending_convs = [
            ConversationState.from_conversation(conv, str(uuid.uuid4())[:8])
            for conv in conversations
        ]

        # Concurrency tracking
        current_concurrency = ramp_start if use_ramp_up else max_concurrent
        completed_requests = 0
        completed_convs = 0

        # Start initial batch of conversations
        initial_count = min(current_concurrency, len(pending_convs))
        active_conv_tasks = {}  # task -> ConversationState mapping

        for _ in range(initial_count):
            conv_state = pending_convs.pop(0)
            task = asyncio.create_task(
                self._process_single_turn_no_semaphore(conv_state)
            )
            active_conv_tasks[task] = conv_state

        # Track when to ramp up (after batch of requests complete)
        requests_at_last_ramp = 0

        # Store all completed conversation states to collect results later
        all_completed_convs = []

        # Process conversations as requests complete
        while active_conv_tasks:
            done, pending = await asyncio.wait(
                active_conv_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                conv_state = await task
                completed_requests += 1

                # Progress reporting
                if completed_requests % 10 == 0 or completed_requests == total_turns:
                    elapsed = time.perf_counter() - benchmark_start
                    print(
                        f"  Progress: {completed_requests}/{total_turns} requests, "
                        f"{completed_convs}/{total_convs} conversations, "
                        f"{len(active_conv_tasks)}/{current_concurrency} active "
                        f"({elapsed:.1f}s)"
                    )

                # Remove completed task
                del active_conv_tasks[task]

                # Check if conversation has more turns
                if conv_state.has_more_turns():
                    # Schedule next turn from SAME conversation (same user continues)
                    new_task = asyncio.create_task(
                        self._process_single_turn_no_semaphore(conv_state)
                    )
                    active_conv_tasks[new_task] = conv_state
                else:
                    # Conversation fully complete
                    completed_convs += 1
                    all_completed_convs.append(conv_state)

                    # Replace with a new conversation if available
                    if pending_convs:
                        new_conv = pending_convs.pop(0)
                        new_task = asyncio.create_task(
                            self._process_single_turn_no_semaphore(new_conv)
                        )
                        active_conv_tasks[new_task] = new_conv

                # Ramp-up logic: increase active conversations after batch completes
                if use_ramp_up and current_concurrency < max_concurrent:
                    # Ramp up after processing a batch of requests (one full round)
                    requests_since_ramp = completed_requests - requests_at_last_ramp

                    # Trigger ramp-up after current_concurrency requests complete
                    # (meaning each active conversation completed one request)
                    if requests_since_ramp >= current_concurrency:
                        old_concurrency = current_concurrency
                        current_concurrency = min(
                            current_concurrency + ramp_step, max_concurrent
                        )

                        conversations_to_add = current_concurrency - old_concurrency
                        requests_at_last_ramp = completed_requests

                        print(
                            f"  Ramping up: {old_concurrency} -> {current_concurrency} conversations"
                        )

                        # Start additional conversations
                        for _ in range(min(conversations_to_add, len(pending_convs))):
                            new_conv = pending_convs.pop(0)
                            new_task = asyncio.create_task(
                                self._process_single_turn_no_semaphore(new_conv)
                            )
                            active_conv_tasks[new_task] = new_conv

        # Collect all results from all conversation states
        for conv_state in all_completed_convs:
            self.results.extend(conv_state.results)

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


def print_stats(stats: LLMBenchmarkStats, args) -> None:
    """Pretty print benchmark statistics."""
    print_header("BENCHMARK RESULTS")

    print("\nConfiguration:")
    print(f"  URL:              {args.url}")
    print(f"  Model:            {args.model}")
    print(f"  Conversations:    {stats.total_conversations}")
    print(f"  Total Turns:      {stats.total_requests} (target: {args.total_requests})")
    print(f"  Max Concurrent:   {args.max_concurrent}")
    if args.ramp_start > 0 and args.ramp_step > 0:
        print(
            f"  Ramp-up:          {args.ramp_start} -> {args.max_concurrent} "
            f"(step: {args.ramp_step})"
        )

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
    total_requests_target: int
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


def export_results(stats: LLMBenchmarkStats, benchmark: LLMBenchmark, args) -> str:
    """Export benchmark results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"benchmark_llm_{args.model}_{timestamp}.json"

    output = LLMBenchmarkOutput(
        timestamp=datetime.now().isoformat(),
        config=LLMBenchmarkConfig(
            url=args.url,
            model=args.model,
            max_concurrent=args.max_concurrent,
            total_requests_target=args.total_requests,
            ramp_start=args.ramp_start,
            ramp_step=args.ramp_step,
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


async def run_llm_benchmark(args) -> None:
    """Main entry point for LLM benchmark."""
    random.seed(args.seed)

    df = load_and_prepare_dataset(args.dataset, args.split)
    print(f"Loaded {len(df)} conversations from dataset")

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

    # Sample conversations to reach target total_requests (hard limit)
    # Each request = one turn (assistant + users)
    conv_turns = [(i, count_turns(conv)) for i, conv in enumerate(conversations)]

    random.shuffle(conv_turns)
    selected_convs = []
    current_turns = 0
    target_turns = args.total_requests

    for idx, turns in conv_turns:
        if current_turns >= target_turns:
            break

        conv = conversations[idx]
        turns_needed = target_turns - current_turns

        if turns <= turns_needed:
            selected_convs.append(conv)
            current_turns += turns
        else:
            truncated = truncate_conversation(conv, turns_needed)
            if truncated:
                selected_convs.append(truncated)
                current_turns += turns_needed

    conversations = selected_convs

    total_turns = sum(count_turns(conv) for conv in conversations)
    print(
        f"Selected {len(conversations)} conversations with {total_turns} total turns "
        f"(target: {target_turns})"
    )

    benchmark = LLMBenchmark(
        url=args.url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )

    await benchmark.warmup(conversations, args.num_warmups)

    stats = await benchmark.run_benchmark(
        conversations=conversations,
        max_concurrent=args.max_concurrent,
        ramp_start=args.ramp_start,
        ramp_step=args.ramp_step,
    )

    print_stats(stats, args)
    export_results(stats, benchmark, args)
