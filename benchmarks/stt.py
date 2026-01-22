"""
Audio STT Benchmark Implementation.

Benchmarks /v1/audio/transcriptions and /v1/chat/completions endpoints
for audio transcription models like Voxtral-Mini and Qwen3-Omni.
"""

import asyncio
import base64
import random
import time
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from datasets import Audio, load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel

from ..common import BenchmarkStats, print_footer, print_header


class STTRequestResult(BaseModel):
    """Result of a single STT benchmark request."""

    request_id: str
    start_time: float
    end_time: float
    latency_ms: float
    ttft_ms: float
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    transcript: Optional[str] = None
    audio_size_bytes: int = 0
    prompt: Optional[str] = None


def load_and_prepare_dataset(dataset: str, split: str = "validation") -> pd.DataFrame:
    """Load and prepare the audio dataset."""
    print(f"Loading dataset: {dataset} (split: {split})")
    ds = load_dataset(dataset, split=split)
    ds = ds.cast_column("audio_bytes", Audio(sampling_rate=16000))
    return ds.to_pandas()


class AudioSTTBenchmark:
    """Benchmark runner for Audio STT endpoints."""

    def __init__(
        self,
        url: str,
        model: str,
        endpoint: str,
        api_key: str = "DUMMY",
        prompt: Optional[str] = None,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.prompt = prompt or ""

        self.client = AsyncOpenAI(base_url=self.url, api_key=self.api_key)
        self.results: list[STTRequestResult] = []

    async def _request_transcriptions(
        self, audio_bytes: bytes, request_id: str, save_inputs: bool = False
    ) -> STTRequestResult:
        """Make a streaming request to /v1/audio/transcriptions endpoint."""
        start_time = time.perf_counter()
        ttft = 0.0
        first_token_received = False
        collected_text = ""

        try:
            file = (f"request-{request_id}.wav", audio_bytes, "audio/wav")
            kwargs = {
                "file": file,
                "model": self.model,
                "stream": True,
            }

            if self.prompt:
                kwargs["prompt"] = self.prompt

            stream = await self.client.audio.transcriptions.create(**kwargs)

            async for chunk in stream:
                # Transcription stream uses choices[].delta.content format (like chat)
                chunk_text = None

                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    # choices can be dict or object
                    if isinstance(choice, dict):
                        delta = choice.get("delta", {})
                        chunk_text = (
                            delta.get("content") if isinstance(delta, dict) else None
                        )
                    elif hasattr(choice, "delta") and choice.delta:
                        chunk_text = getattr(choice.delta, "content", None)

                # Fallback for other formats
                if not chunk_text:
                    if hasattr(chunk, "text") and chunk.text:
                        chunk_text = chunk.text
                    elif hasattr(chunk, "delta") and chunk.delta:
                        chunk_text = chunk.delta
                    elif isinstance(chunk, str):
                        chunk_text = chunk

                if chunk_text:
                    if not first_token_received:
                        ttft = (time.perf_counter() - start_time) * 1000
                        first_token_received = True
                    collected_text += chunk_text

            end_time = time.perf_counter()

            # If streaming returned no text, the stream object might have final text
            if not collected_text and hasattr(stream, "text"):
                collected_text = stream.text

            # TTFT remains 0 if no tokens received (no meaningful "first token" time)

            return STTRequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                ttft_ms=ttft,
                success=True,
                transcript=collected_text,
                audio_size_bytes=len(audio_bytes),
                prompt=self.prompt if save_inputs else None,
            )

        except Exception as e:
            end_time = time.perf_counter()
            return STTRequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                ttft_ms=0,
                success=False,
                error=str(e),
                audio_size_bytes=len(audio_bytes),
                prompt=self.prompt if save_inputs else None,
            )

    async def _request_chat(
        self, audio_bytes: bytes, request_id: str, save_inputs: bool = False
    ) -> STTRequestResult:
        """Make a streaming request to /v1/chat/completions with audio."""
        start_time = time.perf_counter()
        ttft = 0.0
        first_token_received = False
        collected_content = ""
        input_tokens = 0
        output_tokens = 0

        try:
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            content = []
            if self.prompt:
                content.append({"type": "text", "text": self.prompt})

            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_base64, "format": "wav"},
                }
            )

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
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

            return STTRequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                ttft_ms=ttft,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                transcript=collected_content,
                audio_size_bytes=len(audio_bytes),
                prompt=self.prompt if save_inputs else None,
            )

        except Exception as e:
            end_time = time.perf_counter()
            return STTRequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                latency_ms=(end_time - start_time) * 1000,
                ttft_ms=0,
                success=False,
                error=str(e),
                audio_size_bytes=len(audio_bytes),
                prompt=self.prompt if save_inputs else None,
            )

    async def _make_request(
        self, audio_bytes: bytes, request_id: str, save_inputs: bool = False
    ) -> STTRequestResult:
        """Route request to appropriate endpoint handler."""
        if self.endpoint == "transcriptions":
            return await self._request_transcriptions(
                audio_bytes, request_id, save_inputs
            )
        elif self.endpoint == "chat":
            return await self._request_chat(audio_bytes, request_id, save_inputs)
        else:
            raise ValueError(f"Unknown endpoint: {self.endpoint}")

    async def warmup(
        self, audio_samples: list[bytes], num_warmups: int = 5, shuffle: bool = True
    ) -> None:
        """Run warmup requests (not counted in metrics)."""
        if num_warmups <= 0:
            return

        print(f"\nRunning {num_warmups} warmup requests...")
        for i in range(num_warmups):
            if shuffle:
                idx = random.randint(0, len(audio_samples) - 1)
            else:
                idx = i % len(audio_samples)
            result = await self._make_request(audio_samples[idx], f"warmup-{i}")
            status = "✓" if result.success else "✗"
            ttft_str = f", ttft: {result.ttft_ms:.0f}ms" if result.ttft_ms > 0 else ""
            print(
                f"  Warmup {i + 1}/{num_warmups}: {status} "
                f"(latency: {result.latency_ms:.0f}ms{ttft_str})"
            )

    async def run_benchmark(
        self,
        audio_samples: list[bytes],
        total_requests: int,
        max_concurrent: int,
        ramp_start: int = 0,
        ramp_step: int = 0,
        shuffle: bool = True,
        save_inputs: bool = False,
    ) -> BenchmarkStats:
        """Run the benchmark with request-level concurrency control.

        Maintains exact concurrency at the request level. New requests start
        immediately as previous ones complete. Ramp-up increases concurrency
        after batches of requests complete.
        """
        self.results = []
        benchmark_start = time.perf_counter()

        # Pre-generate sample indices for reproducibility
        if shuffle:
            sample_indices = [
                random.randint(0, len(audio_samples) - 1) for _ in range(total_requests)
            ]
        else:
            # Sequential indices (cycling through dataset)
            sample_indices = [i % len(audio_samples) for i in range(total_requests)]

        use_ramp_up = ramp_start > 0 and ramp_step > 0 and ramp_start < max_concurrent

        if use_ramp_up:
            print(
                f"\nRunning benchmark: {total_requests} requests"
                f"\n  Ramp-up: {ramp_start} -> {max_concurrent} concurrent requests (step: {ramp_step})"
            )
        else:
            print(
                f"\nRunning benchmark: {total_requests} requests, "
                f"max {max_concurrent} concurrent requests"
            )

        # Concurrency tracking
        current_concurrency = ramp_start if use_ramp_up else max_concurrent
        completed_requests = 0
        requests_at_last_ramp = 0

        # Queue of pending requests (index, audio_bytes, request_id)
        pending_requests = [
            (i, audio_samples[sample_indices[i]], str(uuid.uuid4())[:8])
            for i in range(total_requests)
        ]
        active_tasks = {}  # task -> request_index mapping

        # Start initial batch of requests
        for _ in range(min(current_concurrency, len(pending_requests))):
            idx, audio, req_id = pending_requests.pop(0)
            task = asyncio.create_task(
                self._make_request(audio, req_id, save_inputs=save_inputs)
            )
            active_tasks[task] = idx

        # Process requests as they complete
        while active_tasks:
            done, pending = await asyncio.wait(
                active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                result = await task
                self.results.append(result)
                completed_requests += 1

                # Progress reporting
                if completed_requests % 10 == 0 or completed_requests == total_requests:
                    elapsed = time.perf_counter() - benchmark_start
                    rps = completed_requests / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress: {completed_requests}/{total_requests} "
                        f"({rps:.1f} req/s, concurrency: {current_concurrency})"
                    )

                # Remove completed task
                del active_tasks[task]

                # Start new request if available
                if pending_requests:
                    idx, audio, req_id = pending_requests.pop(0)
                    new_task = asyncio.create_task(
                        self._make_request(audio, req_id, save_inputs=save_inputs)
                    )
                    active_tasks[new_task] = idx

                # Ramp-up logic: increase concurrency after batch completes
                if use_ramp_up and current_concurrency < max_concurrent:
                    requests_since_ramp = completed_requests - requests_at_last_ramp

                    # Trigger ramp-up after current_concurrency requests complete
                    if requests_since_ramp >= current_concurrency:
                        old_concurrency = current_concurrency
                        current_concurrency = min(
                            current_concurrency + ramp_step, max_concurrent
                        )
                        requests_at_last_ramp = completed_requests

                        print(
                            f"  Ramping up: {old_concurrency} -> {current_concurrency} concurrent requests"
                        )

                        # Start additional requests
                        requests_to_add = current_concurrency - old_concurrency
                        for _ in range(min(requests_to_add, len(pending_requests))):
                            idx, audio, req_id = pending_requests.pop(0)
                            new_task = asyncio.create_task(
                                self._make_request(
                                    audio, req_id, save_inputs=save_inputs
                                )
                            )
                            active_tasks[new_task] = idx

        total_time = time.perf_counter() - benchmark_start
        return BenchmarkStats.compute(self.results, total_time)


def print_stats(stats: BenchmarkStats, args) -> None:
    """Pretty print benchmark statistics."""
    print_header("BENCHMARK RESULTS")

    print("\nConfiguration:")
    print(f"  URL:              {args.url}")
    print(f"  Model:            {args.model}")
    endpoint_str = (
        f"/v1/audio/{args.endpoint}"
        if args.endpoint == "transcriptions"
        else "/v1/chat/completions"
    )
    print(f"  Endpoint:         {endpoint_str}")
    print(f"  Concurrency:      {args.max_concurrent}")
    print(f"  Total Requests:   {args.total_requests}")
    if args.ramp_start > 0 and args.ramp_step > 0:
        print(
            f"  Ramp-up:          {args.ramp_start} -> {args.max_concurrent} "
            f"(step: {args.ramp_step})"
        )

    show_tokens = args.endpoint == "chat"
    stats.print_summary(
        show_tokens=show_tokens,
        show_ttft=True,
        show_tpot=show_tokens,
    )

    print_footer()


class STTBenchmarkConfig(BaseModel):
    """STT benchmark configuration."""

    url: str
    model: str
    endpoint: str
    max_concurrent: int
    total_requests: int
    ramp_start: int
    ramp_step: int
    num_warmups: int
    prompt: Optional[str]
    dataset: str
    split: str
    seed: int
    shuffle: bool
    save_inputs: bool
    api_key: str


class STTBenchmarkOutput(BaseModel):
    """Complete STT benchmark output."""

    benchmark_type: str = "stt"
    timestamp: str
    config: STTBenchmarkConfig
    stats: "BenchmarkStatsOutput"
    requests: list[STTRequestResult]


# Import here to avoid circular import
from ..common.stats import BenchmarkStatsOutput


def export_results(stats: BenchmarkStats, benchmark: AudioSTTBenchmark, args) -> str:
    """Export benchmark results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"benchmark_stt_{args.model}_{args.endpoint}_{timestamp}.json"

    output = STTBenchmarkOutput(
        timestamp=datetime.now().isoformat(),
        config=STTBenchmarkConfig(
            url=args.url,
            model=args.model,
            endpoint=args.endpoint,
            max_concurrent=args.max_concurrent,
            total_requests=args.total_requests,
            ramp_start=args.ramp_start,
            ramp_step=args.ramp_step,
            num_warmups=args.num_warmups,
            prompt=args.prompt,
            dataset=args.dataset,
            split=args.split,
            seed=args.seed,
            shuffle=args.shuffle,
            save_inputs=args.save_inputs,
            api_key="***" if args.api_key != "DUMMY" else "DUMMY",
        ),
        stats=stats.to_output(),
        requests=benchmark.results,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    print(f"\nResults exported to: {output_path}")
    return output_path


async def run_stt_benchmark(args) -> None:
    """Main entry point for STT benchmark."""
    random.seed(args.seed)

    df = load_and_prepare_dataset(args.dataset, args.split)
    print(f"Loaded {len(df)} samples from dataset")

    audio_samples = []
    for _, row in df.iterrows():
        audio_data = row["audio_bytes"]
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_samples.append(audio_data["bytes"])
        elif isinstance(audio_data, bytes):
            audio_samples.append(audio_data)

    if not audio_samples:
        print("Error: No valid audio samples found in dataset")
        return

    print(f"Extracted {len(audio_samples)} audio samples")
    print(f"Will randomly sample {args.total_requests} requests from dataset")

    benchmark = AudioSTTBenchmark(
        url=args.url,
        model=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        prompt=args.prompt,
    )

    await benchmark.warmup(audio_samples, args.num_warmups, shuffle=args.shuffle)

    stats = await benchmark.run_benchmark(
        audio_samples=audio_samples,
        total_requests=args.total_requests,
        max_concurrent=args.max_concurrent,
        ramp_start=args.ramp_start,
        ramp_step=args.ramp_step,
        shuffle=args.shuffle,
        save_inputs=args.save_inputs,
    )

    print_stats(stats, args)
    export_results(stats, benchmark, args)
