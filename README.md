# Benchmarks

Load testing suite for vLLM inference servers with support for LLM chat completions and audio STT endpoints.

## Features

- **Guaranteed concurrency**: Semaphore-based control ensures exact concurrent request count
- **Multi-turn conversations**: LLM benchmark processes realistic multi-turn conversations
- **Sequential turns, parallel conversations**: Turns within a conversation run sequentially; multiple conversations run in parallel
- **Streaming metrics**: TTFT (Time to First Token), TPOT (Time Per Output Token), E2E latency
- **Token throughput**: Input/output tokens per second tracking
- **Pre-built requests**: All requests pre-built with deterministic context for reproducibility
- **Reproducible benchmarks**: Control shuffling with `--shuffle/--no-shuffle` and set `--seed` for deterministic runs

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### LLM Benchmark - Test 20 Concurrent Requests

```bash
python -m benchmarks llm \
  --url https://your-model.example.com/v1 \
  --model your-model-name \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --num-conversations 40 \
  --max-turns 5
```

This runs:
- 40 conversations × 5 turns = 200 total requests
- Exactly 20 requests running concurrently at all times
- Each conversation's turns execute sequentially

## Concurrency Model

### How It Works

```
Conversation 1:  [Turn 0] → [Turn 1] → [Turn 2] → ...
Conversation 2:  [Turn 0] → [Turn 1] → [Turn 2] → ...
Conversation 3:  [Turn 0] → [Turn 1] → [Turn 2] → ...
...
                    ↓         ↓         ↓
              Semaphore limits to N concurrent requests
```

1. **All conversation chains start simultaneously**
2. **Semaphore gates execution**: Only `max_concurrent` requests run at once
3. **Sequential within chains**: A conversation's Turn 1 waits for Turn 0 to complete
4. **Parallel across chains**: Different conversations execute turns in parallel
5. **FIFO scheduling**: When a request completes, the next waiting request starts

### Why This Works

- With 40 conversations and `--max-concurrent 20`:
  - 20 chains acquire semaphore, start Turn 0
  - 20 chains wait in queue
  - As Turn 0 completes, that chain releases semaphore and queues for Turn 1
  - A waiting chain acquires the slot

- **Result**: Exactly 20 requests running at all times until the end tail

### Tail Effect

Concurrency drops only when `remaining_chains < max_concurrent`:

| Conversations | Max Concurrent | Full Concurrency Until |
|---------------|----------------|------------------------|
| 40            | 20             | ~80% of requests       |
| 50            | 20             | ~90% of requests       |
| 60            | 20             | ~93% of requests       |

**Rule of thumb**: Use `num_conversations >= 1.5 × max_concurrent` for sustained concurrency.

## Usage

### LLM Benchmark

```bash
# Specify conversations and turns explicitly
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --num-conversations 50 \
  --max-turns 4

# Auto-calculate turns from total requests
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --total-requests 200 \
  --num-conversations 40
# → Calculates: 200 / 40 = 5 turns per conversation

# Just specify max turns (uses all valid conversations)
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --max-turns 5
```

### STT Benchmark

```bash
# Transcriptions endpoint
python -m benchmarks stt \
  --url https://voxtral.example.com/v1 \
  --model voxtral-mini \
  --endpoint transcriptions \
  --dataset your-audio-dataset \
  --max-concurrent 10 \
  --total-requests 100

# Chat endpoint with audio
python -m benchmarks stt \
  --url https://qwen3-omni.example.com/v1 \
  --model qwen3-omni \
  --endpoint chat \
  --dataset your-audio-dataset \
  --max-concurrent 20 \
  --total-requests 100
```

## CLI Arguments

### Common Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--url` | str | required | Base URL for the API (e.g., `https://api.example.com/v1`) |
| `--model` | str | required | Model name to use |
| `--dataset` | str | required | HuggingFace dataset name |
| `--max-concurrent` | int | 10 | Maximum concurrent requests |
| `--total-requests` | int | auto | Total requests to run (auto-calculated if not specified) |
| `--num-warmups` | int | 5 | Number of warmup requests |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--shuffle` / `--no-shuffle` | bool | True | Shuffle dataset before sampling (use `--no-shuffle` for deterministic order) |
| `--split` | str | train | Dataset split to use |
| `--api-key` | str | DUMMY | API key for authentication |

### LLM-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-tokens` | int | 256 | Max tokens per response |
| `--max-turns` | int | required* | Turns per conversation (all truncated to this) |
| `--num-conversations` | int | auto | Number of conversations to use |
| `--ramp-start` | int | 0 | Starting concurrency for ramp-up (0 = disabled) |
| `--ramp-step` | int | 0 | Step size for ramp-up (0 = disabled) |

*Either `--max-turns` or `--num-conversations` with `--total-requests` must be specified.

### STT-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--endpoint` | str | required | Endpoint type: `transcriptions` or `chat` |
| `--prompt` | str | (default) | Prompt for transcription |
| `--ramp-start` | int | 0 | Starting concurrency for ramp-up |
| `--ramp-step` | int | 0 | Step size for ramp-up |

## Examples

### Load Test: Find Maximum Sustainable Concurrency

Test increasing concurrency levels to find the server's limit:

```bash
# Test at 10 concurrent
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 10 \
  --num-conversations 20 \
  --max-turns 10

# Test at 20 concurrent
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --num-conversations 40 \
  --max-turns 10

# Test at 50 concurrent
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 50 \
  --num-conversations 100 \
  --max-turns 10
```

### Throughput Test: Maximum Requests per Second

High concurrency with short conversations:

```bash
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 100 \
  --num-conversations 200 \
  --max-turns 2 \
  --max-tokens 50
```

### Latency Test: Measure Response Times Under Load

Moderate concurrency, many requests for statistical significance:

```bash
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 10 \
  --num-conversations 50 \
  --max-turns 10
```

### Context Length Test: Long Conversations

Test performance with growing context:

```bash
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 10 \
  --num-conversations 20 \
  --max-turns 20 \
  --max-tokens 512
```

### Ramp-Up Test: Gradual Load Increase

Start with low concurrency and gradually increase:

```bash
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --num-conversations 40 \
  --max-turns 5 \
  --ramp-start 2 \
  --ramp-step 2
```

This starts with 2 concurrent requests and increases by 2 after each batch completes (2 → 4 → 6 → ... → 20).

### Reproducible Test: Deterministic Order

Run benchmarks with deterministic data ordering for reproducible results:

```bash
python -m benchmarks llm \
  --url https://api.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --num-conversations 40 \
  --max-turns 5 \
  --no-shuffle \
  --seed 42
```

With `--no-shuffle`, conversations are selected in dataset order. Combined with `--seed`, this ensures the same requests are used across runs.

## Metrics Collected

### Latency Metrics

| Metric | Description |
|--------|-------------|
| **E2E Latency** | Total time from request start to completion |
| **TTFT** | Time to First Token (streaming) |
| **TPOT** | Time Per Output Token (excluding first token) |

### Throughput Metrics

| Metric | Description |
|--------|-------------|
| **Requests/sec** | Successful requests per second |
| **Input tok/sec** | Input token throughput |
| **Output tok/sec** | Output token throughput |

### Percentile Statistics

All latency metrics include: avg, min, max, p50, p90, p95, p99

## Output

Results are exported to JSON files:

```
benchmark_llm_{model}_{timestamp}.json
benchmark_stt_{model}_{endpoint}_{timestamp}.json
```

### JSON Structure

```json
{
  "benchmark_type": "llm",
  "timestamp": "2024-01-15T10:30:00",
  "config": {
    "url": "https://api.example.com/v1",
    "model": "llama-3.1-8b",
    "max_concurrent": 20,
    "total_requests": 200,
    "num_conversations": 40,
    "max_turns": 5,
    "max_tokens": 256,
    "seed": 42,
    "shuffle": true
  },
  "stats": {
    "summary": {
      "total_requests": 200,
      "successful_requests": 200,
      "failed_requests": 0,
      "total_conversations": 40,
      "total_time_sec": 45.2
    },
    "throughput": {
      "requests_per_sec": 4.42,
      "output_tokens_per_sec": 1130.5
    },
    "latency_ms": {"avg": 450, "p50": 420, "p99": 890},
    "ttft_ms": {"avg": 85, "p50": 78, "p99": 195},
    "tpot_ms": {"avg": 12.5, "p50": 11.8, "p99": 25.3}
  },
  "requests": [...]
}
```

## Troubleshooting

### "No conversations found with >= N turns"

The dataset doesn't have enough multi-turn conversations. Solutions:
- Lower `--max-turns`
- Use a different dataset with longer conversations

### "Need at least N conversations to maintain concurrency"

Not enough conversations after filtering. Solutions:
- Lower `--max-concurrent`
- Lower `--max-turns` (more conversations will qualify)
- Use a larger dataset

### Warning about sustained concurrency

The benchmark warns if concurrency may drop before completion:

```
⚠️  Warning: 25 conversations may not sustain 20 concurrency until end.
   Full concurrency sustained for ~25% of requests.
   Recommendation: Use >= 30 conversations
```

Increase `--num-conversations` or decrease `--max-turns` to fix.

### High failure rate

Check:
- Server is running and accessible
- Model name is correct
- API key is valid (if required)
- Server can handle the concurrency level
