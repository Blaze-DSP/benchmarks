# Benchmarks

Real-time load testing suite for vLLM inference servers with support for LLM chat completions and audio STT endpoints.

## Features

- **Request-level concurrency control**: Maintains exact number of concurrent requests at all times
- **Dynamic scheduling**: New requests start immediately as previous ones complete
- **Gradual ramp-up**: Smoothly increase load from starting concurrency to maximum
- **Multi-turn conversation support**: LLM benchmark processes realistic multi-turn conversations
- **Streaming metrics**: TTFT (Time to First Token), TPOT (Time Per Output Token), E2E latency
- **Token throughput**: Input/output tokens per second tracking
- **vLLM optimized**: Designed to maximize prefix caching and KV cache efficiency

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### LLM Benchmark

Benchmarks `/v1/chat/completions` endpoint with multi-turn conversations.

```bash
# Basic usage
python -m benchmarks llm \
  --url https://llama.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 10 \
  --total-requests 100

# With ramp-up (start at 5, increase by 5 until reaching 20)
python -m benchmarks llm \
  --url https://llama.example.com/v1 \
  --model llama-3.1-8b \
  --dataset HuggingFaceH4/ultrachat_200k \
  --max-concurrent 20 \
  --ramp-start 5 \
  --ramp-step 5 \
  --total-requests 200
```

### STT Benchmark

Benchmarks audio transcription via `/v1/audio/transcriptions` or `/v1/chat/completions` endpoints.

```bash
# Transcriptions endpoint
python -m benchmarks stt \
  --url https://voxtral-mini.example.com/v1 \
  --model voxtral-mini \
  --endpoint transcriptions \
  --dataset your-audio-dataset \
  --max-concurrent 10 \
  --total-requests 100

# Chat endpoint with audio input
python -m benchmarks stt \
  --url https://qwen3-omni.example.com/v1 \
  --model qwen3-omni \
  --endpoint chat \
  --dataset your-audio-dataset \
  --max-concurrent 20 \
  --ramp-start 5 \
  --ramp-step 5 \
  --total-requests 100
```

## CLI Arguments

### Common Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--url` | str | required | Base URL for the API |
| `--model` | str | required | Model name to use |
| `--dataset` | str | required | HuggingFace dataset name |
| `--max-concurrent` | int | 10 | Maximum concurrent requests/conversations |
| `--total-requests` | int | 100 | Total requests (turns) to run |
| `--ramp-start` | int | 0 | Starting concurrency for ramp-up (0 = disabled) |
| `--ramp-step` | int | 0 | Step size for ramp-up (0 = disabled) |
| `--num-warmups` | int | 5 | Number of warmup requests |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--split` | str | train | Dataset split to use |
| `--api-key` | str | DUMMY | API key for authentication |

### LLM-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-tokens` | int | 256 | Max tokens per response |

### STT-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--endpoint` | str | required | Endpoint type: `transcriptions` or `chat` |
| `--prompt` | str | (default prompt) | Prompt for transcription |

## Concurrency Model

### LLM Benchmark
- **Concurrency = Number of active conversations** (simulating concurrent users)
- Each conversation processes turns sequentially
- New conversations start only when previous ones complete ALL turns
- Optimizes for vLLM prefix caching (same conversation stays "warm")

### STT Benchmark
- **Concurrency = Number of active requests**
- Independent requests with no state dependencies
- New requests start immediately when previous ones complete

### Ramp-up Behavior
- Starts with `ramp-start` concurrent requests/conversations
- After every `current_concurrency` requests complete, increases by `ramp-step`
- Continues until reaching `max-concurrent`

## Metrics Collected

### Latency Metrics
- **E2E Latency**: Total time from request start to completion
- **TTFT**: Time to First Token (streaming)
- **TPOT**: Time Per Output Token (generation speed)

### Throughput Metrics
- **Requests/sec**: Successful requests per second
- **Input tokens/sec**: Input token throughput
- **Output tokens/sec**: Output token throughput

### Percentile Statistics
All latency metrics include: avg, min, max, p50, p90, p95, p99

## Output

Results are exported to JSON files with full request details:
- `benchmark_llm_{model}_{timestamp}.json`
- `benchmark_stt_{model}_{endpoint}_{timestamp}.json`

---

## Future Improvements

### Load Testing Scenarios

- [ ] **Sustained load testing**: Run at fixed concurrency for a specified duration
- [ ] **Spike testing**: Sudden increase in load to test server resilience
- [ ] **Stress testing**: Gradually increase load until failure to find breaking point
- [ ] **Soak testing**: Extended duration tests to detect memory leaks
- [ ] **Variable load patterns**: Sinusoidal, step, or custom load curves

### Additional Benchmark Parameters

- [ ] **Request rate limiting**: Target specific requests/sec instead of max concurrency
- [ ] **Think time**: Configurable delay between requests (simulate real user behavior)
- [ ] **Request timeout**: Configurable timeout with proper error handling
- [ ] **Retry logic**: Configurable retry attempts for failed requests
- [ ] **Custom prompts**: Load prompts from file for reproducible testing

### Additional Metrics

- [ ] **Latency distribution histograms**: Visual latency distribution
- [ ] **Time-series metrics**: Track metrics over time during benchmark
- [ ] **Error categorization**: Detailed breakdown of error types
- [ ] **Queue depth tracking**: Monitor server-side queue buildup
- [ ] **GPU utilization**: Integrate with server metrics (if available)
- [ ] **Cache hit rates**: Track vLLM prefix cache effectiveness

### Output Enhancements

- [ ] **Real-time dashboard**: Live metrics visualization during benchmark
- [ ] **HTML reports**: Generate visual benchmark reports
- [ ] **Comparison mode**: Compare results between runs
- [ ] **Prometheus metrics**: Export metrics for monitoring systems
- [ ] **CSV export**: Additional export format for analysis

### Architecture Improvements

- [ ] **Distributed benchmarking**: Run from multiple clients simultaneously
- [ ] **Plugin system**: Extensible benchmark types
- [ ] **Configuration files**: YAML/JSON config instead of CLI args
- [ ] **Async dataset loading**: Stream large datasets without memory issues
