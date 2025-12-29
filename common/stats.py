"""Benchmark statistics models and computation utilities."""

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class PercentileStats(BaseModel):
    """Statistics with percentile breakdowns."""

    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @classmethod
    def from_array(cls, values: np.ndarray) -> "PercentileStats":
        """Compute percentile stats from a numpy array."""
        if len(values) == 0:
            return cls()

        return cls(
            avg=float(np.mean(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            p50=float(np.percentile(values, 50)),
            p90=float(np.percentile(values, 90)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
        )

    def print_stats(self, indent: str = "  ") -> None:
        """Print formatted statistics."""
        print(f"{indent}Avg:              {self.avg:.2f}")
        print(f"{indent}Min:              {self.min:.2f}")
        print(f"{indent}Max:              {self.max:.2f}")
        print(f"{indent}P50:              {self.p50:.2f}")
        print(f"{indent}P90:              {self.p90:.2f}")
        print(f"{indent}P95:              {self.p95:.2f}")
        print(f"{indent}P99:              {self.p99:.2f}")


class TokenStats(BaseModel):
    """Token throughput statistics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    input_tokens_per_sec: float = 0.0
    output_tokens_per_sec: float = 0.0

    @classmethod
    def compute(
        cls,
        input_tokens: int,
        output_tokens: int,
        total_time: float,
    ) -> "TokenStats":
        """Compute token statistics."""
        return cls(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            input_tokens_per_sec=input_tokens / total_time if total_time > 0 else 0,
            output_tokens_per_sec=output_tokens / total_time if total_time > 0 else 0,
        )

    def print_stats(self, indent: str = "  ") -> None:
        """Print formatted statistics."""
        print(f"{indent}Input tok/sec:    {self.input_tokens_per_sec:.2f}")
        print(f"{indent}Output tok/sec:   {self.output_tokens_per_sec:.2f}")
        print(f"{indent}Total Input Tok:  {self.total_input_tokens}")
        print(f"{indent}Total Output Tok: {self.total_output_tokens}")


class BaseRequestResult(BaseModel):
    """Base result for a single benchmark request."""

    request_id: str
    start_time: float
    end_time: float
    latency_ms: float
    ttft_ms: float
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None


class SummaryStats(BaseModel):
    """Summary statistics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_sec: float


class ThroughputStats(BaseModel):
    """Throughput statistics."""

    requests_per_sec: float
    total_input_tokens: int
    total_output_tokens: int
    input_tokens_per_sec: float
    output_tokens_per_sec: float


class BenchmarkStatsOutput(BaseModel):
    """Structured output for benchmark stats."""

    summary: SummaryStats
    throughput: ThroughputStats
    latency_ms: PercentileStats
    ttft_ms: PercentileStats
    tpot_ms: PercentileStats
    errors: dict = Field(default_factory=dict)


class BenchmarkStats:
    """Aggregated benchmark statistics."""

    def __init__(
        self,
        total_requests: int,
        successful_requests: int,
        failed_requests: int,
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
        self.total_time_sec = total_time_sec
        self.throughput_rps = throughput_rps
        self.tokens = tokens
        self.latency = latency
        self.ttft = ttft
        self.tpot = tpot
        self.errors = errors or {}

    @classmethod
    def compute(
        cls,
        results: list,
        total_time: float,
        get_latency=lambda r: r.latency_ms,
        get_ttft=lambda r: r.ttft_ms,
        get_input_tokens=lambda r: r.input_tokens,
        get_output_tokens=lambda r: r.output_tokens,
    ) -> "BenchmarkStats":
        """Compute statistics from a list of results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        errors = {}
        for r in failed:
            error_key = r.error[:50] if r.error else "Unknown"
            errors[error_key] = errors.get(error_key, 0) + 1

        if not successful:
            return cls(
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(failed),
                total_time_sec=total_time,
                throughput_rps=0,
                tokens=TokenStats(),
                latency=PercentileStats(),
                ttft=PercentileStats(),
                tpot=PercentileStats(),
                errors=errors,
            )

        total_input = sum(get_input_tokens(r) for r in successful)
        total_output = sum(get_output_tokens(r) for r in successful)

        latencies = np.array([get_latency(r) for r in successful])
        ttfts = np.array([get_ttft(r) for r in successful if get_ttft(r) > 0])

        tpots = []
        for r in successful:
            output_tok = get_output_tokens(r)
            ttft = get_ttft(r)
            latency = get_latency(r)
            if output_tok > 0 and ttft > 0:
                generation_time = latency - ttft
                tpots.append(generation_time / output_tok)
        tpots_arr = np.array(tpots) if tpots else np.array([])

        return cls(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_time_sec=total_time,
            throughput_rps=len(successful) / total_time if total_time > 0 else 0,
            tokens=TokenStats.compute(total_input, total_output, total_time),
            latency=PercentileStats.from_array(latencies),
            ttft=PercentileStats.from_array(ttfts),
            tpot=PercentileStats.from_array(tpots_arr),
            errors=errors,
        )

    def to_output(self) -> BenchmarkStatsOutput:
        """Convert to structured output model."""
        return BenchmarkStatsOutput(
            summary=SummaryStats(
                total_requests=self.total_requests,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                total_time_sec=self.total_time_sec,
            ),
            throughput=ThroughputStats(
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

    def print_summary(
        self,
        show_tokens: bool = True,
        show_ttft: bool = True,
        show_tpot: bool = True,
    ) -> None:
        """Print formatted benchmark summary."""
        success_rate = (
            100 * self.successful_requests / self.total_requests
            if self.total_requests > 0
            else 0
        )

        print("\nResults:")
        print(
            f"  Successful:       {self.successful_requests}/{self.total_requests} "
            f"({success_rate:.1f}%)"
        )
        print(f"  Failed:           {self.failed_requests}")
        print(f"  Total Time:       {self.total_time_sec:.2f}s")

        print("\nThroughput:")
        print(f"  Requests/sec:     {self.throughput_rps:.2f}")
        if show_tokens and self.tokens.total_output_tokens > 0:
            self.tokens.print_stats()

        print("\nE2E Latency (ms):")
        self.latency.print_stats()

        if show_ttft and self.ttft.avg > 0:
            print("\nTTFT - Time To First Token (ms):")
            self.ttft.print_stats()

        if show_tpot and self.tpot.avg > 0:
            print("\nTPOT - Time Per Output Token (ms):")
            self.tpot.print_stats()

        if self.errors:
            print("\nErrors:")
            for error, count in sorted(self.errors.items(), key=lambda x: -x[1]):
                print(f"  [{count}x] {error}")

