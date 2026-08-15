"""Concurrent extraction engine with ThreadPoolExecutor, retry logic, and rich progress."""

import json
import os
import re
import sys
import time
import logging
import subprocess
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.table import Table as RichTable
from rich.panel import Panel
from rich import box

from ollama_extract.model import get_movie_schema
from ollama_extract.backends import get_backend

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MAX_WORKERS = 4
MAX_RETRIES = 3
BACKEND_NAMES = ["urllib", "requests", "ollama"]

# Localhost patterns to detect local Ollama
_LOCAL_PATTERNS = ("localhost", "127.0.0.1", "0.0.0.0")


def is_local_ollama(ollama_url: str) -> bool:
    """Check if the Ollama endpoint is on localhost."""
    return any(p in ollama_url for p in _LOCAL_PATTERNS)


def get_ollama_parallel() -> int | None:
    """Detect the server's OLLAMA_NUM_PARALLEL setting.

    Tries to read it from the environment of the running Ollama process.
    Returns None if it can't be determined (defaults to 1).
    """
    # Check local environment first
    env_val = os.environ.get("OLLAMA_NUM_PARALLEL")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass

    # Check the running Ollama process environment
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["ps", "ax", "-o", "command", "-E"],
                capture_output=True, text=True, timeout=5,
            )
        else:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True, timeout=5,
            )
        for line in result.stdout.splitlines():
            if "ollama" in line.lower() and "serve" in line.lower():
                match = re.search(r"OLLAMA_NUM_PARALLEL=(\d+)", line)
                if match:
                    return int(match.group(1))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def auto_detect_workers(
    ollama_url: str,
    model_name: str,
    sample_inputs: list[str] | None = None,
) -> int:
    """Auto-detect the optimal client-side worker count via a warmup benchmark.

    Strategy:
        1. Detect whether Ollama is local or remote.
        2. Read ``OLLAMA_NUM_PARALLEL`` if available (default 1).
        3. If local with server parallel=1: return 1 (sequential server → sequential client).
        4. Otherwise, build a candidate pool of worker counts (powers of 2 +
           server_parallel).
        5. Run a warmup benchmark with 30 inputs for each candidate.
        6. Pick the worker count with the best throughput, penalizing candidates
           where latency grew faster than throughput (queue contention).
    """
    cpu_cores = os.cpu_count() or 4
    local = is_local_ollama(ollama_url)
    server_parallel = get_ollama_parallel() or 1

    # Fast path: local Ollama with no server parallelism → 1 worker
    if server_parallel == 1 and local:
        console.print(
            f"  [dim]Auto-detect: server parallel=1 (local), "
            f"using 1 worker (sequential server → sequential client is optimal)[/dim]"
        )
        schema = get_movie_schema()
        if sample_inputs is None:
            from ollama_extract.generator import TEST_INPUTS
            sample_inputs = TEST_INPUTS[:12]
        backend = get_backend("urllib", model_name, ollama_url)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(_timed_extract, backend, text, schema)
                for text in sample_inputs[:12]
            ]
            for f in as_completed(futures):
                f.result()
        elapsed = time.perf_counter() - start
        throughput = 12 / elapsed
        console.print(
            f"  [green]✓ Confirmed 1 worker: {throughput:.2f}/s "
            f"({elapsed:.1f}s for 12 inputs)[/green]"
        )
        return 1

    # Build candidate pool
    max_candidate = max(8, server_parallel * 2)
    if not local:
        max_candidate = max(max_candidate, 16)

    candidates: list[int] = [1]
    w = 2
    while w <= max_candidate:
        if w not in candidates:
            candidates.append(w)
        w *= 2
    if server_parallel > 1 and server_parallel not in candidates:
        candidates.append(server_parallel)
    sp2 = server_parallel * 2
    if sp2 > max_candidate and sp2 <= 32 and sp2 not in candidates:
        candidates.append(sp2)

    candidates = sorted(c for c in candidates if c <= 32)

    schema = get_movie_schema()

    if sample_inputs is None:
        from ollama_extract.generator import TEST_INPUTS
        sample_inputs = TEST_INPUTS[:30]

    benchmark_inputs = sample_inputs[:30]
    backend = get_backend("urllib", model_name, ollama_url)

    console.print(
        f"  [dim]Auto-detecting optimal workers "
        f"(server parallel={server_parallel}, local={local}, "
        f"candidates={candidates}, warmup={len(benchmark_inputs)} inputs)...[/dim]"
    )

    # Pre-warm: send 1 request to load the model into VRAM
    try:
        _timed_extract(backend, benchmark_inputs[0], schema)
        console.print(f"  [dim]Model warmed up.[/dim]")
    except Exception:
        pass

    results: dict[int, tuple[float, float, float, float]] = {}

    for workers in candidates:
        start = time.perf_counter()
        latencies: list[float] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_timed_extract, backend, text, schema): i
                for i, text in enumerate(benchmark_inputs)
            }
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    latencies.append(result.elapsed)

        elapsed = time.perf_counter() - start
        success = len(latencies)
        throughput = success / elapsed if elapsed > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        sorted_lats = sorted(latencies)
        p90 = sorted_lats[int(len(sorted_lats) * 0.9)] if sorted_lats else 0
        p50 = sorted_lats[len(sorted_lats) // 2] if sorted_lats else 0

        results[workers] = (throughput, avg_latency, p90, p50)
        console.print(
            f"  [dim]warmup: {workers:>3} workers → {throughput:.2f}/s "
            f"(p50 {p50:.2f}s, avg {avg_latency:.2f}s, p90 {p90:.2f}s, "
            f"{success}/{len(benchmark_inputs)} ok, {elapsed:.1f}s)[/dim]"
        )

    # Select best: highest throughput, but penalize candidates that show
    # diminishing returns (latency grew faster than throughput).
    # effective_ratio = (throughput_ratio) / (latency_ratio)
    # A value >= 1.0 means parallelization is genuinely beneficial.
    best_throughput = 0.0
    best_workers = DEFAULT_MAX_WORKERS
    baseline_avg = results.get(1, (0, 0, 0, 0))[1]
    baseline_tp = results.get(1, (0, 0, 0, 0))[0]

    for workers, (throughput, avg_lat, p90, p50) in results.items():
        if baseline_avg > 0 and baseline_tp > 0:
            latency_ratio = avg_lat / baseline_avg
            throughput_ratio = throughput / baseline_tp
            effective_ratio = throughput_ratio / latency_ratio if latency_ratio > 0 else 0

            # For local Ollama, require strong evidence of parallelism benefit
            threshold = 0.95 if local else 0.8

            if effective_ratio < threshold:
                console.print(
                    f"    [yellow]⚠ {workers} workers: throughput {throughput_ratio:.1f}x, "
                    f"latency {latency_ratio:.1f}x → diminishing returns "
                    f"(effective ratio {effective_ratio:.2f} < {threshold}), "
                    f"penalizing[/yellow]"
                )
                throughput *= 0.1

        if throughput > best_throughput:
            best_throughput = throughput
            best_workers = workers

    console.print(
        f"  [green]✓ Selected {best_workers} workers "
        f"(best throughput: {best_throughput:.2f}/s)[/green]"
    )
    return best_workers


def _timed_extract(backend: Any, text: str, schema: dict[str, Any]) -> "ExtractionResult":
    """Single extraction that records precise per-request latency."""
    from ollama_extract.model import Movie

    result = ExtractionResult(
        index=0,
        input_text=text,
        success=False,
        elapsed=0.0,
        backend=backend.name,
    )
    start = time.perf_counter()
    try:
        resp = backend.generate(_build_prompt(text, schema), schema)
        result.elapsed = time.perf_counter() - start

        result.prompt_tokens = resp.prompt_tokens
        result.eval_tokens = resp.eval_tokens
        result.total_duration_ms = resp.total_duration_ms

        movie = backend.validate(resp.content)
        result.success = True
        result.title = movie.title
        result.year = movie.year
        result.genres = movie.genres
    except Exception as e:
        result.elapsed = time.perf_counter() - start
        result.error = str(e)
        result.error_type = type(e).__name__
    return result


def _build_prompt(raw_text: str, schema: dict[str, Any]) -> str:
    return f"""You are a data extraction assistant.

Extract the movie information from the text below and return ONLY valid JSON that conforms to this JSON Schema:

{json.dumps(schema, indent=2)}

Rules:
1. Output ONLY valid JSON — no markdown fences, no preamble, no explanation.
2. Match the schema exactly; extra fields will cause validation to fail.
3. If a value is missing, use "Unknown" (for strings) or null (for integers).

Text to extract from:
---
{raw_text}
---"""


# ---------------------------------------------------------------------------
#  Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    index: int
    input_text: str
    success: bool
    elapsed: float
    retries: int = 0
    prompt_tokens: int | None = None
    eval_tokens: int | None = None
    total_duration_ms: int | None = None
    backend: str = ""
    title: str | None = None
    year: int | None = None
    genres: list[str] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None


@dataclass
class BatchResult:
    backend: str
    results: list[ExtractionResult]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return len(self.results) - self.success_count

    @property
    def total_elapsed(self) -> float:
        return sum(r.elapsed for r in self.results)

    @property
    def avg_elapsed(self) -> float:
        s = [r.elapsed for r in self.results if r.success]
        return sum(s) / len(s) if s else 0.0

    @property
    def latencies(self) -> list[float]:
        return sorted(r.elapsed for r in self.results if r.success)

    @property
    def first_attempt_success_rate(self) -> float:
        if not self.results:
            return 100.0
        sa = sum(1 for r in self.results if r.success and r.retries == 0)
        return sa / len(self.results) * 100


# ---------------------------------------------------------------------------
#  Interactive live display
# ---------------------------------------------------------------------------

class _LiveResults:
    """Thread-safe live-updating results table for interactive mode."""

    def __init__(self, n: int, backend_names: list[str], start_time: float):
        self.n = n
        self.backend_names = backend_names
        self.start_time = start_time
        self.completed: dict[str, dict[int, ExtractionResult]] = {
            name: {} for name in backend_names
        }
        self.lock = __import__("threading").Lock()

    def add_result(self, backend: str, result: ExtractionResult) -> None:
        with self.lock:
            self.completed[backend][result.index] = result

    def render(self) -> RichTable:
        with self.lock:
            table = RichTable(box=box.MINIMAL, show_header=True, header_style="bold cyan")
            table.add_column("#", justify="right", style="dim", width=4)
            for name in self.backend_names:
                table.add_column(name, overflow="fold", max_width=30)

            total_completed = sum(len(v) for v in self.completed.values())
            total_success = 0
            for i in range(self.n):
                row = [str(i)]
                for name in self.backend_names:
                    r = self.completed[name].get(i)
                    if r is None:
                        row.append("[yellow]⏳[/yellow]")
                    elif r.success:
                        row.append(f"✅ {r.title} ({r.year})")
                    else:
                        row.append(f"[red]❌ {r.error_type}[/red]")
                table.add_row(*row)

            elapsed = time.time() - self.start_time
            throughput = total_completed / elapsed if elapsed > 0 else 0
            success_rate = (total_success / total_completed * 100) if total_completed else 0

            panel = Panel(
                table,
                title=(
                    f"[bold]Live Extraction[/bold]  "
                    f"[green]✓ {total_success}/{total_completed}[/green]  "
                    f"[yellow]⏳ {total_completed - total_success}/{self.n * len(self.backend_names)}[yellow]  "
                    f"[cyan]⏱ {elapsed:.1f}s[/cyan]  "
                    f"[magenta]{throughput:.1f}/s[/magenta]  "
                    f"[dim]workers: auto-tuned[/dim]"
                ),
                border_style="cyan",
            )
            return panel


# ---------------------------------------------------------------------------
#  Concurrent extractor
# ---------------------------------------------------------------------------

class ConcurrentExtractor:
    """Runs extractions concurrently across multiple backends.

    Uses ``ThreadPoolExecutor`` — the workload is I/O-bound (HTTP to an Ollama
    server + LLM inference time dominates).  Thread-pool size defaults to 4
    workers, which is the empirically optimal concurrency for local
    qwen2.5:0.5b (higher worker counts cause server-side latency degradation).

    Pass ``max_workers=0`` to auto-detect the optimal count via a warmup benchmark.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        ollama_url: str = "http://localhost:11434/api/generate",
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_retries: int = MAX_RETRIES,
        interactive: bool = False,
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_retries = max_retries
        self.interactive = interactive
        self._schema = get_movie_schema()

        if max_workers <= 0:
            self.max_workers = auto_detect_workers(ollama_url, model_name)
        else:
            self.max_workers = max_workers

    def _extract_one(self, backend: Any, text: str, index: int) -> ExtractionResult:
        """Single extraction with retry logic.  Must be thread-safe."""
        result = ExtractionResult(
            index=index,
            input_text=text,
            success=False,
            elapsed=0.0,
            backend=backend.name,
        )

        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                resp = backend.generate(_build_prompt(text, self._schema), self._schema)
                result.elapsed += time.perf_counter() - start

                result.prompt_tokens = resp.prompt_tokens
                result.eval_tokens = resp.eval_tokens
                result.total_duration_ms = resp.total_duration_ms

                movie = backend.validate(resp.content)
                result.success = True
                result.title = movie.title
                result.year = movie.year
                result.genres = movie.genres
                return result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    result.retries += 1
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} for #{index}: {e}")
                else:
                    result.error = str(e)
                    result.error_type = type(e).__name__

        logger.warning(f"{backend.name} #[index] failed after {self.max_retries} attempts — {result.error_type}")
        return result

    def _run_single_backend(
        self,
        inputs: list[str],
        backend: Any,
        show_progress: bool,
    ) -> BatchResult:
        """Run all inputs through a single backend with ThreadPoolExecutor."""
        results_by_index: dict[int, ExtractionResult] = {}
        live: _LiveResults | None = None
        live_display: Live | None = None

        if show_progress and self.interactive:
            live = _LiveResults(
                n=len(inputs),
                backend_names=[backend.name],
                start_time=time.time(),
            )
            live_display = Live(live.render, console=console, refresh_per_second=10, transient=False)

        if show_progress and live_display is None:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            )
        else:
            progress = None

        try:
            if progress:
                progress.start()
                task = progress.add_task(f"[cyan]{backend.name:>12}", total=len(inputs))

            if live_display:
                live_display.start()

            with ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix=f"{backend.name}-worker",
            ) as executor:
                future_to_idx = {
                    executor.submit(self._extract_one, backend, text, idx): idx
                    for idx, text in enumerate(inputs)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    results_by_index[idx] = result

                    if live:
                        live.add_result(backend.name, result)
                        live_display.update(live.render)
                    if progress:
                        progress.advance(task)

        finally:
            if progress:
                progress.stop()
            if live_display:
                live_display.stop()

        ordered = [results_by_index[i] for i in range(len(inputs))]
        return BatchResult(backend=backend.name, results=ordered)

    def run(
        self,
        inputs: list[str],
        backend_name: str = "all",
        show_progress: bool = True,
    ) -> dict[str, BatchResult]:
        """Run extractions for all *inputs* across the specified backend(s).

        Args:
            inputs: List of natural-language movie descriptions.
            backend_name: "urllib", "requests", "ollama", or "all".
            show_progress: If True, display Rich progress bar per backend.

        Returns:
            Dict mapping backend name → ``BatchResult``.
        """
        names = BACKEND_NAMES if backend_name == "all" else [backend_name]
        all_results: dict[str, BatchResult] = {}

        for name in names:
            backend = get_backend(name, self.model_name, self.ollama_url)
            if show_progress:
                console.print(
                    f"\n[cyan]  Running {name} backend on {len(inputs)} inputs "
                    f"with {self.max_workers} workers...[/cyan]"
                )
            batch = self._run_single_backend(inputs, backend, show_progress)
            all_results[name] = batch

        return all_results
