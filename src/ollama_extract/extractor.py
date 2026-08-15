"""Concurrent extraction engine with ThreadPoolExecutor, retry logic, and rich progress."""

import json
import os
import time
import logging
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


def auto_detect_workers(ollama_url: str, model_name: str, sample_inputs: list[str] | None = None) -> int:
    """Auto-detect the optimal worker count by running a warmup benchmark.

    Strategy:
        1. Detect CPU cores and whether Ollama is local or remote.
        2. Compute a heuristic initial count (local = min(cores, 4), remote = min(cores*2, 16)).
        3. Run a warmup benchmark with workers in {1, 2, heuristic} on 3 inputs.
        4. Pick the worker count with the best throughput.
    """
    from ollama_extract.model import Movie, get_movie_schema
    from ollama_extract.backends import get_backend

    cpu_cores = os.cpu_count() or 4
    local = is_local_ollama(ollama_url)

    if local:
        heuristic = min(cpu_cores, 4)
    else:
        heuristic = min(cpu_cores * 2, 16)

    # Warmup candidates
    candidates = sorted(set([1, 2, heuristic]))
    schema = get_movie_schema()

    # Use sample inputs or a default
    if sample_inputs is None:
        from ollama_extract.generator import TEST_INPUTS
        sample_inputs = TEST_INPUTS[:3]

    samples = sample_inputs[:3]
    best_throughput = 0.0
    best_workers = heuristic
    backend = get_backend("urllib", model_name, ollama_url)

    for workers in candidates:
        if workers > len(samples):
            continue
        start = time.perf_counter()
        success = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_single_extract, backend, text, schema): i
                for i, text in enumerate(samples)
            }
            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    success += 1
        elapsed = time.perf_counter() - start
        throughput = success / elapsed if elapsed > 0 else 0
        console.print(f"  [dim]warmup: {workers} workers → {throughput:.2f}/s ({success}/{len(samples)} ok)[/dim]")

        if throughput > best_throughput:
            best_throughput = throughput
            best_workers = workers

    return best_workers


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


def _single_extract(backend: Any, text: str, schema: dict[str, Any]) -> "ExtractionResult":
    """Low-level single extraction used by auto_detect_workers."""
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
