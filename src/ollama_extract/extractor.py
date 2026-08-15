"""Concurrent extraction with ThreadPoolExecutor, retry, and rich progress."""

import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
from rich.table import Table
from rich.panel import Panel
from rich import box

from pydantic import ValidationError

from ollama_extract.model import Movie, get_movie_schema
from ollama_extract.backends import get_backend, BackendName, BackendResponse, OllamaBackend

BACKEND_NAMES = ["urllib", "requests", "ollama"]

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MAX_WORKERS = 4
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120


@dataclass
class ExtractionResult:
    """Per-input extraction result."""
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
    """Aggregated results from a batch run."""
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


def _p50(values: list[float]) -> float:
    if not values:
        return 0.0
    return values[len(values) // 2]


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    return values[int(len(values) * 0.9)]


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    return values[int(len(values) * 0.99)]


class ConcurrentExtractor:
    """Runs extractions concurrently across multiple backends.

    Uses ``ThreadPoolExecutor`` since the workload is I/O-bound (HTTP requests
    to a local Ollama instance).  Thread-pool size is capped by *max_workers*
    to avoid overwhelming the Ollama server.

    Sizing rationale (Goetz / Python docs):
        N_threads = cores × (1 + Wait/Service)
        For ~0.4 s LLM-wait and ~0.01 s service overhead that gives ~40 threads,
        but the local Ollama server caps useful concurrency at ~4–8 workers.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        ollama_url: str = "http://localhost:11434/api/generate",
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._schema = get_movie_schema()

    # ------------------------------------------------------------------
    #  Single extraction (with retry + metrics)
    # ------------------------------------------------------------------

    def _extract_one(
        self,
        backend: OllamaBackend,
        text: str,
        index: int,
    ) -> ExtractionResult:
        result = ExtractionResult(
            index=index,
            input_text=text,
            success=False,
            elapsed=0.0,
            backend=backend.name,
        )
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                start = time.perf_counter()
                resp = backend.generate(_build_prompt(text, self._schema), self._schema)
                elapsed = time.perf_counter() - start
                result.elapsed += elapsed

                result.prompt_tokens = resp.prompt_tokens
                result.eval_tokens = resp.eval_tokens
                result.total_duration_ms = resp.total_duration_ms

                movie = backend.validate(resp.content)
                result.success = True
                result.title = movie.title
                result.year = movie.year
                result.genres = movie.genres
                break
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    result.retries += 1
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} for #{index}: {e}")
                else:
                    result.error = str(e)
                    result.error_type = type(e).__name__

        return result

    # ------------------------------------------------------------------
    #  Batch run with progress display
    # ------------------------------------------------------------------

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
            show_progress: If True, display a Rich progress bar.

        Returns:
            Dict mapping backend name → ``BatchResult``.
        """
        backend_names = {"all": ["urllib", "requests", "ollama"]}.get(
            backend_name, [backend_name]
        )

        backends: list[OllamaBackend] = [
            get_backend(name, self.model_name, self.ollama_url) for name in backend_names
        ]

        results: dict[str, BatchResult] = {}

        for backend in backends:
            if show_progress:
                console.print(f"\n[cyan]Running {backend.name} backend on {len(inputs)} inputs "
                               f"with {self.max_workers} workers...[/cyan]")

            batch = self._run_single_backend(inputs, backend, show_progress)
            results[backend.name] = batch

            if show_progress:
                self._print_backend_summary(backend.name, batch)

        return results

    def _run_single_backend(
        self,
        inputs: list[str],
        backend: OllamaBackend,
        show_progress: bool,
    ) -> BatchResult:
        """Run all inputs through a single backend with concurrency."""
        results_by_index: dict[int, ExtractionResult] = {}

        if show_progress:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            with progress:
                task = progress.add_task(
                    f"[cyan]{backend.name:>12}", total=len(inputs)
                )
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._extract_one, backend, text, idx): idx
                        for idx, text in enumerate(inputs)
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        result = future.result()
                        results_by_index[idx] = result
                        progress.advance(task)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._extract_one, backend, text, idx): idx
                    for idx, text in enumerate(inputs)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results_by_index[idx] = future.result()

        # Order results by original index
        ordered = [results_by_index[i] for i in range(len(inputs))]
        return BatchResult(backend=backend.name, results=ordered)

    # ------------------------------------------------------------------
    #  Display
    # ------------------------------------------------------------------

    def _print_backend_summary(self, name: str, batch: BatchResult) -> None:
        latencies = batch.latencies
        table = Table(title=f"{name} — results", box=box.ROUNDED, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Inputs", str(len(batch.results)))
        table.add_row("Success", str(batch.success_count))
        table.add_row("Failures", str(batch.failure_count))
        table.add_row("First-attempt rate", f"{(len(batch.results) - sum(r.retries for r in batch.results if r.success)) / len(batch.results) * 100:.1f}%")

        if latencies:
            table.add_row("Latency min (s)", f"{min(latencies):.3f}")
            table.add_row("Latency p50 (s)", f"{_p50(latencies):.3f}")
            table.add_row("Latency p90 (s)", f"{_p90(latencies):.3f}")
            table.add_row("Latency p99 (s)", f"{_p99(latencies):.3f}")
            table.add_row("Latency max (s)", f"{max(latencies):.3f}")
            table.add_row("Latency mean (s)", f"{batch.avg_elapsed:.3f}")

        throughput = batch.success_count / batch.total_elapsed if batch.total_elapsed > 0 else 0
        table.add_row("Throughput (rps)", f"{throughput:.2f}")
        table.add_row("Total time (s)", f"{batch.total_elapsed:.2f}")

        console.print(table)

    def print_comparison(self, all_results: dict[str, BatchResult]) -> None:
        """Print a side-by-side comparison table across backends."""
        n = len(next(iter(all_results.values())).results)

        console.print()
        console.print(Panel.fit(
            f"[bold]Cross-Backend Title & Year Comparison[/bold] "
            f"(yellow = disagreement, red = failure)",
            border_style="cyan",
        ))

        table = Table(box=box.ASCII)
        table.add_column("#", justify="right", style="dim")
        for name in BACKEND_NAMES:
            table.add_column(name, overflow="fold")

        for i in range(n):
            cells = []
            titles = set()
            years = set()
            for name in BACKEND_NAMES:
                r = all_results[name].results[i]
                if r.success:
                    cell = f"{r.title} ({r.year})"
                    titles.add(r.title)
                    years.add(r.year)
                else:
                    cell = f"[red]FAIL: {r.error_type}[/red]"
                cells.append(cell)

            disagreement = len(titles) > 1 or len(years) > 1
            row_cells = []
            for cell in cells:
                if "FAIL" in cell:
                    row_cells.append(cell)
                elif disagreement:
                    row_cells.append(f"[yellow]{cell}[/yellow]")
                else:
                    row_cells.append(cell)

            table.add_row(str(i), *row_cells)

        console.print(table)

        # Consistency matrix
        console.print()
        console.print(Panel.fit("[bold]Consistency Matrix (title + year agreement)[/bold]", border_style="cyan"))
        consistency_table = Table(box=box.ASCII)
        consistency_table.add_column("Pair", style="cyan")
        consistency_table.add_column("Agreement", justify="right")

        pairs = [
            ("urllib", "requests"),
            ("urllib", "ollama"),
            ("requests", "ollama"),
        ]
        for a, b in pairs:
            if a in all_results and b in all_results:
                agreements = 0
                compared = 0
                for i in range(n):
                    ra, rb = all_results[a].results[i], all_results[b].results[i]
                    if ra.success and rb.success:
                        compared += 1
                        if ra.title == rb.title and ra.year == rb.year:
                            agreements += 1
                rate = agreements / compared * 100 if compared else 0
                color = "green" if rate >= 95 else ("yellow" if rate >= 80 else "red")
                consistency_table.add_row(f"{a} vs {b}", f"[{color}]{agreements}/{compared} ({rate:.1f}%)[/{color}]")

        console.print(consistency_table)


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def _build_prompt(raw_text: str, schema: dict[str, Any]) -> str:
    """Build a structured extraction prompt that includes the JSON schema."""
    return f"""You are a data extraction assistant.

Extract the movie information from the text below and return ONLY valid JSON that conforms to this JSON Schema:

{__import__('json').dumps(schema, indent=2)}

Rules:
1. Output ONLY valid JSON — no markdown fences, no preamble, no explanation.
2. Match the schema exactly; extra fields will cause validation to fail.
3. If a value is missing, use "Unknown" (for strings) or null (for integers).

Text to extract from:
---
{raw_text}
---"""
