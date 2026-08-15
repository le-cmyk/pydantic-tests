"""Command-line interface for ollama_extract."""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ollama_extract.extractor import ConcurrentExtractor
from ollama_extract.generator import TEST_INPUTS, generate_inputs
from ollama_extract.backends import list_backends
from ollama_extract.extractor import BatchResult

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-extract",
        description=(
            "Benchmark urllib, requests, and ollama-library backends for "
            "structured LLM extraction with Pydantic v2 validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # 30 hand-crafted edge-cases (default)
  uv run ollama-extract

  # 200 inputs, 4 parallel workers
  uv run ollama-extract --count 200 --workers 4

  # 3000 inputs, single backend, JSON export
  uv run ollama-extract --count 3000 --backend requests --output results.json

  # Higher-quality model
  OLLAMA_MODEL=gemma4:12b uv run ollama-extract --count 50

  # Verbose output: show samples, full comparison tables
  SHOW_SAMPLES=1 uv run ollama-extract --count 30
""",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=30,
        help="Number of test inputs (default: 30). >30 generates inputs programmatically.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ollama model (default: $OLLAMA_MODEL or qwen2.5:0.5b).",
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["urllib", "requests", "ollama", "all"],
        default="all",
        help="Which backend(s) to run (default: all).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Concurrent ThreadPoolExecutor workers (default: 4).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for generated inputs (default: 42).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write full results as JSON to this file.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress bar and detailed tables.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
    )
    return parser


def _prepare_inputs(count: int, seed: int) -> tuple[list[str], str]:
    if count <= len(TEST_INPUTS):
        return TEST_INPUTS[:count], "hand-crafted edge-cases"
    inputs = list(TEST_INPUTS) + generate_inputs(count - len(TEST_INPUTS), seed=seed)
    return inputs, f"mixed ({len(TEST_INPUTS)} hand-crafted + {count - len(TEST_INPUTS)} generated, seed={seed})"


def _print_comparison(all_results: dict[str, BatchResult]) -> None:
    """Cross-backend comparison tables with color-coded differences."""
    names = list(all_results.keys())
    n = len(next(iter(all_results.values())).results)

    console.print()

    # --- Per-input title/year table ---
    table = Table(title="Cross-Backend: Title & Year (yellow=disagreement, red=failure)", box=box.ASCII)
    table.add_column("#", justify="right", style="dim")
    for name in names:
        table.add_column(name)

    for i in range(min(n, 100)):
        cells = []
        titles_seen = set()
        years_seen = set()
        for name in names:
            r = all_results[name].results[i]
            if r.success:
                cell = f"{r.title} ({r.year})"
                titles_seen.add(r.title)
                years_seen.add(r.year)
            else:
                cell = f"FAIL: {r.error_type}"
            cells.append(cell)

        disagreement = len(titles_seen) > 1 or len(years_seen) > 1
        for idx, cell in enumerate(cells):
            if "FAIL" in cell:
                cells[idx] = f"[red]{cell}[/red]"
            elif disagreement:
                cells[idx] = f"[yellow]{cell}[/yellow]"

        table.add_row(str(i), *cells)

    console.print(table)

    # --- Consistency matrix ---
    if len(names) > 1:
        consistency = Table(title="Consistency Matrix (title + year agreement)", box=box.ASCII)
        consistency.add_column("Pair", style="cyan")
        consistency.add_column("Agreement", justify="right")

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                agreements = 0
                compared = 0
                for idx in range(n):
                    ra, rb = all_results[a].results[idx], all_results[b].results[idx]
                    if ra.success and rb.success:
                        compared += 1
                        if ra.title == rb.title and ra.year == rb.year:
                            agreements += 1
                rate = agreements / compared * 100 if compared else 0
                color = "green" if rate >= 95 else ("yellow" if rate >= 80 else "red")
                consistency.add_row(
                    f"{a} vs {b}",
                    f"[{color}]{agreements}/{compared} ({rate:.1f}%)[/{color}]",
                )
        console.print(consistency)


def _print_summary(all_results: dict[str, BatchResult], n: int) -> None:
    console.print()
    summary = Table(title="Summary", box=box.ROUNDED, show_header=True)
    summary.add_column("Backend", style="cyan")
    summary.add_column("Success", justify="right", style="green")
    summary.add_column("Failures", justify="right", style="red")
    summary.add_column("Total (s)", justify="right")
    summary.add_column("Avg (s)", justify="right")
    summary.add_column("p50 (s)", justify="right")
    summary.add_column("Throughput", justify="right")

    for name, batch in all_results.items():
        lats = batch.latencies
        p50 = lats[len(lats) // 2] if lats else 0
        tp = batch.success_count / batch.total_elapsed if batch.total_elapsed > 0 else 0
        summary.add_row(
            name,
            f"{batch.success_count}/{n}",
            str(batch.failure_count),
            f"{batch.total_elapsed:.2f}",
            f"{batch.avg_elapsed:.3f}",
            f"{p50:.3f}",
            f"{tp:.2f}/s",
        )

    total_ok = sum(b.success_count for b in all_results.values())
    console.print(summary)
    console.print(f"\nTotal successful extractions: {total_ok}/{n * len(all_results)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    model = args.model or os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
    inputs, source = _prepare_inputs(args.count, args.seed)
    n = len(inputs)
    show_progress = not args.quiet

    if show_progress:
        console.print(Panel.fit(
            f"[bold cyan]ollama-extract[/bold cyan]\n"
            f"Model: [green]{model}[/green]  |  Inputs: {n} ({source})  |  "
            f"Workers: {args.workers}  |  Backend(s): {args.backend}",
            border_style="cyan",
        ))

    extractor = ConcurrentExtractor(
        model_name=model,
        ollama_url=args.ollama_url,
        max_workers=args.workers,
    )

    results = extractor.run(
        inputs,
        backend_name=args.backend,
        show_progress=show_progress,
    )

    _print_summary(results, n)

    if not args.quiet and len(results) > 1 and n <= 100:
        _print_comparison(results)

    if args.output:
        _write_json(results, args.output, n)
        console.print(f"\n[green]Results written to {args.output}[/green]")

    return 0


def _write_json(all_results: dict[str, BatchResult], path: Path, n: int) -> None:
    output = {}
    for backend_name, batch in all_results.items():
        output[backend_name] = {
            "backend": batch.backend,
            "inputs": n,
            "success_count": batch.success_count,
            "failure_count": batch.failure_count,
            "total_elapsed": batch.total_elapsed,
            "avg_elapsed": batch.avg_elapsed,
            "p50_latency": batch.latencies[len(batch.latencies) // 2] if batch.latencies else None,
            "p90_latency": batch.latencies[int(len(batch.latencies) * 0.9)] if batch.latencies else None,
            "throughput_rps": batch.success_count / batch.total_elapsed if batch.total_elapsed > 0 else 0,
            "results": [
                {
                    "index": r.index,
                    "success": r.success,
                    "elapsed": r.elapsed,
                    "retries": r.retries,
                    "title": r.title,
                    "year": r.year,
                    "genres": r.genres,
                    "error": r.error,
                    "error_type": r.error_type,
                    "prompt_tokens": r.prompt_tokens,
                    "eval_tokens": r.eval_tokens,
                }
                for r in batch.results
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
