"""Command-line interface for the Ollama extraction benchmark."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from ollama_extract.extractor import ConcurrentExtractor, DEFAULT_MAX_WORKERS
from ollama_extract.generator import TEST_INPUTS, generate_inputs

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-extract",
        description="Benchmark urllib, requests, and ollama-library backends for structured LLM extraction with Pydantic v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # 30 hand-crafted edge-cases on qwen2.5:0.5b (default)
  uv run ollama-extract

  # 200 generated inputs, 8 parallel workers
  uv run ollama-extract --count 200 --workers 8

  # Single backend, higher-quality model
  OLLAMA_MODEL=gemma4:12b uv run ollama-extract --backend ollama --count 50

  # Export results to JSON
  uv run ollama-extract --count 3000 --output results.json
""",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=30,
        help="Number of test inputs (default: 30). Values > 30 generate inputs programmatically.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Ollama model to use (default: from OLLAMA_MODEL env or qwen2.5:0.5b).",
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
        default=DEFAULT_MAX_WORKERS,
        help=f"Concurrent workers (default: {DEFAULT_MAX_WORKERS}).",
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
        help="Suppress rich progress bar and tables (still prints summary).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity (default: WARNING).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama /api/generate URL (default: http://localhost:11434/api/generate).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    model = args.model or os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b")
    n = args.count
    seed = args.seed

    # Prepare inputs
    if n <= len(TEST_INPUTS):
        inputs = TEST_INPUTS[:n]
        source = "hand-crafted edge-cases"
    else:
        # Mix: use hand-crafted first, then generate the rest
        inputs = list(TEST_INPUTS) + generate_inputs(n - len(TEST_INPUTS), seed=seed)
        source = f"mixed ({len(TEST_INPUTS)} hand-crafted + {n - len(TEST_INPUTS)} generated, seed={seed})"

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

    # --- Always print a summary table ---
    console.print()
    console.print(f"[bold]Model: {model}  |  Inputs: {n}  |  Workers: {args.workers}[/bold]")

    summary = Table(title="Summary", box=box.ROUNDED, show_header=True)
    summary.add_column("Backend", style="cyan")
    summary.add_column("Success", justify="right", style="green")
    summary.add_column("Failures", justify="right", style="red")
    summary.add_column("Total (s)", justify="right")
    summary.add_column("Avg (s)", justify="right")
    summary.add_column("Throughput", justify="right")

    for label in results:
        batch = results[label]
        ok = batch.success_count
        fail = batch.failure_count
        t = batch.total_elapsed
        avg = batch.avg_elapsed
        tp = ok / t if t > 0 else 0
        summary.add_row(label, f"{ok}/{n}", str(fail), f"{t:.2f}", f"{avg:.3f}", f"{tp:.2f}/s")

    total_ok = sum(b.success_count for b in results.values())
    console.print(summary)
    console.print(f"\nTotal successful extractions: {total_ok}/{n * len(results)}")

    # Cross-backend comparison (only for small counts and verbose mode)
    if not args.quiet and len(results) > 1 and n <= 100:
        extractor.print_comparison(results)

    # JSON output
    if args.output:
        _write_json(results, args.output)
        console.print(f"\n[green]Results written to {args.output}[/green]")

    return 0


def _write_json(results: dict, path: Path) -> None:
    """Export full results as JSON."""
    output = {}
    for backend_name, batch in results.items():
        output[backend_name] = {
            "backend": batch.backend,
            "success_count": batch.success_count,
            "failure_count": batch.failure_count,
            "total_elapsed": batch.total_elapsed,
            "avg_elapsed": batch.avg_elapsed,
            "latencies": {
                "min": min(batch.latencies) if batch.latencies else None,
                "p50": batch.latencies[len(batch.latencies) // 2] if batch.latencies else None,
                "p90": batch.latencies[int(len(batch.latencies) * 0.9)] if batch.latencies else None,
                "max": max(batch.latencies) if batch.latencies else None,
            },
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
