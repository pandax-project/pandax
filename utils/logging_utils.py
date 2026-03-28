"""Shared CSV logging helpers for rewrite and agent flows."""

import csv
from pathlib import Path
from typing import Any

def log_rewrite_timing(
    *,
    nb_path: Path,
    benchmark_name: str,
    cell_idx: int,
    try_num: int,
    category: str,
    elapsed_seconds: float,
) -> None:
    """Append one wall-time measurement for rewrite flow to CSV.

    CSV output path:
        <nb_path.parent>/rewrite_wall_time_timings.csv

    Row schema:
        benchmark_name, cell_index, try_number, category, elapsed_seconds
    """
    csv_path = nb_path.parent / "rewrite_wall_time_timings.csv"
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "benchmark_name",
                    "cell_index",
                    "try_number",
                    "category",
                    "elapsed_seconds",
                ]
            )
        writer.writerow(
            [
                benchmark_name,
                cell_idx,
                try_num,
                category,
                f"{elapsed_seconds:.6f}",
            ]
        )


def _to_int_or_none(value: Any) -> int | None:
    """Convert token-like values to int, returning None when unavailable."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_token_usage(result: Any) -> tuple[int | None, int | None, int | None]:
    """Extract (prompt, response, total) tokens from a Runner result."""
    usage_candidates: list[Any] = []

    direct_usage = getattr(result, "usage", None)
    if direct_usage is not None:
        usage_candidates.append(direct_usage)

    for attr in ("raw_responses", "responses"):
        responses = getattr(result, attr, None)
        if responses:
            for response in responses:
                usage = getattr(response, "usage", None)
                if usage is not None:
                    usage_candidates.append(usage)

    for usage in usage_candidates:
        prompt = _to_int_or_none(
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
        )
        response = _to_int_or_none(
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
        )
        total = _to_int_or_none(getattr(usage, "total_tokens", None))
        if prompt is not None or response is not None or total is not None:
            return prompt, response, total

    return None, None, None


def log_agent_token_usage(
    *,
    output_dir: Path,
    benchmark_name: str,
    cell_index: int,
    try_number: int,
    category: str,
    result: Any,
) -> None:
    """Append agent token usage for one invocation to CSV."""
    csv_path = output_dir / "rewrite_agent_token_usage.csv"
    file_exists = csv_path.exists()
    prompt_tokens, response_tokens, total_tokens = extract_token_usage(result)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "benchmark_name",
                    "cell_index",
                    "try_number",
                    "category",
                    "prompt_tokens",
                    "response_tokens",
                    "total_tokens",
                ]
            )
        writer.writerow(
            [
                benchmark_name,
                cell_index,
                try_number,
                category,
                prompt_tokens if prompt_tokens is not None else "",
                response_tokens if response_tokens is not None else "",
                total_tokens if total_tokens is not None else "",
            ]
        )
