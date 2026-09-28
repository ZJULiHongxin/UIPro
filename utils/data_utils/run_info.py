"""Unified run-info helper for make_*_data scripts.

This module is the single source of truth for:

1. **Canonical task names** — the mapping from internal counter keys
   (``<task>_cnt``, snake_case) to the CamelCase task labels surfaced in
   reports and ``info.json`` files. Sister scripts must use the same
   identifiers so downstream tooling can aggregate stats across datasets.

2. **build_report** — formats a uniform, human-readable processing report.

3. **save_run_info** — writes the structured ``*_info.json`` alongside a
   dataset (and optionally a preview subset and a plain-text report).

The public surface is intentionally small; per-dataset specifics (such as
``per_part``, ``top_packages``, ``elem_class_stats_before_rebalance``, ...)
are passed via ``extra_sections`` so they sit beside — not inside — the
canonical fields.
"""
from __future__ import annotations

import datetime
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Multiprocessing helpers (shared by make_*_data scripts that use a Pool)
# ---------------------------------------------------------------------------
# Sample IDs look like ``autogui_<dataset>_<task>_<n>``. After parallel
# aggregation each worker's local ``<n>`` collides with others; this regex
# extracts the prefix so we can renumber globally per prefix.
_SAMPLE_ID_PATTERN = re.compile(r'^(.+)_(\d+)$')


def renumber_sample_ids(samples: List[Dict[str, Any]]) -> None:
    """Re-stripe per-task ``<n>`` suffixes after parallel aggregation.

    Each worker starts its local counter from 0, so after merging samples
    multiple entries can share an id like ``autogui_<ds>_<task>_0``.  This
    rewrites the trailing integer per ``<prefix>`` so the global output has
    unique, contiguous IDs again.  IDs with no trailing integer (e.g. the
    elemclass placeholder ``autogui_webui_elemclass_``) are left alone so a
    later rebalancing step can own those.
    """
    seq: Dict[str, int] = defaultdict(int)
    for sample in samples:
        sid = sample.get('id')
        if not isinstance(sid, str):
            continue
        m = _SAMPLE_ID_PATTERN.match(sid)
        if not m:
            continue
        prefix = m.group(1)
        sample['id'] = f'{prefix}_{seq[prefix]}'
        seq[prefix] += 1


def summarize_errors(
    errors: List[List[str]], *, examples_per_group: int = 5
) -> List[Dict[str, Any]]:
    """Group worker error tracebacks by tail line and keep a few example paths.

    Mass failures would otherwise dump hundreds of thousands of identical
    tracebacks into info.json. Grouping keeps the file readable while
    preserving enough detail to root-cause.

    Expects ``errors`` to be a list of ``[path, traceback_str]`` pairs.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for path, tb in errors:
        tail = tb.strip().splitlines()[-1] if tb.strip() else '<empty>'
        g = groups.setdefault(
            tail, {'count': 0, 'example_paths': [], 'traceback': tb}
        )
        g['count'] += 1
        if len(g['example_paths']) < examples_per_group:
            g['example_paths'].append(path)
    return sorted(
        ({'tail': tail, **info} for tail, info in groups.items()),
        key=lambda x: -x['count'],
    )


def print_error_summary(
    errors: List[List[str]],
    *,
    total_tasks: int,
    high_water_mark: float = 0.01,
) -> None:
    """Print a grouped end-of-run error summary to stdout.

    No-op when ``errors`` is empty. Adds a ``[HIGH]`` marker when the error
    rate exceeds ``high_water_mark`` (default: 1% of total tasks) — useful
    for catching regressions that break a worker code path in bulk.
    """
    if not errors:
        return
    from collections import Counter as _Counter
    tail_counts: _Counter = _Counter()
    for _, tb in errors:
        tail = tb.strip().splitlines()[-1] if tb.strip() else '<empty>'
        tail_counts[tail] += 1
    err_ratio = len(errors) / max(1, total_tasks)
    flag = " [HIGH]" if err_ratio > high_water_mark else ""
    print(
        f"\n⚠ {len(errors)} worker exceptions caught "
        f"({err_ratio:.1%} of tasks){flag}. Top error patterns:"
    )
    for tail, cnt in tail_counts.most_common(5):
        print(f"  {cnt:>8,}  {tail}")


# ---------------------------------------------------------------------------
# Canonical task naming
# ---------------------------------------------------------------------------
# Keys are the internal counter names used inside make_*_data scripts;
# values are the CamelCase labels rendered in reports and JSON outputs.
TASK_DISPLAY_NAMES: Dict[str, str] = {
    'text_loc_cnt': 'TextLoc',
    'ocr_cnt': 'OCR',
    'icongnd_cnt': 'IconGnd',
    'iconref_cnt': 'IconRef',
    'elemgnd_cnt': 'ElemGnd',
    'elemref_cnt': 'ElemRef',
    'elemclass_cnt': 'ElemClass',
    'intentgnd_cnt': 'IntentGnd',
    'widgetlist_cnt': 'WidgetList',
    'actionplanning_cnt': 'ActionPlanning',
    # MultiUI-flavoured tasks (also reusable by any future dataset that
    # generates captions, QA, or specialized OCR variants).
    'uicaption_cnt': 'UICaption',
    'embedcaption_cnt': 'EmbedCaption',
    'webqa_cnt': 'WebQA',
    'embedqa_cnt': 'EmbedQA',
    'titleocr_cnt': 'TitleOCR',
}


def canonical_samples_by_task(
    counters: Mapping[str, int],
    *,
    only_keys: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Return ``{CamelCase task name -> count}`` for known canonical counters.

    Args:
        counters: Mapping from ``<task>_cnt`` to integer count.
        only_keys: If given, restrict the output to this subset of canonical
            counter keys (preserves the canonical order).
    """
    keys = list(TASK_DISPLAY_NAMES.keys())
    if only_keys is not None:
        wanted = set(only_keys)
        keys = [k for k in keys if k in wanted]
    return {TASK_DISPLAY_NAMES[k]: int(counters.get(k, 0)) for k in keys}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_report(
    *,
    dataset: str,
    generated_at: str,
    num_samples: int,
    images: Mapping[str, Any],
    elements: Mapping[str, Any],
    samples_by_task: Mapping[str, int],
    extra_lines: Optional[List[str]] = None,
) -> str:
    """Compose the multi-section processing report shared across scripts."""
    valid_ratio = float(images.get('valid_ratio') or 0.0)
    invalid_ratio = float(elements.get('invalid_ratio') or 0.0)
    valid_imgs = int(images.get('valid') or 0)
    avg_per_valid = num_samples / valid_imgs if valid_imgs else 0.0

    task_lines = ''.join(
        f'  {name:<14}: {count}\n'
        for name, count in samples_by_task.items()
    )

    body = (
        '========================================\n'
        f'{dataset} Processing Report ({generated_at})\n'
        '========================================\n'
        f"#Samples Generated      : {num_samples}\n"
        f"#Valid Images           : {valid_imgs}\n"
        f"#Processed Images       : {images.get('processed', 0)}\n"
        f"Valid Image Ratio       : {valid_ratio:.2%}\n"
        f"#Invalid Elements       : {elements.get('invalid', 0)}\n"
        f"#Iterated Elements      : {elements.get('iterated', 0)}\n"
        f"Invalid Element Ratio   : {invalid_ratio:.2%}\n"
        f"Avg Samples / Valid Img : {avg_per_valid:.2f}\n"
        '----------------------------------------\n'
        'Task Counters:\n'
        f'{task_lines}'
    )
    if extra_lines:
        body += '----------------------------------------\n'
        body += ''.join(f'{line}\n' for line in extra_lines)
    body += '========================================'
    return body


# ---------------------------------------------------------------------------
# Pretty (colored) rendering — uses `rich` when available, else plain text.
# ---------------------------------------------------------------------------
def _print_pretty_report(
    *,
    dataset: str,
    generated_at: str,
    num_samples: int,
    images: Mapping[str, Any],
    elements: Mapping[str, Any],
    samples_by_task: Mapping[str, int],
    tasks_enabled: Optional[Mapping[str, bool]],
    plain_fallback: str,
) -> None:
    """Render the report with rich tables/colors.

    Color semantics (kept consistent across all make_*_data scripts):
      - cyan  : section headers, dataset name, task labels
      - green : healthy positive counts (>0)
      - dim   : zero/empty values
      - yellow: ratios / percentages
      - red   : invalid / error categories
      - magenta: configuration toggles
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
    except ImportError:
        print(plain_fallback)
        return

    console = Console()

    # --- Header panel -------------------------------------------------
    console.print(
        Panel.fit(
            f"[bold cyan]{dataset}[/bold cyan]  "
            f"[dim]Processing Report — {generated_at}[/dim]",
            border_style="cyan",
            box=box.DOUBLE,
        )
    )

    valid = int(images.get('valid') or 0)
    processed = int(images.get('processed') or 0)
    valid_ratio = float(images.get('valid_ratio') or 0.0)
    invalid = int(elements.get('invalid') or 0)
    iterated = int(elements.get('iterated') or 0)
    invalid_ratio = float(elements.get('invalid_ratio') or 0.0)
    avg_per_valid = num_samples / valid if valid else 0.0

    def _ratio_color(r: float, *, inverted: bool = False) -> str:
        """Green when in the healthy band, yellow mid-range, red otherwise."""
        if inverted:  # higher = worse (e.g., invalid ratio)
            if r < 0.10:
                return "green"
            if r < 0.30:
                return "yellow"
            return "red"
        if r >= 0.70:
            return "green"
        if r >= 0.40:
            return "yellow"
        return "red"

    # --- Summary table (two columns: Images / Elements / Samples) -----
    summary = Table(
        title="[bold]Overall Summary[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_justify="left",
    )
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_column("Ratio", justify="right")

    summary.add_row(
        "Samples generated",
        f"[bold green]{num_samples:,}[/bold green]",
        "—",
    )
    summary.add_row(
        "Images (valid / processed)",
        f"{valid:,} / {processed:,}",
        f"[{_ratio_color(valid_ratio)}]{valid_ratio:.2%}[/{_ratio_color(valid_ratio)}]",
    )
    # Sanity flag: invalid > iterated implies a caller-side tracking bug
    # (e.g., counting cumulative records against a per-run denominator).
    invalid_ratio_cell = (
        f"[{_ratio_color(invalid_ratio, inverted=True)}]{invalid_ratio:.2%}"
        f"[/{_ratio_color(invalid_ratio, inverted=True)}]"
    )
    if invalid > iterated and iterated > 0:
        invalid_ratio_cell = (
            f"[reverse red]{invalid_ratio:.2%} BUG[/reverse red]"
        )
    summary.add_row(
        "Elements (invalid / iterated)",
        f"[red]{invalid:,}[/red] / {iterated:,}",
        invalid_ratio_cell,
    )
    if invalid > iterated and iterated > 0:
        console.print(
            "[bold reverse red] ⚠  invalid > iterated [/bold reverse red] "
            "[red]The invalid count exceeds the iterated count; "
            "check whether you're mixing cumulative vs. per-run tallies.[/red]"
        )
    summary.add_row(
        "Avg samples / valid image",
        f"{avg_per_valid:.2f}",
        "—",
    )
    console.print(summary)

    # --- Task counters table ------------------------------------------
    tasks_table = Table(
        title="[bold]Task Counters[/bold]",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        title_justify="left",
    )
    tasks_table.add_column("Task", style="cyan")
    tasks_table.add_column("Count", justify="right")
    tasks_table.add_column("Enabled", justify="center")

    enabled_set = (
        {name for name, on in tasks_enabled.items() if on}
        if tasks_enabled is not None
        else set(samples_by_task.keys())
    )
    for name, count in samples_by_task.items():
        is_enabled = name in enabled_set
        count_style = "green" if count > 0 else "dim"
        if tasks_enabled is not None and not is_enabled and count == 0:
            # Skip purely-absent tasks when the script declared which it ran.
            continue
        enabled_cell = (
            "[green]●[/green]" if is_enabled else "[dim]○[/dim]"
        )
        tasks_table.add_row(
            name,
            f"[{count_style}]{count:,}[/{count_style}]",
            enabled_cell,
        )
    console.print(tasks_table)

    # --- Invalid breakdown (if provided) ------------------------------
    breakdown = elements.get('invalid_breakdown') or {}
    if breakdown:
        bd_table = Table(
            title="[bold]Invalid Elements Breakdown[/bold]",
            box=box.SIMPLE_HEAD,
            header_style="bold red",
            title_justify="left",
        )
        bd_table.add_column("Reason", style="red")
        bd_table.add_column("Count", justify="right")
        bd_table.add_column("Share", justify="right")
        total = sum(int(v) for v in breakdown.values()) or 1
        for reason, count in sorted(
            breakdown.items(), key=lambda kv: -int(kv[1])
        )[:15]:  # top-15 to avoid overwhelming output
            share = int(count) / total
            bd_table.add_row(
                str(reason),
                f"{int(count):,}",
                f"[yellow]{share:.1%}[/yellow]",
            )
        console.print(bd_table)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _ensure_ratio(d: Mapping[str, Any], numer: str, denom: str, key: str) -> Dict[str, Any]:
    """Return a copy of ``d`` with ``key`` populated as numer/denom (rounded)."""
    out = dict(d)
    if key not in out:
        n = out.get(numer) or 0
        de = out.get(denom) or 0
        out[key] = round(n / de, 4) if de else 0.0
    return out


def save_run_info(
    *,
    dataset: str,
    output_path: str,
    samples: List[Dict[str, Any]],
    counters: Mapping[str, int],
    config: Mapping[str, Any],
    images: Mapping[str, Any],
    elements: Mapping[str, Any],
    tasks_enabled: Optional[Mapping[str, bool]] = None,
    extra_sections: Optional[Mapping[str, Any]] = None,
    extra_report_lines: Optional[List[str]] = None,
    write_preview: bool = True,
    preview_size: int = 128,
    write_report_txt: bool = False,
    print_report: bool = True,
) -> Dict[str, Any]:
    """Persist a unified ``*_info.json`` (plus optional preview / report.txt).

    Args:
        dataset: Dataset name (e.g. ``'MobileViews'``). Used in the report
            header and as ``info['dataset']``.
        output_path: Path to the main samples ``.json``. Sibling
            ``_info.json`` / ``_sample.json`` / ``_report.txt`` filenames are
            derived from it.
        samples: The full sample list (used for ``num_samples`` and preview).
        counters: Per-task counter mapping. Keys should be canonical
            ``<task>_cnt`` names from :data:`TASK_DISPLAY_NAMES`.
        config: Configuration snapshot (scale, debug flag, paths, ...).
        images: Image stats. Must contain ``processed`` and ``valid``;
            ``valid_ratio`` is computed if absent.
        elements: Element stats. Must contain ``iterated`` and ``invalid``;
            ``invalid_ratio`` is computed if absent. May also carry
            ``invalid_breakdown`` and ``by_type`` dicts.
        tasks_enabled: Optional ``{CamelCase task: bool}`` map of enabled
            generators; merged into ``info['config']`` for traceability.
        extra_sections: Optional top-level sections appended to the JSON
            (e.g. ``per_part``, ``top_packages``, ``errors``).
        extra_report_lines: Optional plain strings appended to the report
            after the task counters block.
        write_preview: If True, also write a random ``_sample.json`` subset.
        preview_size: Max preview size.
        write_report_txt: If True, also write a plain-text ``_report.txt``.
        print_report: If True (default), print the report to stdout.

    Returns:
        The assembled info dict (the same content as written to disk).
    """
    generated_at = datetime.datetime.now().isoformat(timespec='seconds')
    num_samples = len(samples)

    images = _ensure_ratio(images, 'valid', 'processed', 'valid_ratio')
    elements = _ensure_ratio(elements, 'invalid', 'iterated', 'invalid_ratio')

    samples_by_task = canonical_samples_by_task(counters)
    valid_imgs = int(images.get('valid') or 0)
    avg_per_valid = num_samples / valid_imgs if valid_imgs else 0.0

    report = build_report(
        dataset=dataset,
        generated_at=generated_at,
        num_samples=num_samples,
        images=images,
        elements=elements,
        samples_by_task=samples_by_task,
        extra_lines=extra_report_lines,
    )

    merged_config = dict(config)
    if tasks_enabled is not None:
        merged_config['tasks_enabled'] = dict(tasks_enabled)

    info: Dict[str, Any] = {
        'dataset': dataset,
        'generated_at': generated_at,
        'config': merged_config,
        'images': images,
        'elements': elements,
        'samples': {
            'total': num_samples,
            'avg_per_valid_image': round(avg_per_valid, 2),
            'by_task': samples_by_task,
        },
        'outputs': {
            'samples_file': os.path.basename(output_path),
            'preview_file': os.path.basename(
                output_path.replace('.json', '_sample.json')
            ),
            'info_file': os.path.basename(
                output_path.replace('.json', '_info.json')
            ),
        },
        'report': report,
    }
    if extra_sections:
        for key, value in extra_sections.items():
            info[key] = value

    info_path = output_path.replace('.json', '_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2, default=str)

    if write_preview and samples:
        preview_path = output_path.replace('.json', '_sample.json')
        with open(preview_path, 'w') as f:
            json.dump(
                random.sample(samples, min(len(samples), preview_size)),
                f,
                indent=2,
            )

    if write_report_txt:
        with open(output_path.replace('.json', '_report.txt'), 'w') as f:
            f.write(report)

    if print_report:
        _print_pretty_report(
            dataset=dataset,
            generated_at=generated_at,
            num_samples=num_samples,
            images=images,
            elements=elements,
            samples_by_task=samples_by_task,
            tasks_enabled=tasks_enabled,
            plain_fallback=report,
        )

    return info
