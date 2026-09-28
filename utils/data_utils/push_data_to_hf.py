"""Push a processed UIPro dataset to a Hugging Face dataset repo.

Given the full ``*.json`` file produced by one of the ``make_*_data`` scripts
(e.g. ``WebUI_scale1000_485k.json``), this script:

  1. Loads the samples and collects every ``image`` path they reference.
  2. Bundles those images into a single ``.zip`` (deduplicated, preserving
     the relative directory layout so paths inside the JSON still resolve).
  3. Uploads the zip plus the sibling ``_sample.json`` and ``_info.json``
     files to a Hugging Face dataset repository.

Example:
  python utils/data_utils/push_data_to_hf.py \\
      --samples data/WebUI_processed/WebUI_scale1000_485k.json \\
      --images-root data \\
      --repo-id your-username/uipro-webui

Auth resolves in this order: ``--token`` flag → ``HF_TOKEN`` env var →
the cached login from ``hf auth login``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from tqdm import tqdm

try:
    from huggingface_hub import HfApi, create_repo
    # HfHubHTTPError moved between submodules across hub versions: it lives in
    # `huggingface_hub.errors` on >=0.25 and in `huggingface_hub.utils` on the
    # 0.24.x line. Try both so the script runs on whatever the reproducer has.
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    sys.exit(
        "huggingface_hub is required. Install with `pip install huggingface_hub`."
    )


# A minimal `.gitattributes` so the giant payloads (images.zip + the full
# samples .json) go through LFS instead of regular Git. Without this the Hub
# rejects pushes >100MB for non-LFS files.
GITATTRIBUTES_CONTENT = """\
*.zip filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
"""


def collect_image_paths(samples: list) -> Set[str]:
    """Return the deduplicated set of relative image paths used in ``samples``.

    Handles both ``sample['image']`` being a single string (the common case)
    and a list of strings (multi-image grounding tasks).
    """
    paths: Set[str] = set()
    for s in samples:
        img = s.get('image') if isinstance(s, dict) else None
        if isinstance(img, str):
            paths.add(img)
        elif isinstance(img, list):
            for p in img:
                if isinstance(p, str):
                    paths.add(p)
    return paths


def build_image_zip(
    images_root: Path,
    rel_paths: Set[str],
    zip_path: Path,
    workers: int = 16,
) -> Tuple[int, int]:
    """Write a zip of the referenced images and return ``(added, missing)``.

    Paths inside the zip mirror the ``image`` field in the dataset JSON, so
    downstream consumers can extract the zip wherever they like and the
    sample-to-image lookup keeps working.

    Uses ``ZIP_STORED`` (no compression) because PNG/JPG/WebP are already
    compressed — deflating them burns CPU for ~0% size reduction. File reads
    run in a thread pool so slow per-file I/O (esp. on networked storage)
    overlaps with the single-threaded zip writer.
    """
    added = missing = 0
    rels_sorted = sorted(rel_paths)

    def read_one(rel: str) -> Tuple[str, Optional[bytes]]:
        full = images_root / rel
        try:
            with open(full, 'rb') as fh:
                return rel, fh.read()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            return rel, None

    # Sliding window of in-flight reads bounds peak memory to
    # ~window_size files × avg image size.
    window_size = max(workers * 4, 32)

    # ``allowZip64=True`` lifts the 4 GiB / 65k-entry cap for big datasets.
    with zipfile.ZipFile(
        zip_path, 'w', zipfile.ZIP_STORED, allowZip64=True
    ) as zf, ThreadPoolExecutor(max_workers=workers) as ex:
        in_flight: deque = deque()
        rels_iter = iter(rels_sorted)

        for rel in rels_iter:
            in_flight.append(ex.submit(read_one, rel))
            if len(in_flight) >= window_size:
                break

        with tqdm(total=len(rels_sorted), desc="Zipping images", smoothing=0.05) as pbar:
            for next_rel in rels_iter:
                rel, data = in_flight.popleft().result()
                if data is None:
                    missing += 1
                else:
                    zf.writestr(rel, data)
                    added += 1
                pbar.update(1)
                in_flight.append(ex.submit(read_one, next_rel))

            while in_flight:
                rel, data = in_flight.popleft().result()
                if data is None:
                    missing += 1
                else:
                    zf.writestr(rel, data)
                    added += 1
                pbar.update(1)

    return added, missing


def build_image_zip_sharded(
    images_root: Path,
    rel_paths: Set[str],
    zip_base_path: Path,
    num_shards: int,
) -> Tuple[List[Path], int, int]:
    """Build ``num_shards`` zip files, each holding a slice of the images.

    Returns ``(shard_paths, total_added, total_missing)``. Sharding gives:
      - Bounded blast radius for a single transient upload/commit failure.
      - Parallel-friendly uploads if you wrap the resulting files in a
        worker pool downstream.
      - Easier resumability — only the failed shard needs a re-push.

    Shard filenames are derived from ``zip_base_path`` as
    ``<stem>_part<NN>of<MM>.zip``. The sample-to-image relative paths
    inside each shard mirror the dataset JSON, so consumers can extract
    all shards into one directory and the paths line up.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")

    rels_sorted = sorted(rel_paths)
    total = len(rels_sorted)
    # Round up so the last shard isn't oversized when len doesn't divide.
    per_shard = max(1, (total + num_shards - 1) // num_shards)

    stem = zip_base_path.stem
    parent = zip_base_path.parent
    width = max(2, len(str(num_shards)))  # 2-digit minimum: _part01of10.zip

    shard_paths: List[Path] = []
    total_added = total_missing = 0

    for shard_idx in range(num_shards):
        start = shard_idx * per_shard
        end = min(total, start + per_shard)
        if start >= end:
            break
        shard_rels = set(rels_sorted[start:end])
        shard_path = parent / f"{stem}_part{shard_idx + 1:0{width}d}of{num_shards:0{width}d}.zip"
        print(f"  Building shard {shard_idx + 1}/{num_shards}: {shard_path.name} "
              f"({len(shard_rels):,} images)")
        added, missing = build_image_zip(images_root, shard_rels, shard_path)
        total_added += added
        total_missing += missing
        shard_paths.append(shard_path)

    return shard_paths, total_added, total_missing


def reshard_existing_zip(
    src_zip_path: Path,
    shard_paths: List[Path],
) -> Tuple[int, int]:
    """Split an already-built zip into ``len(shard_paths)`` smaller zips.

    Much faster than re-zipping from source on a hit — we sequentially read
    each entry from the existing zip and write it straight to its target
    shard. No random-access I/O on the original ~1.37M image files.

    Entries are distributed by count (not by byte size) so a typical UI
    dataset ends up roughly balanced; this keeps the splitting logic simple
    and matches the layout ``build_image_zip_sharded`` would have produced
    from source.

    Returns ``(total_added, total_missing)``. ``missing`` is always 0 here
    — there's no way for an entry already inside the zip to be "missing" —
    but the signature mirrors the other zip builders so callers can plug
    the result into the same logging path.
    """
    num_shards = len(shard_paths)
    if num_shards < 1:
        raise ValueError("shard_paths must be non-empty")

    with zipfile.ZipFile(src_zip_path, 'r') as src:
        names = sorted(src.namelist())
        total = len(names)
        per_shard = max(1, (total + num_shards - 1) // num_shards)

        added = 0
        # Open all shards up-front so a half-built run leaves consistent
        # files behind (rather than e.g. shard 1 fully written and shard 2
        # not created at all).
        shard_zips = [
            zipfile.ZipFile(p, 'w', zipfile.ZIP_STORED, allowZip64=True)
            for p in shard_paths
        ]
        try:
            with tqdm(total=total, desc="Re-sharding from existing zip", smoothing=0.05) as pbar:
                for i, name in enumerate(names):
                    shard_idx = min(i // per_shard, num_shards - 1)
                    # ``open(name, 'r')`` streams the entry — important
                    # because individual screenshots can be several MB.
                    with src.open(name) as srcf, shard_zips[shard_idx].open(
                        name, 'w', force_zip64=True,
                    ) as dstf:
                        # 1 MiB chunks: large enough to amortise per-call
                        # overhead, small enough that memory stays bounded.
                        while True:
                            chunk = srcf.read(1024 * 1024)
                            if not chunk:
                                break
                            dstf.write(chunk)
                    added += 1
                    pbar.update(1)
        finally:
            for zf in shard_zips:
                zf.close()

    return added, 0


def derive_sibling_paths(samples_path: Path) -> Tuple[Path, Path]:
    """Return ``(<base>_sample.json, <base>_info.json)`` sibling paths.

    Mirrors the filename convention used by ``run_info.save_run_info``.
    """
    stem = samples_path.stem
    parent = samples_path.parent
    return parent / f"{stem}_sample.json", parent / f"{stem}_info.json"


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a transient upload/commit failure.

    Order matters: ``HfHubHTTPError`` is a subclass of ``OSError`` (via
    ``requests.HTTPError`` → ``RequestException`` → ``IOError``), so we have
    to check it *first* with status-code-aware logic. Otherwise a permanent
    HTTP 404 would also match the generic ``OSError`` catch and trigger
    pointless retries.

    Retryable cases:
      - HF Hub 429 (rate limit), 408/425 (timeouts), 5xx (server fault).
      - Xet CAS 400 on ``cas-server.xethub.hf.co/shards`` — observed in the
        wild as a transient commit-finalisation failure after the bytes have
        already landed in the content-addressed store. Retrying just
        re-issues the shard registration without re-uploading data.
      - Plain ``ConnectionError`` / ``TimeoutError`` (network blips that
        aren't wrapped in HTTPError).
    """
    if isinstance(exc, HfHubHTTPError):
        resp = getattr(exc, 'response', None)
        status = getattr(resp, 'status_code', 0) or 0
        if status in {408, 425, 429, 500, 502, 503, 504}:
            return True
        if status == 400:
            url = str(getattr(resp, 'url', '') or '')
            if 'xethub.hf.co' in url or 'cas-server' in url:
                return True
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


def _with_retry(
    fn: Callable[[], None],
    *,
    label: str,
    max_retries: int = 5,
    initial_backoff: float = 10.0,
) -> None:
    """Run ``fn`` with exponential backoff for transient errors."""
    for attempt in range(1, max_retries + 1):
        try:
            fn()
            return
        except BaseException as exc:  # noqa: BLE001
            if not _is_retryable_error(exc) or attempt == max_retries:
                raise
            backoff = initial_backoff * (2 ** (attempt - 1))
            print(
                f"  ! {label} failed (attempt {attempt}/{max_retries}): "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                f"  Sleeping {backoff:.0f}s before retry — "
                "the bytes are already in HF's content-addressed store, "
                "so retries skip re-upload and just re-issue the commit."
            )
            time.sleep(backoff)


def upload_to_hub(
    *,
    repo_id: str,
    files: List[Path],
    token: Optional[str],
    private: bool,
    commit_message: str,
    max_retries: int = 5,
) -> None:
    """Create the repo (if missing) and upload ``files`` into its root.

    A ``.gitattributes`` is pushed first so ``*.zip`` / ``*.json`` are tracked
    via LFS — required for the multi-GB payloads typical of these datasets.

    Each upload is wrapped in :func:`_with_retry` so transient backend
    failures (xet 400, 429, 5xx, ConnectionError) self-heal instead of
    killing the run after a long upload.
    """
    api = HfApi(token=token)
    create_repo(
        repo_id=repo_id,
        repo_type='dataset',
        private=private,
        exist_ok=True,
        token=token,
    )

    _with_retry(
        lambda: api.upload_file(
            path_or_fileobj=GITATTRIBUTES_CONTENT.encode(),
            path_in_repo='.gitattributes',
            repo_id=repo_id,
            repo_type='dataset',
            commit_message='Configure LFS for *.zip and *.json',
            token=token,
        ),
        label="upload .gitattributes",
        max_retries=max_retries,
    )

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"Uploading {f.name} ({size_mb:,.1f} MB)...")
        _with_retry(
            lambda f=f: api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=repo_id,
                repo_type='dataset',
                commit_message=commit_message,
                token=token,
            ),
            label=f"upload {f.name}",
            max_retries=max_retries,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--samples', required=True, type=Path,
        help='Full samples .json produced by a make_*_data script.',
    )
    ap.add_argument(
        '--images-root', required=True, type=Path,
        help='Base directory the `image` paths in samples are relative to. '
             "For the standard UIPro layout this is the project's `data/` dir.",
    )
    ap.add_argument(
        '--repo-id', required=True,
        help='Destination HF dataset repo, e.g. `your-username/uipro-webui`.',
    )
    ap.add_argument(
        '--token', default=os.environ.get('HF_TOKEN'),
        help='HF auth token. Defaults to $HF_TOKEN or the cached `hf auth login`.',
    )
    ap.add_argument(
        '--private', action='store_true',
        help='Create the repo as private if it does not exist yet.',
    )
    ap.add_argument(
        '--zip-name', default=None,
        help='Override the zip filename (default: `<samples_stem>_images.zip`).',
    )
    ap.add_argument(
        '--skip-zip', action='store_true',
        help='Reuse an existing zip at the target zip path instead of rebuilding.',
    )
    ap.add_argument(
        '--zip-workers', type=int, default=32,
        help='Parallel read threads when building the image zip. Bump for '
             'networked storage (32-64); drop to 4-8 for slow local disks.',
    )
    ap.add_argument(
        '--zip-shards', type=int, default=1,
        help='Split the image zip into this many shard files '
             '(`<stem>_partXXofYY.zip`). Use ~50-100 GB per shard to bound '
             'the blast radius of any single upload/commit failure — a '
             'transient 400 from the xet CAS only forfeits one shard, not '
             'the whole 485 GB upload. Default 1 keeps a single zip.',
    )
    ap.add_argument(
        '--max-retries', type=int, default=5,
        help='How many times each upload retries on transient errors '
             '(xet 400, 429, 5xx, ConnectionError). Backoff is exponential '
             'starting at 10s. Bytes already in the content-addressed store '
             'are NOT re-transmitted on retry — only the commit is retried.',
    )
    ap.add_argument(
        '--dry-run', action='store_true',
        help='Build the zip locally but skip the upload step.',
    )
    ap.add_argument(
        '--commit-message', default='Upload UIPro dataset artifacts',
        help='Commit message used for every upload.',
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    samples_path: Path = args.samples
    if not samples_path.is_file():
        sys.exit(f"samples file not found: {samples_path}")
    if not args.images_root.is_dir():
        sys.exit(f"--images-root is not a directory: {args.images_root}")

    sample_path, info_path = derive_sibling_paths(samples_path)

    print(f"Loading {samples_path}...")
    with open(samples_path) as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        sys.exit(f"expected a JSON list of samples, got {type(samples).__name__}")
    print(f"  -> {len(samples):,} samples")

    rel_paths = collect_image_paths(samples)
    print(f"  -> {len(rel_paths):,} unique images referenced")
    if not rel_paths:
        sys.exit(
            "No 'image' fields found in samples — nothing to zip. "
            "Check the input file or its schema."
        )

    zip_name = args.zip_name or f"{samples_path.stem}_images.zip"
    zip_path = samples_path.parent / zip_name
    num_shards = max(1, args.zip_shards)

    # Resolve which zip artifacts will be uploaded. Sharded paths follow
    # ``<stem>_partXXofYY.zip``; the single-zip case keeps the legacy name.
    if num_shards == 1:
        shard_paths = [zip_path]
    else:
        width = max(2, len(str(num_shards)))
        stem = zip_path.stem
        shard_paths = [
            zip_path.parent / f"{stem}_part{i + 1:0{width}d}of{num_shards:0{width}d}.zip"
            for i in range(num_shards)
        ]

    # Decision tree for the image-zip step. Three paths in priority order:
    #   1. All target shards already exist on disk → reuse them as-is.
    #   2. ``--skip-zip`` + only the legacy single zip exists + we now want
    #      shards → split the existing zip into shards (sequential I/O,
    #      no re-walking the source images — important when the source tree
    #      lives on slow/networked storage).
    #   3. Otherwise → build from source.
    legacy_single_exists = num_shards > 1 and zip_path.is_file()
    shards_all_present = all(p.is_file() for p in shard_paths)

    if args.skip_zip and shards_all_present:
        for p in shard_paths:
            print(f"Reusing existing zip {p} ({p.stat().st_size:,} bytes)")
    elif args.skip_zip and legacy_single_exists:
        print(
            f"Splitting existing {zip_path.name} "
            f"({zip_path.stat().st_size:,} bytes) into {num_shards} shards "
            "(sequential read + write, no re-walking source images)..."
        )
        added, missing = reshard_existing_zip(zip_path, shard_paths)
        print(f"  -> {added:,} entries written across {num_shards} shards")
        if missing:
            print(f"  ! {missing:,} entries skipped (corrupt zip entry?)")
    else:
        if num_shards == 1:
            print(f"Building image zip at {zip_path}...")
            added, missing = build_image_zip(
                args.images_root, rel_paths, zip_path, workers=args.zip_workers,
            )
        else:
            print(f"Building {num_shards} sharded image zips at {zip_path.parent}/{zip_path.stem}_part*.zip ...")
            shard_paths, added, missing = build_image_zip_sharded(
                args.images_root, rel_paths, zip_path, num_shards=num_shards,
            )
        print(f"  -> {added:,} added, {missing:,} missing")
        if missing:
            # Surface the count and keep going — better to upload a partial
            # dataset than to abort after a long zip build.
            print(
                f"  ! {missing:,} images were missing under {args.images_root}; "
                "their samples will reference absent files."
            )

    # Order matters for resumability: small files first (sibling JSONs), then
    # the bulky shards. If something dies on shard N, the earlier metadata
    # commits are already up.
    upload_manifest: List[Path] = [samples_path]
    for sibling in (sample_path, info_path):
        if sibling.is_file():
            upload_manifest.append(sibling)
        else:
            print(f"  (skipping {sibling.name} — not found next to samples file)")
    upload_manifest.extend(shard_paths)

    if args.dry_run:
        print("\nDry run — would upload these files:")
        for f in upload_manifest:
            print(f"  {f.name:<60s} {f.stat().st_size:>14,} bytes")
        print(f"  -> repo: {args.repo_id} (private={args.private})")
        return

    upload_to_hub(
        repo_id=args.repo_id,
        files=upload_manifest,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
        max_retries=args.max_retries,
    )
    print(f"\nDone. Dataset URL: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
