"""
Renames default-named NVPerf CSV files in the HTMLReports directory structure.

Expected structure:
    HTMLReports/
    ├── mesh/
    │   ├── 100/
    │   │   ├── 50/
    │   │   │   ├── nvperf_metrics.csv         -> nvperf_metrics_mesh_100_50.csv
    │   │   │   └── nvperf_metrics_summary.csv -> nvperf_metrics_summary_mesh_100_50.csv
    │   │   ├── 100/ ...
    │   ...
    └── vertex/
        ├── 100/
        │   ├── 50/ ...
        ...

Usage:
    python rename_nvperf_csvs.py                         # Uses ./HTMLReports
    python rename_nvperf_csvs.py /path/to/HTMLReports    # Custom path
    python rename_nvperf_csvs.py --dry-run               # Preview without renaming
"""

import os
import sys
import argparse

# CSV filenames to look for and rename
TARGET_CSVS = [
    "nvperf_metrics.csv",
    "nvperf_metrics_summary.csv",
]


def build_new_name(original_name: str, shader_type: str, outer: str, inner: str) -> str:
    """
    Build the new filename by inserting _<shader>_<outer>_<inner> before the .csv extension.

    Examples:
        nvperf_metrics.csv         -> nvperf_metrics_mesh_100_50.csv
        nvperf_metrics_summary.csv -> nvperf_metrics_summary_mesh_100_50.csv
    """
    stem, ext = os.path.splitext(original_name)  # ext == ".csv"
    return f"{stem}_{shader_type}_{outer}_{inner}{ext}"


def rename_csvs(root: str, dry_run: bool = False) -> None:
    """
    Walk HTMLReports and rename matching CSVs in every leaf (innermost) folder.

    The folder hierarchy under root is expected to be:
        <shader_type>/<outer>/<inner>/  (leaf)
    """
    renamed_count = 0
    skipped_count = 0
    already_named_count = 0

    # os.walk yields (dirpath, subdirs, files) for every directory
    for dirpath, subdirs, files in os.walk(root):
        # We only care about leaf directories (no subdirectories)
        if subdirs:
            continue

        # Determine the relative path from root to work out the folder levels
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep)

        # Expecting exactly 3 parts: shader_type / outer / inner
        if len(parts) != 3:
            print(f"  [SKIP] Unexpected depth ({len(parts)} levels): {rel}")
            skipped_count += 1
            continue

        shader_type, outer, inner = parts

        # Validate shader_type is one we expect
        if shader_type not in ("mesh", "vertex"):
            print(f"  [SKIP] Unknown shader type '{shader_type}' in: {rel}")
            skipped_count += 1
            continue

        # Process each target CSV found in this leaf folder
        for csv_name in TARGET_CSVS:
            src_path = os.path.join(dirpath, csv_name)

            if not os.path.isfile(src_path):
                continue  # This CSV doesn't exist in this folder, that's fine

            new_name = build_new_name(csv_name, shader_type, outer, inner)
            dst_path = os.path.join(dirpath, new_name)

            # Skip if already correctly named (shouldn't happen, but just in case)
            if os.path.exists(dst_path):
                print(f"  [EXISTS] Already renamed, skipping: {dst_path}")
                already_named_count += 1
                continue

            if dry_run:
                print(f"  [DRY-RUN] Would rename:\n"
                      f"    {src_path}\n"
                      f"    -> {dst_path}")
            else:
                os.rename(src_path, dst_path)
                print(f"  [OK] {csv_name} -> {new_name}  (in {rel})")

            renamed_count += 1

    # Summary
    print()
    if dry_run:
        print(f"Dry-run complete. {renamed_count} file(s) would be renamed.")
    else:
        print(f"Done. {renamed_count} file(s) renamed.")

    if skipped_count:
        print(f"         {skipped_count} folder(s) skipped (unexpected structure).")
    if already_named_count:
        print(f"         {already_named_count} file(s) already had the target name.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename NVPerf CSVs in an HTMLReports directory tree."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="HTMLReports",
        help="Path to the HTMLReports root folder (default: ./HTMLReports)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without making any changes",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)

    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a valid directory.")
        sys.exit(1)

    print(f"Root   : {root}")
    print(f"Mode   : {'DRY-RUN (no changes)' if args.dry_run else 'LIVE (files will be renamed)'}")
    print()

    rename_csvs(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()