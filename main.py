#!/usr/bin/env python3
"""
Clinical Condition Extraction Pipeline — Main Entrypoint

Extracts structured condition summaries from longitudinal patient clinical notes
using a two-stage LLM pipeline.

Required environment variables:
  OPENAI_API_KEY   — API key (e.g., Groq API key)
  OPENAI_BASE_URL  — API base URL (e.g., https://api.groq.com/openai/v1)
  OPENAI_MODEL     — Model identifier (e.g., llama-3.3-70b-versatile)

Usage:
  python main.py --data-dir ./train --patient-list ./patients.json --output-dir ./output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from extractor import (
    get_client,
    get_model,
    load_taxonomy,
    format_taxonomy_compact,
    format_taxonomy_for_prompt,
    extract_from_note,
    synthesize_patient,
)
from postprocess import postprocess


def get_patient_notes(data_dir, patient_id):
    """Get sorted note files for a patient in chronological order."""
    patient_dir = Path(data_dir) / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    note_files = sorted(
        patient_dir.glob("text_*.md"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not note_files:
        raise FileNotFoundError(f"No note files found in {patient_dir}")

    return note_files


def find_taxonomy(data_dir, explicit_path=None):
    """Locate taxonomy.json, searching common locations."""
    if explicit_path and Path(explicit_path).exists():
        return str(explicit_path)

    candidates = [
        Path(data_dir) / "taxonomy.json",
        Path(data_dir).parent / "taxonomy.json",
        Path(data_dir).parent.parent / "taxonomy.json",
        Path("taxonomy.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return None


def process_patient(
    client, model, data_dir, patient_id, taxonomy, taxonomy_compact, taxonomy_path,
    verbose=False, delay=5.0,
):
    """Process a single patient through the full two-stage pipeline."""
    note_files = get_patient_notes(data_dir, patient_id)
    total_start = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Processing {patient_id} ({len(note_files)} notes)")
        print(f"{'='*60}")

    # ── Stage 1: Per-note extraction ──
    per_note_results = {}
    note_order = []

    for note_file in note_files:
        note_id = note_file.stem
        note_order.append(note_id)

        if verbose:
            print(f"  [Stage 1] Extracting from {note_id}...", end=" ", flush=True)

        try:
            result = extract_from_note(
                client, model, note_file, note_id, taxonomy_compact, delay=delay
            )
            per_note_results[note_id] = result

            if verbose:
                n = len(result.get("conditions", []))
                note_date = result.get("note_date", "?")
                print(f"→ {n} conditions (date: {note_date})")
        except Exception as e:
            print(f"\n    ✗ Error extracting {note_id}: {e}")
            per_note_results[note_id] = {"note_date": None, "conditions": []}

    # ── Stage 2: Cross-note synthesis ──
    if verbose:
        total_cands = sum(len(r.get("conditions", [])) for r in per_note_results.values())
        print(f"  [Stage 2] Synthesizing {total_cands} candidates across {len(note_order)} notes...", end=" ", flush=True)

    try:
        final_result = synthesize_patient(
            client, model, patient_id, per_note_results, note_order, taxonomy, delay=delay
        )
        if verbose:
            n = len(final_result.get("conditions", []))
            print(f"→ {n} final conditions")
    except Exception as e:
        print(f"\n    ✗ Error in synthesis: {e}")
        # Fallback: flatten per-note results without synthesis
        final_result = fallback_flatten(patient_id, per_note_results, note_order)

    # ── Post-processing ──
    if verbose:
        print(f"  [Post-process] Validating...", end=" ", flush=True)

    final_result, issues = postprocess(final_result, data_dir, taxonomy_path)

    elapsed = time.time() - total_start
    if verbose:
        print(f"done ({elapsed:.1f}s)")
        if issues:
            print(f"    ⚠ {len(issues)} post-processing fixes:")
            for issue in issues[:5]:
                print(f"      · {issue}")
            if len(issues) > 5:
                print(f"      ... and {len(issues) - 5} more")

    return final_result


def fallback_flatten(patient_id, per_note_results, note_order):
    """Emergency fallback if Stage 2 fails — flatten per-note results."""
    conditions = []
    seen = set()
    for note_id in note_order:
        for cond in per_note_results.get(note_id, {}).get("conditions", []):
            name = cond.get("condition_name", "").lower().strip()
            if name not in seen:
                seen.add(name)
                conditions.append({
                    "condition_name": cond.get("condition_name"),
                    "category": cond.get("category"),
                    "subcategory": cond.get("subcategory"),
                    "status": cond.get("status_in_note", "active"),
                    "onset": None,
                    "evidence": cond.get("evidence", []),
                })
    return {"patient_id": patient_id, "conditions": conditions}


def main():
    parser = argparse.ArgumentParser(
        description="Clinical Condition Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (required):
  OPENAI_API_KEY    API key for the LLM provider
  OPENAI_BASE_URL   API base URL (e.g., https://api.groq.com/openai/v1)
  OPENAI_MODEL      Model identifier (e.g., llama-3.3-70b-versatile)

Example:
  export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
  export OPENAI_API_KEY="gsk_..."
  export OPENAI_MODEL="llama-3.3-70b-versatile"

  python main.py \\
    --data-dir ./train \\
    --patient-list ./patients.json \\
    --output-dir ./output \\
    --verbose
""",
    )

    # Required args
    parser.add_argument("--data-dir", required=True, help="Path to data directory containing patient folders")
    parser.add_argument("--patient-list", required=True, help="Path to JSON file with list of patient IDs to process")
    parser.add_argument("--output-dir", required=True, help="Directory where output JSON files will be written")

    # Optional args
    parser.add_argument("--taxonomy", default=None, help="Path to taxonomy.json (auto-detected if not specified)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls in seconds (default: 2.0)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature (default: 0.1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress output")

    args = parser.parse_args()

    # ── Validate environment ──
    missing_vars = []
    for var in ["OPENAI_API_KEY"]:
        if not os.environ.get(var):
            missing_vars.append(var)
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Set them before running: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    # ── Locate taxonomy ──
    taxonomy_path = find_taxonomy(args.data_dir, args.taxonomy)
    if not taxonomy_path:
        print("Error: Could not find taxonomy.json. Specify with --taxonomy")
        sys.exit(1)

    # ── Load inputs ──
    taxonomy = load_taxonomy(taxonomy_path)
    taxonomy_compact = format_taxonomy_compact(taxonomy)

    with open(args.patient_list, "r", encoding="utf-8") as f:
        patient_ids = json.load(f)

    if not isinstance(patient_ids, list):
        print("Error: Patient list JSON must be a list of patient IDs")
        sys.exit(1)

    # ── Create output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialize LLM client ──
    client = get_client()
    model = get_model()

    # ── Print configuration ──
    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  Clinical Condition Extraction Pipeline                 ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  Model:      {model:<43}║")
    print(f"║  Base URL:   {os.environ.get('OPENAI_BASE_URL', 'default'):<43}║")
    print(f"║  Data dir:   {args.data_dir:<43}║")
    print(f"║  Patients:   {len(patient_ids):<43}║")
    print(f"║  Output:     {args.output_dir:<43}║")
    print(f"║  API delay:  {args.delay:<43}║")
    print(f"║  Taxonomy:   {Path(taxonomy_path).name:<43}║")
    print(f"╚══════════════════════════════════════════════════════════╝")

    # ── Process each patient ──
    pipeline_start = time.time()
    success = 0
    errors = 0

    for idx, patient_id in enumerate(patient_ids, 1):
        print(f"\n[{idx}/{len(patient_ids)}] {patient_id}")

        try:
            result = process_patient(
                client, model, args.data_dir, patient_id,
                taxonomy, taxonomy_compact, taxonomy_path,
                verbose=args.verbose, delay=args.delay,
            )

            # Write output
            output_path = output_dir / f"{patient_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            n_conds = len(result.get("conditions", []))
            print(f"  ✓ {n_conds} conditions → {output_path}")
            success += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            if args.verbose:
                traceback.print_exc()
            errors += 1

    # ── Summary ──
    elapsed = time.time() - pipeline_start
    print(f"\n{'─'*60}")
    print(f"Done in {elapsed:.1f}s • {success} succeeded • {errors} failed")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
