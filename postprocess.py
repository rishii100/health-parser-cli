"""
Post-processing module — validates and fixes LLM output.
- Taxonomy validation (category/subcategory pairs)
- Evidence span verification against actual note text
- Date format normalization
- Status validation
"""

import json
import re
import difflib
from pathlib import Path


# ---------------------------------------------------------------------------
# Taxonomy Validation
# ---------------------------------------------------------------------------

def load_valid_taxonomy_pairs(taxonomy_path):
    """Load all valid (category, subcategory) pairs from taxonomy.json."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    valid_pairs = set()
    for cat_key, cat_val in taxonomy["condition_categories"].items():
        for sub_key in cat_val["subcategories"]:
            valid_pairs.add((cat_key, sub_key))
    return valid_pairs


def validate_taxonomy(conditions, valid_pairs):
    """Check and fix category/subcategory pairs against taxonomy."""
    issues = []
    all_pair_strings = [f"{c}.{s}" for c, s in valid_pairs]

    for cond in conditions:
        pair = (cond.get("category", ""), cond.get("subcategory", ""))
        if pair in valid_pairs:
            continue

        issue = f"Invalid taxonomy pair '{pair[0]}.{pair[1]}' for '{cond.get('condition_name', '?')}'"

        # Try fuzzy match
        candidates = difflib.get_close_matches(
            f"{pair[0]}.{pair[1]}", all_pair_strings, n=1, cutoff=0.5
        )
        if candidates:
            cat, sub = candidates[0].split(".", 1)
            cond["category"] = cat
            cond["subcategory"] = sub
            issue += f" → auto-fixed to {candidates[0]}"
        else:
            # Try matching just the category
            valid_cats = set(c for c, _ in valid_pairs)
            cat_matches = difflib.get_close_matches(pair[0], valid_cats, n=1, cutoff=0.5)
            if cat_matches:
                cond["category"] = cat_matches[0]
                # Pick first subcategory as fallback
                subs = [s for c, s in valid_pairs if c == cat_matches[0]]
                if subs:
                    cond["subcategory"] = subs[0]
                    issue += f" → category fixed to {cat_matches[0]}, subcategory defaulted to {subs[0]}"

        issues.append(issue)

    return issues


# ---------------------------------------------------------------------------
# Status Validation
# ---------------------------------------------------------------------------

VALID_STATUSES = {"active", "resolved", "suspected"}


def validate_status(conditions):
    """Ensure all status values are valid."""
    issues = []
    for cond in conditions:
        status = cond.get("status", "")
        if status in VALID_STATUSES:
            continue

        # Try to fix common variants
        status_lower = status.lower().strip()
        if status_lower in VALID_STATUSES:
            cond["status"] = status_lower
            continue

        issues.append(
            f"Invalid status '{status}' for '{cond.get('condition_name', '?')}' → defaulting to 'active'"
        )
        cond["status"] = "active"

    return issues


# ---------------------------------------------------------------------------
# Date Format Validation
# ---------------------------------------------------------------------------

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Valid date patterns
RE_FULL_DATE = re.compile(r"^\d{1,2}\s+[A-Z][a-z]+\s+\d{4}$")
RE_MONTH_YEAR = re.compile(r"^[A-Z][a-z]+\s+\d{4}$")
RE_YEAR_ONLY = re.compile(r"^\d{4}$")

# Common non-standard patterns to fix
RE_MM_YYYY = re.compile(r"^(\d{1,2})/(\d{4})$")
RE_YYYY_MM = re.compile(r"^(\d{4})-(\d{1,2})$")
RE_MM_DD_YYYY = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
RE_DD_MM_YYYY = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")
RE_YYYY_MM_DD = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
RE_MONTH_DD_YYYY = re.compile(r"^([A-Z][a-z]+)\s+(\d{1,2}),?\s+(\d{4})$")


def month_name(idx):
    """Convert 1-based month index to month name."""
    if 1 <= idx <= 12:
        return MONTHS[idx - 1]
    return None


def normalize_onset_date(onset):
    """Normalize an onset date string to one of the valid formats, or return as-is."""
    if onset is None:
        return None

    onset = str(onset).strip()
    if not onset or onset.lower() in ("null", "none", "unknown", "n/a"):
        return None

    # Already valid?
    if RE_FULL_DATE.match(onset) or RE_MONTH_YEAR.match(onset) or RE_YEAR_ONLY.match(onset):
        return onset

    # MM/YYYY → Month YYYY
    m = RE_MM_YYYY.match(onset)
    if m:
        mn = month_name(int(m.group(1)))
        if mn:
            return f"{mn} {m.group(2)}"

    # YYYY-MM → Month YYYY
    m = RE_YYYY_MM.match(onset)
    if m:
        mn = month_name(int(m.group(2)))
        if mn:
            return f"{mn} {m.group(1)}"

    # MM/DD/YYYY → DD Month YYYY
    m = RE_MM_DD_YYYY.match(onset)
    if m:
        mn = month_name(int(m.group(1)))
        if mn:
            return f"{m.group(2)} {mn} {m.group(3)}"

    # DD.MM.YYYY → DD Month YYYY
    m = RE_DD_MM_YYYY.match(onset)
    if m:
        mn = month_name(int(m.group(2)))
        if mn:
            return f"{m.group(1)} {mn} {m.group(3)}"

    # YYYY-MM-DD → DD Month YYYY
    m = RE_YYYY_MM_DD.match(onset)
    if m:
        mn = month_name(int(m.group(2)))
        if mn:
            return f"{int(m.group(3))} {mn} {m.group(1)}"

    # Month DD, YYYY → DD Month YYYY
    m = RE_MONTH_DD_YYYY.match(onset)
    if m:
        return f"{m.group(2)} {m.group(1)} {m.group(3)}"

    return onset  # Return as-is if no pattern matched


def validate_dates(conditions):
    """Validate and normalize onset date formats."""
    issues = []
    for cond in conditions:
        raw = cond.get("onset")
        normalized = normalize_onset_date(raw)
        if normalized != raw:
            if raw is not None and normalized is not None:
                issues.append(
                    f"Date normalized '{raw}' → '{normalized}' for '{cond.get('condition_name', '?')}'"
                )
            cond["onset"] = normalized

        # Final validation
        final = cond.get("onset")
        if final is not None and not (
            RE_FULL_DATE.match(final) or RE_MONTH_YEAR.match(final) or RE_YEAR_ONLY.match(final)
        ):
            issues.append(
                f"Non-standard date format remains: '{final}' for '{cond.get('condition_name', '?')}'"
            )

    return issues


# ---------------------------------------------------------------------------
# Evidence Span Verification
# ---------------------------------------------------------------------------

def load_note_lines(note_path):
    """Load a note file and return list of lines (0-indexed internally)."""
    with open(note_path, "r", encoding="utf-8") as f:
        return f.readlines()


def verify_evidence_spans(conditions, data_dir, patient_id):
    """Verify and fix evidence spans against actual note files."""
    issues = []
    note_cache = {}
    patient_dir = Path(data_dir) / patient_id

    for cond in conditions:
        cleaned_evidence = []
        for ev in cond.get("evidence", []):
            note_id = ev.get("note_id", "")
            line_no = ev.get("line_no")
            span = ev.get("span", "")

            if not note_id or not span:
                continue

            # Load note
            if note_id not in note_cache:
                note_path = patient_dir / f"{note_id}.md"
                if note_path.exists():
                    note_cache[note_id] = load_note_lines(note_path)
                else:
                    issues.append(f"Note file not found: {note_path}")
                    cleaned_evidence.append(ev)
                    continue

            lines = note_cache[note_id]

            # Ensure line_no is an integer
            try:
                line_no = int(line_no)
            except (TypeError, ValueError):
                line_no = 1

            # Strategy 1: Check if span is a substring of the cited line
            if 1 <= line_no <= len(lines):
                actual_line = lines[line_no - 1].strip()
                if span.strip() in actual_line:
                    ev["line_no"] = line_no
                    cleaned_evidence.append(ev)
                    continue

            # Strategy 2: Check nearby lines (±3)
            found = False
            for offset in range(-3, 4):
                check_idx = line_no + offset - 1
                if 0 <= check_idx < len(lines):
                    if span.strip() in lines[check_idx].strip():
                        ev["line_no"] = check_idx + 1
                        cleaned_evidence.append(ev)
                        found = True
                        break
            if found:
                continue

            # Strategy 3: Search entire note for exact span
            for i, line in enumerate(lines):
                if span.strip() in line.strip():
                    ev["line_no"] = i + 1
                    cleaned_evidence.append(ev)
                    found = True
                    break
            if found:
                continue

            # Strategy 4: Case-insensitive search
            span_lower = span.strip().lower()
            for i, line in enumerate(lines):
                if span_lower in line.strip().lower():
                    ev["line_no"] = i + 1
                    # Fix span to match actual case
                    start = line.strip().lower().index(span_lower)
                    ev["span"] = line.strip()[start : start + len(span.strip())]
                    cleaned_evidence.append(ev)
                    found = True
                    break
            if found:
                continue

            # Strategy 5: Fuzzy match — find most similar line
            best_ratio = 0
            best_line_idx = -1
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # Check if a substantial part of the span appears in this line
                ratio = difflib.SequenceMatcher(
                    None, span.strip().lower(), line_stripped.lower()
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_line_idx = i

            if best_ratio > 0.4 and best_line_idx >= 0:
                ev["line_no"] = best_line_idx + 1
                cleaned_evidence.append(ev)
                issues.append(
                    f"Fuzzy-matched span (ratio={best_ratio:.2f}) for '{cond.get('condition_name', '?')}' "
                    f"in {note_id} line {best_line_idx + 1}"
                )
            else:
                # Keep the evidence but flag it
                cleaned_evidence.append(ev)
                issues.append(
                    f"Could not verify span for '{cond.get('condition_name', '?')}' "
                    f"in {note_id} line {line_no}: '{span[:60]}...'"
                )

        cond["evidence"] = cleaned_evidence

    return issues


# ---------------------------------------------------------------------------
# Remove internal fields
# ---------------------------------------------------------------------------

def clean_output(patient_result):
    """Remove any internal-only fields from the final output."""
    for cond in patient_result.get("conditions", []):
        # Remove fields not in the output schema
        for key in list(cond.keys()):
            if key not in ("condition_name", "category", "subcategory", "status", "onset", "evidence"):
                del cond[key]
        # Clean evidence entries
        for ev in cond.get("evidence", []):
            for key in list(ev.keys()):
                if key not in ("note_id", "line_no", "span"):
                    del ev[key]
    return patient_result


# ---------------------------------------------------------------------------
# Main Post-processing Pipeline
# ---------------------------------------------------------------------------

def postprocess(patient_result, data_dir, taxonomy_path):
    """Run all post-processing steps. Returns (cleaned_result, issues_list)."""
    valid_pairs = load_valid_taxonomy_pairs(taxonomy_path)
    conditions = patient_result.get("conditions", [])

    all_issues = []

    # 1. Validate taxonomy
    all_issues.extend(validate_taxonomy(conditions, valid_pairs))

    # 2. Validate status
    all_issues.extend(validate_status(conditions))

    # 3. Normalize and validate dates
    all_issues.extend(validate_dates(conditions))

    # 4. Verify evidence spans
    patient_id = patient_result.get("patient_id", "unknown")
    all_issues.extend(verify_evidence_spans(conditions, data_dir, patient_id))

    # 5. Clean internal fields
    patient_result = clean_output(patient_result)

    return patient_result, all_issues
