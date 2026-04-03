"""
Core extraction module — Stage 1 (per-note) and Stage 2 (cross-note synthesis).
Uses OpenAI-compatible API (Groq, OpenAI, etc.) via environment variables.

Optimized for low-TPM rate limits (Groq free tier: 12K TPM).
"""

import os
import json
import time
import re
from pathlib import Path
from openai import OpenAI


def get_client():
    """Initialize OpenAI-compatible client from environment variables."""
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
    )


def get_model():
    """Get model name from environment variable."""
    return os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")


def load_taxonomy(taxonomy_path):
    """Load taxonomy JSON file."""
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_taxonomy_compact(taxonomy):
    """Ultra-compact taxonomy string for Stage 1 prompts (saves ~60% tokens)."""
    lines = []
    cats = taxonomy["condition_categories"]
    for cat_key, cat_val in cats.items():
        subs = ", ".join(cat_val["subcategories"].keys())
        lines.append(f"{cat_key}: {subs}")
    return "\n".join(lines)


def format_taxonomy_for_prompt(taxonomy):
    """Full taxonomy string for Stage 2 prompts (subcategory descriptions included)."""
    lines = []
    cats = taxonomy["condition_categories"]
    for cat_key, cat_val in cats.items():
        lines.append(f"\n## {cat_key}: {cat_val['description']}")
        for sub_key, sub_desc in cat_val["subcategories"].items():
            lines.append(f"  - {cat_key}.{sub_key}: {sub_desc}")

    lines.append("\n## Status Values")
    for status_key, status_val in taxonomy["status_values"].items():
        signals = ", ".join(status_val["signals"][:4])
        lines.append(f"  - {status_key}: {status_val['description']} (signals: {signals})")

    lines.append("\n## Disambiguation Rules")
    for rule in taxonomy.get("disambiguation_rules", []):
        lines.append(f"  - {rule['rule']}: {rule['explanation']}")

    return "\n".join(lines)


def read_note_with_line_numbers(note_path):
    """Read a note file and return (numbered_text, raw_lines_list)."""
    with open(note_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    numbered = []
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if stripped:  # Skip blank lines to save tokens
            numbered.append(f"{i}: {stripped}")
    return "\n".join(numbered), lines


def call_llm(client, model, system_prompt, user_prompt, max_retries=7, temperature=0.1, max_tokens=4096):
    """Call LLM with retry logic for rate limits, TPM limits, and transient errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            # Handle all rate limit variants: 429, 413 (request too large for TPM), "rate_limit"
            if any(s in err_str for s in ["rate_limit", "429", "413", "too many", "tokens per minute", "request too large"]):
                wait = min(2 ** (attempt + 1) * 5, 90)
                print(f"\n    ⏳ Rate/TPM limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            elif "503" in err_str or "server" in err_str or "overloaded" in err_str:
                wait = min(2 ** attempt * 5, 60)
                print(f"\n    ⏳ Server error (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Max retries ({max_retries}) exceeded for LLM call")


def parse_json_response(text):
    """Parse JSON from LLM response, handling markdown code blocks and extra text."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` code blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding the outermost JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}...")


# ---------------------------------------------------------------------------
# Stage 1: Per-Note Condition Extraction (token-optimized)
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = """You are a clinical NLP system. Extract ALL medical conditions from the clinical note. Output JSON only.

For each condition output: condition_name, category, subcategory (from taxonomy), status_in_note (active/resolved/suspected), mentioned_dates (array of dates associated with this condition), evidence (array of {line_no, span} where span is EXACT text from that line).

Rules:
- Extract: diagnoses, comorbidities, medical history, imaging findings, abnormal labs (low Hgb=anemia, low platelets=thrombocytopenia, etc.)
- "Status post"/"History of" → resolved
- "Suspected"/"possible"/"rule out" → suspected
- One entry per distinct condition per site
- Only conditions in the taxonomy
- Spans must be EXACT verbatim text from the note

Also extract note_date (encounter date) if visible.

Output: {"note_date":"...or null","conditions":[...]}"""


def extract_from_note(client, model, note_path, note_id, taxonomy_compact, delay=5.0):
    """Stage 1: Extract conditions from one clinical note (token-optimized)."""
    numbered_text, raw_lines = read_note_with_line_numbers(note_path)

    user_prompt = f"""TAXONOMY (category: subcategories):
{taxonomy_compact}

NOTE {note_id}:
{numbered_text}

Extract all conditions. JSON only."""

    if delay > 0:
        time.sleep(delay)

    response_text = call_llm(client, model, STAGE1_SYSTEM, user_prompt, max_tokens=4096)
    result = parse_json_response(response_text)

    # Normalize structure
    if isinstance(result, list):
        result = {"note_date": None, "conditions": result}
    if "conditions" not in result:
        result["conditions"] = []

    # Tag each evidence entry with note_id
    for cond in result["conditions"]:
        for ev in cond.get("evidence", []):
            ev["note_id"] = note_id
        if "mentioned_dates" not in cond:
            cond["mentioned_dates"] = []

    return result


# ---------------------------------------------------------------------------
# Stage 2: Cross-Note Synthesis
# ---------------------------------------------------------------------------

STAGE2_SYSTEM = """You are a clinical NLP synthesis system. Merge per-note condition extractions for ONE patient into a final deduplicated summary.

RULES:
1. DEDUPLICATE: Same condition across notes → ONE entry. Match generously ("Diabetes mellitus type II" = "Non-insulin-dependent diabetes mellitus"). Keep different sites separate.
2. CONDITION NAME: Most specific descriptive name from the notes.
3. STATUS: From LATEST note where condition appears (text_0=earliest, higher=later).
4. ONSET: Earliest documented date. Priority: stated date > note encounter date > relative date > null. Formats: "16 March 2026", "March 2014", "2014", or null.
5. EVIDENCE: ALL {note_id, line_no, span} from ALL notes. Keep every entry.
6. DISAMBIGUATION: Heart failure → categorize by cause. Diabetic complications → metabolic_endocrine.diabetes.

Output: {"patient_id":"...","conditions":[{"condition_name":"...","category":"...","subcategory":"...","status":"...","onset":"...","evidence":[{"note_id":"...","line_no":N,"span":"..."}]}]}"""


def synthesize_patient(client, model, patient_id, per_note_results, note_order, taxonomy, delay=5.0):
    """Stage 2: Merge per-note extractions into final patient output."""

    # Build note metadata
    note_meta_lines = []
    for idx, note_id in enumerate(note_order):
        note_data = per_note_results.get(note_id, {})
        note_date = note_data.get("note_date", "unknown")
        position = "earliest" if idx == 0 else ("latest" if idx == len(note_order) - 1 else f"#{idx+1}")
        note_meta_lines.append(f"- {note_id}: date={note_date}, pos={position}")

    # Collect all per-note conditions (compact format to save tokens)
    all_conditions = []
    for note_id in note_order:
        note_data = per_note_results.get(note_id, {})
        for cond in note_data.get("conditions", []):
            cond_entry = {
                "n": cond.get("condition_name"),
                "cat": cond.get("category"),
                "sub": cond.get("subcategory"),
                "st": cond.get("status_in_note"),
                "dates": cond.get("mentioned_dates", []),
                "src": note_id,
                "src_date": per_note_results.get(note_id, {}).get("note_date"),
                "ev": cond.get("evidence", []),
            }
            all_conditions.append(cond_entry)

    # Build compact taxonomy reference for synthesis
    taxonomy_ref = format_taxonomy_compact(taxonomy)

    user_prompt = f"""PATIENT: {patient_id}

TAXONOMY (category: subcategories):
{taxonomy_ref}

NOTES (chronological):
{chr(10).join(note_meta_lines)}

CONDITIONS ({len(all_conditions)} entries, n=name, cat=category, sub=subcategory, st=status, ev=evidence):
{json.dumps(all_conditions, ensure_ascii=False, separators=(',', ':'))}

Merge into final deduplicated summary. Use full field names in output (condition_name, category, subcategory, status, onset, evidence). JSON only."""

    if delay > 0:
        time.sleep(delay)

    response_text = call_llm(client, model, STAGE2_SYSTEM, user_prompt, max_tokens=8192)
    result = parse_json_response(response_text)

    # Ensure patient_id is set
    if "patient_id" not in result:
        result["patient_id"] = patient_id
    if "conditions" not in result:
        result["conditions"] = []

    return result
