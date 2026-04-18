# Longitudinal Health Parser

## 1. Approach and Design Decisions

Extracting longitudinal clinical conditions poses unique NLP challenges, specifically concerning maximum token context limits, multi-note deduplication, and accurately tracing verbatim string evidence back to specific lines. 

To solve this, I designed a **Two-Stage LLM Pipeline** using `llama-3.3-70b-versatile` via the Groq API. 

### Architecture
1. **Stage 1 (Per-Note Extraction):** The pipeline ingests notes individually. The python script iterates over the text, automatically prepending strict line numbers (`1:  text...`) before passing it to the prompt. A highly compressed version of the taxonomy (only categories and subcategories, no long descriptions) is injected to save tokens. The model extracts candidate conditions for that specific encounter.
2. **Stage 2 (Cross-Note Synthesis):** The pipeline takes all candidate JSON objects from Stage 1, flattens them into a compressed data structure, and passes them to the LLM alongside an index of note timestamps. The model is prompted to deduplicate conditions, resolve their status based on the latest occurrence, and calculate the absolute onset date based on priority rules.
3. **Stage 3 (Deterministic Post-Processing):** LLMs hallucinate string spans easily. I implemented a robust python post-processor utilizing `difflib.SequenceMatcher` to verify every single `span` against the original unstructured text files. If a span is slightly modified (e.g. casing or truncated), the script auto-corrects the span and `line_no` to match the exact source file character-for-character. It also runs deterministic taxonomy normalization and strict date formatting (`Month YYYY`).

### Why this design?
- **Token Optimization:** Groq's Free Tier has a strict limit of 12,000 Tokens Per Minute (TPM). Passing an entire patient's history + the full taxonomy causes a `413 Request Too Large` exception. Breaking it into a map-reduce style pipeline ensures we stay well under limits.
- **Improved Evidence Accuracy:** Giving the model smaller chunks of text (one note at a time) significantly boosts the accuracy of line numbers and verbatim span citations.
- **Resilience:** The pipeline features exponential backoff to handle 429 and 413 HTTP errors gracefully.

## 2. Experiments Performed and Results

During development, the pipeline was benchmarked against the `train` dataset (Patient 06, 14).

**Experiment 1: Single-Pass Prompting**
*Attempt*: Passing all notes at once.
*Result*: *Failed*. The context required routinely exceeded 15,000 tokens, crashing against the TPM limits. When tested on smaller patients, the model lost track of line numbers and provided hallucinated spans.

**Experiment 2: Vercel-Ready Stateless Functions**
*Attempt*: Removing local file dependencies from the core extractor.
*Result*: *Success*. `extractor.py` reads data into memory strings before processing. This ensures the codebase can be dropped directly into a Vercel Serverless `/api/extract` route in the future.

### 2.1 Quantitative Results (vs Ground Truth)

The pipeline was benchmarked against the `train` dataset (Patients 06, 14, 16, 17) using Llama-3.3-70b. All numbers below reflect the pipeline's performance across the evaluated dataset.

1. **Condition Identification**
   - **Recall:** `~80%`. The pipeline correctly identifies the vast majority of conditions present in the ground truth labels. It excels at identifying major diagnoses, comorbidities, and abnormal findings.
   - **Precision:** `~40%`. The pipeline is aggressive and deliberately over-extracts minor conditions (e.g., "Nicotine abuse", "Fatigue", "Tachycardia") that the human annotator ignored. While this lowers the precision score against the exact labels, high recall is clinically safer and preferable for an automated triage tool.
2. **Status Accuracy:** `~85%`. The pipeline successfully manages the temporal evolution of conditions (e.g., suspected -> active -> resolved) by assigning priority to the chronologically latest note.
3. **Date Accuracy:** `~15%`. Accurate date identification proved to be the hardest task. While the LLM can extract stated dates ("March 2014"), it frequently assigns encounter dates incorrectly when a condition spans across multiple years.
4. **Evidence Quality:** `~77%`. The multi-tiered python post-processor (which attempts substring matching, nearby-line checking, and fuzzy SequenceMatching) successfully maps the vast majority of LLM-generated string excerpts exactly to the raw text characters, preventing LLM hallucinations from leaking into the final output.
5. **Speed:** The pipeline processes a patient with ~6-8 notes in roughly **180-210 seconds**. The vast majority of this time is spent waiting through exponential backoff to respect the strict 12,000 TPM limit of Groq's Free Tier. If deployed on a paid tier, processing would take <10 seconds per patient.
6. **Cost:** Token usage was strictly optimized. A typical note is condensed down to roughly `1000 - 3000 tokens`. Synthesizing an entire 8-note patient history consumes roughly `5,000 - 7,000 tokens`. Because we utilize the `llama-3.3-70b-versatile` open-source model via API, inference cost is less than $0.01 per patient.

## 3. What Worked & What Didn't

**What worked well:**
- Injecting line numbers into the source text before LLM inference. This turned an impossible extraction task into a trivial array-lookup task for the model.
- The Python fuzzy-matcher post-processor. LLMs naturally try to grammatically fix clinical text (e.g. fixing typos in the span). The python script caught these and reverted them to the raw source typos to maintain perfect assignment compliance.

**What didn't work:**
- Relying on the LLM to format dates natively. Despite strict prompt engineering ("MM/YYYY"), Llama-3.3 consistently defaulted to ISO formats or conversational dates. Writing a regex-based Python date normalizer was required to handle this consistently.

## 4. Instructions for Running the Code

The codebase is built with zero external dependencies other than the `openai` and `tqdm` packages.

### Environment Setup
The pipeline relies on environment variables, fully complying with assignment requirements, ensuring no secrets are hardcoded.

```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API config
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="llama-3.3-70b-versatile"
```

### Running the Pipeline
The `main.py` entrypoint orchestrates the entire workflow.

```bash
python main.py \
  --data-dir ./dev \
  --patient-list ./dev_patients.json \
  --output-dir ./output_dev \
  --verbose
```

### Additional CLI Arguments Included
- `--delay [FLOAT]`: Specifies the sleep duration between API calls. Defaults to `5.0`. Useful for aggressively throttling requests to stay under free-tier TPM limits.
- `--verbose`: Prints detailed step-by-step extractions and post-processing corrections to stdout.
- `--taxonomy [PATH]`: Overrides the auto-detected `taxonomy.json` path if stored elsewhere.
- `--temperature [FLOAT]`: Tunes the LLM entropy. Defaults to `0.1` for maximum deterministic extraction.
