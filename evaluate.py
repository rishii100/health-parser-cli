"""
Evaluation module — scores pipeline output against ground truth labels.
Computes: condition identification (precision/recall/F1), status accuracy,
date accuracy, and evidence quality.
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(name):
    """Normalize condition name for fuzzy matching."""
    name = name.lower().strip()
    # Remove common noise words
    for noise in ["status post ", "history of ", "s/p "]:
        name = name.replace(noise, "")
    return name


def name_similarity(a, b):
    """Compute similarity between two condition names."""
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def match_conditions(predicted, ground_truth, threshold=0.55):
    """
    Match predicted conditions to ground truth using fuzzy name matching
    + taxonomy bonus. Returns (matches, unmatched_pred_indices, unmatched_gt_indices).
    """
    n_pred = len(predicted)
    n_gt = len(ground_truth)

    # Build similarity matrix
    sim_matrix = []
    for i, pred in enumerate(predicted):
        for j, gt in enumerate(ground_truth):
            sim = name_similarity(pred["condition_name"], gt["condition_name"])
            # Bonus if category and subcategory match
            if pred.get("category") == gt.get("category"):
                sim += 0.05
                if pred.get("subcategory") == gt.get("subcategory"):
                    sim += 0.05
            sim_matrix.append((sim, i, j))

    # Greedy best-first matching
    sim_matrix.sort(reverse=True, key=lambda x: x[0])
    matched_pred = set()
    matched_gt = set()
    matches = []

    for sim, i, j in sim_matrix:
        if sim < threshold:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matches.append((i, j, sim))
        matched_pred.add(i)
        matched_gt.add(j)

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]

    return matches, unmatched_pred, unmatched_gt


def evaluate_evidence(pred_evidence, gt_evidence):
    """Evaluate evidence quality: how many GT evidence entries are covered by predictions."""
    if not gt_evidence:
        return 1.0, 0, 0

    gt_keys = set()
    for ev in gt_evidence:
        gt_keys.add((ev.get("note_id"), ev.get("line_no")))

    pred_keys = set()
    for ev in pred_evidence:
        pred_keys.add((ev.get("note_id"), ev.get("line_no")))

    # Also check nearby lines (±2) for approximate matches
    covered = 0
    for gt_key in gt_keys:
        if gt_key in pred_keys:
            covered += 1
        else:
            # Check ±2 lines
            note_id, line_no = gt_key
            for offset in range(-2, 3):
                if (note_id, line_no + offset) in pred_keys:
                    covered += 1
                    break

    recall = covered / len(gt_keys)
    precision = covered / max(len(pred_keys), 1)

    return recall, len(gt_keys), len(pred_keys)


def evaluate_patient(pred_path, gt_path):
    """Evaluate one patient's predictions against ground truth."""
    pred = load_json(pred_path)
    gt = load_json(gt_path)

    pred_conds = pred.get("conditions", [])
    gt_conds = gt.get("conditions", [])

    matches, unmatched_pred, unmatched_gt = match_conditions(pred_conds, gt_conds)

    # Compute metrics
    n_matched = len(matches)
    precision = n_matched / max(len(pred_conds), 1)
    recall = n_matched / max(len(gt_conds), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    status_correct = 0
    date_correct = 0
    evidence_recall_sum = 0.0
    match_details = []

    for pred_i, gt_i, sim in matches:
        p = pred_conds[pred_i]
        g = gt_conds[gt_i]

        s_match = p.get("status") == g.get("status")
        d_match = p.get("onset") == g.get("onset")
        ev_recall, gt_ev_count, pred_ev_count = evaluate_evidence(
            p.get("evidence", []), g.get("evidence", [])
        )

        if s_match:
            status_correct += 1
        if d_match:
            date_correct += 1
        evidence_recall_sum += ev_recall

        match_details.append({
            "pred_name": p.get("condition_name"),
            "gt_name": g.get("condition_name"),
            "similarity": round(sim, 3),
            "status": "✓" if s_match else f"✗ pred={p.get('status')} gt={g.get('status')}",
            "onset": "✓" if d_match else f"✗ pred={p.get('onset')} gt={g.get('onset')}",
            "evidence": f"{ev_recall:.0%} ({pred_ev_count} pred / {gt_ev_count} gt)",
        })

    results = {
        "patient_id": gt.get("patient_id"),
        "conditions": {
            "ground_truth": len(gt_conds),
            "predicted": len(pred_conds),
            "matched": n_matched,
            "false_positives": len(unmatched_pred),
            "false_negatives": len(unmatched_gt),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        },
        "status_accuracy": round(status_correct / max(n_matched, 1), 3),
        "date_accuracy": round(date_correct / max(n_matched, 1), 3),
        "avg_evidence_recall": round(evidence_recall_sum / max(n_matched, 1), 3),
        "match_details": match_details,
        "false_positives": [pred_conds[i].get("condition_name") for i in unmatched_pred],
        "false_negatives": [gt_conds[j].get("condition_name") for j in unmatched_gt],
    }

    return results


def print_patient_report(results):
    """Print a formatted evaluation report for one patient."""
    pid = results["patient_id"]
    c = results["conditions"]

    print(f"\n{'='*70}")
    print(f"  {pid}")
    print(f"{'='*70}")
    print(f"  Conditions:  {c['matched']}/{c['ground_truth']} matched "
          f"(P={c['precision']:.1%}  R={c['recall']:.1%}  F1={c['f1']:.1%})")
    print(f"  Status acc:  {results['status_accuracy']:.1%}")
    print(f"  Date acc:    {results['date_accuracy']:.1%}")
    print(f"  Evidence:    {results['avg_evidence_recall']:.1%} avg recall")

    if results["match_details"]:
        print(f"\n  {'Pred Name':<45} {'GT Name':<45} Status  Onset   Evidence")
        print(f"  {'-'*44} {'-'*44} {'-'*6} {'-'*6} {'-'*15}")
        for d in results["match_details"]:
            pred_short = d["pred_name"][:43]
            gt_short = d["gt_name"][:43]
            s = "✓" if d["status"] == "✓" else "✗"
            o = "✓" if d["onset"] == "✓" else "✗"
            print(f"  {pred_short:<45} {gt_short:<45} {s:<7} {o:<7} {d['evidence']}")

    if results["false_negatives"]:
        print(f"\n  ❌ Missed ({len(results['false_negatives'])}):")
        for name in results["false_negatives"]:
            print(f"     - {name}")

    if results["false_positives"]:
        print(f"\n  ⚠️  Extra ({len(results['false_positives'])}):")
        for name in results["false_positives"]:
            print(f"     - {name}")


def evaluate_all(output_dir, labels_dir):
    """Evaluate all patients in output_dir against labels_dir."""
    output_path = Path(output_dir)
    labels_path = Path(labels_dir)

    all_results = []
    total_gt = total_pred = total_matched = 0
    total_status = total_date = total_ev = 0
    n_matched_total = 0

    for label_file in sorted(labels_path.glob("patient_*.json")):
        patient_id = label_file.stem
        pred_file = output_path / f"{patient_id}.json"

        if not pred_file.exists():
            print(f"⚠ No prediction found for {patient_id}, skipping")
            continue

        results = evaluate_patient(str(pred_file), str(label_file))
        all_results.append(results)
        print_patient_report(results)

        c = results["conditions"]
        total_gt += c["ground_truth"]
        total_pred += c["predicted"]
        total_matched += c["matched"]
        n_matched_total += c["matched"]
        total_status += results["status_accuracy"] * c["matched"]
        total_date += results["date_accuracy"] * c["matched"]
        total_ev += results["avg_evidence_recall"] * c["matched"]

    if all_results:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE SCORES ({len(all_results)} patients)")
        print(f"{'='*70}")
        agg_p = total_matched / max(total_pred, 1)
        agg_r = total_matched / max(total_gt, 1)
        agg_f1 = 2 * agg_p * agg_r / max(agg_p + agg_r, 1e-6)
        print(f"  Condition ID:   P={agg_p:.1%}  R={agg_r:.1%}  F1={agg_f1:.1%}")
        print(f"  Status acc:     {total_status / max(n_matched_total, 1):.1%}")
        print(f"  Date acc:       {total_date / max(n_matched_total, 1):.1%}")
        print(f"  Evidence recall: {total_ev / max(n_matched_total, 1):.1%}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate extraction output against ground truth")
    parser.add_argument("--output-dir", required=True, help="Directory with predicted patient_XX.json files")
    parser.add_argument("--labels-dir", required=True, help="Directory with ground truth patient_XX.json files")
    args = parser.parse_args()

    evaluate_all(args.output_dir, args.labels_dir)
