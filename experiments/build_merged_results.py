#!/usr/bin/env python3
"""
build_merged_results.py - Build comprehensive merged results file from all probe variants.

This script combines results from unified probe files (probs_<variant>.jsonl) and canonical 
wrong answers into a single JSONL file with additional computed fields for metrics analysis.

Output file (merged_results.jsonl) contains one entry per probe variant per question:
- qid: Question ID
- gold: Correct answer letter
- canonical_wrong: Most probable wrong answer letter
- choices: List of answer choice texts
- probe_variant: Which probe variant (bare, upos, uneg, dpos, dneg, updp, updn, undp, undn)
- output_letter: Model's predicted answer letter
- model_correct: Boolean - whether model was correct without priors (copied from bare probe)
- output_correct: Boolean - whether model's predicted answer matches gold for this probe variant
- user_present: Boolean indicating if user prior is present
- doc_present: Boolean indicating if document prior is present
- user_correct: Boolean indicating if user prior supports correct answer
- doc_correct: Boolean indicating if document prior supports correct answer
- user_first: Boolean indicating if user prior comes before doc prior (only relevant for double-prior)
- probs: List of probability objects for each choice
- reasoning_mode_generated_text: (Optional) Full generated text for reasoning mode, including thinking

Note: Each question will have 9 entries (one per probe variant)
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Any

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name


def get_probe_variant_flags(probe_variant: str, exp) -> Dict[str, bool]:
    """Determine which priors are present and their correctness for a probe variant."""
    flags = {
        "user_present": False,
        "doc_present": False,
        "user_correct": False,
        "doc_correct": False,
        "user_first": False  # Only relevant for double-prior variants
    }
    
    if probe_variant == "bare":
        # No priors present
        return flags
    
    # Single prior variants
    if probe_variant == "upos":
        flags["user_present"] = True
        flags["user_correct"] = True
    elif probe_variant == "uneg":
        flags["user_present"] = True
        flags["user_correct"] = False
    elif probe_variant == "dpos":
        flags["doc_present"] = True
        flags["doc_correct"] = True
    elif probe_variant == "dneg":
        flags["doc_present"] = True
        flags["doc_correct"] = False
    
    # Double prior variants
    elif len(probe_variant) == 4:  # e.g., updp, dpun
        flags["user_present"] = True
        flags["doc_present"] = True
        flags["user_first"] = exp.user_first  # Order matters for double-prior
        
        # Determine correctness based on variant and experiment order
        if exp.user_first:
            # User-first experiments: updp, updn, undp, undn
            flags["user_correct"] = probe_variant[1] == 'p'
            flags["doc_correct"] = probe_variant[3] == 'p'
        else:
            # Doc-first experiments: dpup, dpun, dnup, dnun
            flags["doc_correct"] = probe_variant[1] == 'p'
            flags["user_correct"] = probe_variant[3] == 'p'
    
    return flags


def build_merged_results(experiment_name: str, results_root: str = "results") -> None:
    """Build a comprehensive merged results file for an experiment."""
    
    exp = parse_experiment_name(experiment_name)
    results_dir = pathlib.Path(results_root) / experiment_name
    
    # Determine which probe variants to expect
    probe_variants = ["bare", "upos", "uneg", "dpos", "dneg"]
    if exp.user_first:
        probe_variants.extend(["updp", "updn", "undp", "undn"])
    else:
        probe_variants.extend(["dpup", "dpun", "dnup", "dnun"])
    
    # Load canonical wrong answers
    canonical_wrong = {}
    canonical_wrong_path = results_dir / "canonical_wrong.jsonl"
    if canonical_wrong_path.exists():
        with open(canonical_wrong_path) as f:
            for line in f:
                data = json.loads(line)
                canonical_wrong[data["qid"]] = data["canonical_wrong"]
    else:
        print(f"Warning: No canonical_wrong.jsonl found at {canonical_wrong_path}")
    
    # Load all results organized by qid
    all_results = defaultdict(lambda: {"probe_results": {}})
    
    for probe_variant in probe_variants:
        # Load unified results file
        probe_file = results_dir / f"probs_{probe_variant}.jsonl"
        if probe_file.exists():
            with open(probe_file) as f:
                for line in f:
                    data = json.loads(line)
                    qid = data["qid"]
                    
                    # Get probe variant flags
                    flags = get_probe_variant_flags(probe_variant, exp)
                    
                    # Store results
                    if probe_variant not in all_results[qid]["probe_results"]:
                        all_results[qid]["probe_results"][probe_variant] = {}
                    
                    # Get fields from new compute scripts
                    output_letter = data.get("output_letter")
                    output_correct = data.get("output_correct", False)
                    
                    all_results[qid]["probe_results"][probe_variant].update({
                        "output_letter": output_letter,
                        "output_correct": output_correct,
                        "probs": data.get("probs", []),
                        "user_present": flags["user_present"],
                        "doc_present": flags["doc_present"],
                        "user_correct": flags["user_correct"],
                        "doc_correct": flags["doc_correct"],
                        "user_first": flags["user_first"]
                    })
                    
                    # Add reasoning_mode_generated_text for reasoning mode if present
                    if "reasoning_mode_generated_text" in data:
                        all_results[qid]["probe_results"][probe_variant]["reasoning_mode_generated_text"] = data["reasoning_mode_generated_text"]
                    
                    # Store common fields
                    if "gold" not in all_results[qid]:
                        all_results[qid]["gold"] = data["gold"]
                        all_results[qid]["canonical_wrong"] = canonical_wrong.get(qid, "")
                        all_results[qid]["choices"] = data.get("choices", [])
    
    # Write merged results with flattened structure
    output_path = results_dir / "merged_results.jsonl"
    with open(output_path, 'w') as f:
        for qid, data in all_results.items():
            # Get model_correct from bare probe (which only has output_correct)
            bare_data = data["probe_results"].get("bare", {})
            model_correct = bare_data.get("output_correct", False)
            
            # Create a list of results, one for each probe variant
            for probe_variant, probe_data in data["probe_results"].items():
                result = {
                    "qid": qid,
                    "gold": data["gold"],
                    "canonical_wrong": data["canonical_wrong"],
                    "choices": data["choices"],
                    "probe_variant": probe_variant,
                    "model_correct": model_correct,  # Same for all variants (from bare)
                }
                
                # Add output results
                result["output_letter"] = probe_data.get("output_letter")
                result["output_correct"] = probe_data.get("output_correct", False)
                
                # Add reasoning_mode_generated_text for reasoning mode if present
                if "reasoning_mode_generated_text" in probe_data:
                    result["reasoning_mode_generated_text"] = probe_data["reasoning_mode_generated_text"]
                
                # Add prior presence/correctness flags
                result["user_present"] = probe_data["user_present"]
                result["doc_present"] = probe_data["doc_present"]
                result["user_correct"] = probe_data["user_correct"]
                result["doc_correct"] = probe_data["doc_correct"]
                result["user_first"] = probe_data["user_first"]
                
                # Add probability results if available
                if "probs" in probe_data:
                    result["probs"] = probe_data["probs"]
                
                f.write(json.dumps(result) + '\n')
    
    total_entries = sum(len(data["probe_results"]) for data in all_results.values())
    print(f"✓ Built merged results for {len(all_results)} questions ({total_entries} total entries)")
    print(f"✓ Saved to: {output_path}")
    
    # Print summary statistics
    print("\nProbe variant coverage:")
    probe_coverage = defaultdict(int)
    for qid, data in all_results.items():
        for pv in data["probe_results"]:
            probe_coverage[pv] += 1
    
    for pv in probe_variants:
        count = probe_coverage.get(pv, 0)
        print(f"  {pv}: {count} questions")
    
    # Check for missing probe files
    missing_probes = []
    for pv in probe_variants:
        probe_file = results_dir / f"probs_{pv}.jsonl"
        if not probe_file.exists():
            missing_probes.append(pv)
    
    if missing_probes:
        print("\nMissing probe files:")
        for pv in missing_probes:
            print(f"  probs_{pv}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Build comprehensive merged results file")
    parser.add_argument("--experiment_name", required=True,
                       help="Name of the experiment")
    parser.add_argument("--results_root", default="results",
                       help="Root directory for results")
    
    args = parser.parse_args()
    
    build_merged_results(args.experiment_name, args.results_root)


if __name__ == "__main__":
    main()