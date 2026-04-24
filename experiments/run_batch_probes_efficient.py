#!/usr/bin/env python3
"""
run_batch_probes_efficient.py - Efficiently run multiple probe variants with single model load.

This module processes all non-bare probe variants using compute_probs_multi_variant.py,
which loads the model only once and processes all variants, dramatically reducing
the time needed to run experiments.

For reasoning models, it generates vLLM reasoning first before computing HF probabilities.
"""

import argparse
import json
import pathlib
import subprocess
import sys
from typing import List

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name


def generate_reasoning_batch(
    experiment_name: str,
    probe_variants: List[str],
    eval_jsonl: str,
    results_root: str = "results",
    vllm_tp_size: int = 1,
) -> None:
    """Generate vLLM reasoning for multiple probe variants efficiently.

    Args:
        experiment_name: Name of the experiment
        probe_variants: List of probe variants to process
        eval_jsonl: Path to evaluation data
        results_root: Root directory for results
        vllm_tp_size: Tensor parallel size for vLLM
    """
    res_dir = pathlib.Path(results_root) / experiment_name

    # Determine expected number of lines from eval_jsonl
    try:
        with open(eval_jsonl, 'r') as f:
            expected_lines = sum(1 for _ in f)
    except Exception:
        expected_lines = None

    # Check which variants need reasoning generation (missing or incomplete)
    variants_to_generate = []
    incomplete_details = []
    for pv in probe_variants:
        reasoning_path = res_dir / f"{pv}_vllm_reasoning.jsonl"

        needs_generation = False
        if not reasoning_path.exists():
            needs_generation = True
            actual_lines = 0
        elif expected_lines is not None:
            try:
                with open(reasoning_path, 'r') as rf:
                    actual_lines = sum(1 for _ in rf)
                if actual_lines != expected_lines:
                    needs_generation = True
            except Exception:
                needs_generation = True
                actual_lines = None
        else:
            actual_lines = None

        if needs_generation:
            variants_to_generate.append(pv)
            detail = f"{pv}: found {actual_lines if actual_lines is not None else 'unreadable'}/{expected_lines if expected_lines is not None else 'unknown'} lines at {reasoning_path}"
            incomplete_details.append(detail)

    if not variants_to_generate:
        print(f"vLLM reasoning already exists and is complete for all specified probe variants")
        return

    print(f"\nGenerating vLLM reasoning for {len(variants_to_generate)} probe variants...")
    print(f"Variants: {', '.join(variants_to_generate)}")

    cmd = [
        sys.executable,
        str(ROOT / "core" / "generate_with_vllm.py"),
        "--experiment_name", experiment_name,
        "--probe_variants", ",".join(variants_to_generate),
        "--eval_jsonl", eval_jsonl,
        "--tp_size", str(vllm_tp_size)
    ]

    # Add canonical wrong file if it exists (it should for non-bare variants)
    canonical_wrong_path = res_dir / "canonical_wrong.jsonl"
    if canonical_wrong_path.exists():
        cmd.extend(["--canonical_wrong_jsonl", str(canonical_wrong_path)])

    # If we're in an accelerate environment, escape it so vLLM gets full GPU access
    import os
    if 'LOCAL_RANK' in os.environ:
        print("\n" + "="*60)
        print("PHASE 1: vLLM Generation (escaping accelerate to use all GPUs)")
        print("="*60)

        wrapper_script = f"""
import subprocess
import sys
cmd = {cmd}
result = subprocess.run(cmd)
sys.exit(result.returncode)
"""

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_script)
            wrapper_path = f.name

        clean_env = os.environ.copy()
        accelerate_vars = ['LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                          'MASTER_ADDR', 'MASTER_PORT', 'ACCELERATE_MIXED_PRECISION',
                          'CUDA_VISIBLE_DEVICES']
        for var in accelerate_vars:
            clean_env.pop(var, None)

        print("Running vLLM generation with unrestricted GPU access...")

        wrapper_cmd = [sys.executable, wrapper_path]
        result = subprocess.run(wrapper_cmd, env=clean_env)

        os.unlink(wrapper_path)
    else:
        result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate vLLM reasoning for probe variants")

    print(f"✓ Successfully generated vLLM reasoning for {len(variants_to_generate)} variants")


def run_batch_probes_efficient(
    experiment_name: str,
    probe_variants: List[str],
    eval_jsonl: str,
    results_root: str = "results",
    batch_size: int = 8,
    vllm_tp_size: int = 1,
    hf_gpus: int = 1,
) -> None:
    """Run batch processing for multiple probe variants.

    For local models: Uses compute_probs_multi_variant.py which loads the model once
    and processes all variants efficiently.

    For API models (OpenAI): Calls the appropriate compute script for each variant
    sequentially (API models don't benefit from batch loading).

    Args:
        experiment_name: Name of the experiment
        probe_variants: List of probe variants to process (excluding bare)
        eval_jsonl: Path to evaluation data
        results_root: Root directory for results
        batch_size: Number of examples to process in each batch / max concurrent for OpenAI
        vllm_tp_size: Tensor parallel size for vLLM (single replica sharded across N GPUs)
        hf_gpus: Number of GPUs for HF accelerate (data parallel, via --num_processes)
    """

    results_dir = pathlib.Path(results_root) / experiment_name
    canonical_wrong_path = results_dir / "canonical_wrong.jsonl"

    if not canonical_wrong_path.exists():
        raise FileNotFoundError(
            f"Canonical wrong file not found at {canonical_wrong_path}. "
            "Run bare probe first to generate it."
        )

    # Check which probe variants need to be processed
    probes_to_process = []
    for pv in probe_variants:
        output_file = results_dir / f"probs_{pv}.jsonl"
        if not output_file.exists():
            probes_to_process.append(pv)

    if not probes_to_process:
        print(f"\nAll files already exist for specified probe variants")
        return

    exp = parse_experiment_name(experiment_name)

    is_openai_model = (
        exp.model_key.startswith('gpt_') or
        'o1' in exp.hf_model_id or
        'o4' in exp.hf_model_id or
        exp.hf_model_id.startswith('gpt-') or
        exp.hf_model_id.startswith('o1-') or
        exp.hf_model_id.startswith('o4-')
    )

    if is_openai_model:
        print(f"\nProcessing OpenAI model variants sequentially...")
        print(f"Probe variants: {', '.join(probes_to_process)}")

        compute_script = str(ROOT / "core" / "compute_answers_and_probs_openai.py")

        for pv in probes_to_process:
            output_file = results_dir / f"probs_{pv}.jsonl"
            print(f"\nProcessing {pv}...")

            cmd = [
                sys.executable,
                compute_script,
                "--experiment_name", experiment_name,
                "--probe_variant", pv,
                "--eval_jsonl", eval_jsonl,
                "--output_jsonl", str(output_file),
                "--max-concurrent", str(batch_size),
            ]

            result = subprocess.run(cmd)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to process probe variant {pv}")

            print(f"✓ Successfully processed {pv}")

        print(f"\n✓ Successfully processed {len(probes_to_process)} probe variants")
        return

    # Everything below this point is for HuggingFace models only
    # For reasoning models, generate reasoning first
    if exp.reasoning_mode:
        generate_reasoning_batch(
            experiment_name, probes_to_process, eval_jsonl, results_root,
            vllm_tp_size
        )

        # Sanity check: ensure all reasoning files exist before proceeding
        missing_files = []
        for pv in probes_to_process:
            reasoning_file = results_dir / f"{pv}_vllm_reasoning.jsonl"
            if not reasoning_file.exists():
                missing_files.append(str(reasoning_file))

        if missing_files:
            raise FileNotFoundError(
                "vLLM reasoning files not found:\n" + "\n".join(missing_files) +
                "\nFailed to generate vLLM reasoning for some probe variants. "
                "Please check the vLLM generation logs."
            )

    print(f"\nRunning efficient batch processing for {len(probes_to_process)} probe variants...")
    print(f"Probe variants: {', '.join(probes_to_process)}")
    print(f"\nThis will load the model only ONCE and process all variants.")

    if hf_gpus > 1:
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(hf_gpus),
            str(ROOT / "core" / "compute_probs_multi_variant.py"),
            "--experiment_name", experiment_name,
            "--probe_variants", ",".join(probes_to_process),
            "--eval_jsonl", eval_jsonl,
            "--output_dir", str(results_dir),
            "--batch_size", str(batch_size)
        ]
    else:
        cmd = [
            sys.executable,
            str(ROOT / "core" / "compute_probs_multi_variant.py"),
            "--experiment_name", experiment_name,
            "--probe_variants", ",".join(probes_to_process),
            "--eval_jsonl", eval_jsonl,
            "--output_dir", str(results_dir),
            "--batch_size", str(batch_size)
        ]

    if canonical_wrong_path.exists():
        cmd.extend(["--canonical_wrong_jsonl", str(canonical_wrong_path)])

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to run batch probe variants")

    print(f"\n✓ Successfully processed {len(probes_to_process)} probe variants with single model load")


def main():
    parser = argparse.ArgumentParser(description="Efficient batch processing for probe variants")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--probe_variants", nargs="+", required=True,
                       help="List of probe variants to run (excluding bare)")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Number of examples to process in each batch (default: 32)")
    parser.add_argument("--vllm-tp-size", type=int, default=1,
                       help="Tensor parallel size for vLLM (default: 1)")
    parser.add_argument("--hf-gpus", type=int, default=1,
                       help="Number of GPUs for HF accelerate (default: 1)")

    args = parser.parse_args()

    run_batch_probes_efficient(
        args.experiment_name,
        args.probe_variants,
        args.eval_jsonl,
        args.results_root,
        args.batch_size,
        args.vllm_tp_size,
        args.hf_gpus,
    )


if __name__ == "__main__":
    main()
