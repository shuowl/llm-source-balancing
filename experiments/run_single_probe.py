#!/usr/bin/env python3
"""
run_single_probe.py - Run a single probe variant using compute_probs_single_variant.py.

This module handles the execution of individual probe variants using the
compute_probs_single_variant.py script that focuses on probability computation.

USAGE:
------
Can be used as a module or run directly:

python experiments/run_single_probe.py \
    --experiment_name csqa__qwen3_1_7b__d1nu1nin__nocot \
    --probe_variant bare \
    --eval_jsonl data/csqa_test.jsonl
"""

import os
import pathlib
import subprocess
import sys

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.exp_name import parse_experiment_name


def get_result_filename(probe_variant: str) -> str:
    """Get the result filename based on probe variant."""
    return f"probs_{probe_variant}.jsonl"


def generate_reasoning(experiment_name: str, probe_variant: str, eval_jsonl: str,
                      results_root: str = "results",
                      vllm_tp_size: int = 1) -> str:
    """Generate vLLM reasoning for a single probe variant.

    Args:
        experiment_name: Name of the experiment
        probe_variant: Which probe variant to run
        eval_jsonl: Path to evaluation data
        results_root: Root directory for results
        vllm_tp_size: Tensor parallel size for vLLM

    Returns:
        Path to the reasoning output file
    """
    res_dir = pathlib.Path(results_root) / experiment_name
    output_path = res_dir / f"{probe_variant}_vllm_reasoning.jsonl"

    # Skip if file already exists AND is complete; otherwise fail fast
    if output_path.exists():
        try:
            with open(eval_jsonl, 'r') as f:
                expected_lines = sum(1 for _ in f)
        except Exception:
            expected_lines = None
        try:
            with open(output_path, 'r') as f:
                actual_lines = sum(1 for _ in f)
        except Exception:
            actual_lines = None
        if expected_lines is not None and actual_lines is not None and actual_lines != expected_lines:
            raise FileNotFoundError(
                f"Detected incomplete vLLM reasoning for {probe_variant}: "
                f"{actual_lines}/{expected_lines} lines at {output_path}. Please regenerate reasoning and retry."
            )
        print(f"vLLM reasoning already exists for {probe_variant}: {output_path}")
        return str(output_path)

    print(f"Generating vLLM reasoning for {probe_variant}...")

    cmd = [
        sys.executable,
        str(ROOT / "core" / "generate_with_vllm.py"),
        "--experiment_name", experiment_name,
        "--probe_variants", probe_variant,  # Note: plural form
        "--eval_jsonl", eval_jsonl,
        "--output_jsonl", str(output_path),
        "--tp_size", str(vllm_tp_size)
    ]

    # Add canonical wrong file if needed for non-bare variants
    if probe_variant != "bare":
        canonical_wrong_path = res_dir / "canonical_wrong.jsonl"
        if canonical_wrong_path.exists():
            cmd.extend(["--canonical_wrong_jsonl", str(canonical_wrong_path)])

    # If we're in an accelerate environment, escape it so vLLM gets full GPU access
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
        raise RuntimeError(f"Failed to generate vLLM reasoning for {probe_variant}")

    return str(output_path)


def run_single_probe(experiment_name: str, probe_variant: str, eval_jsonl: str,
                    results_root: str = "results", batch_size: int = 8,
                    vllm_tp_size: int = 1, hf_gpus: int = 1) -> str:
    """Run a single probe variant and save results to file.

    Args:
        experiment_name: Name of the experiment
        probe_variant: Which probe variant to run
        eval_jsonl: Path to evaluation data
        results_root: Root directory for results
        batch_size: Number of examples to process in each batch (default: 8)
        vllm_tp_size: Tensor parallel size for vLLM (single replica sharded across N GPUs)
        hf_gpus: Number of GPUs for HF accelerate (data parallel, via --num_processes)

    Returns:
        Path to the output file
    """

    res_dir = pathlib.Path(results_root) / experiment_name
    out_filename = get_result_filename(probe_variant)
    out_path = res_dir / out_filename

    # Skip if file already exists
    if out_path.exists():
        return str(out_path)

    # Parse experiment to check if it's a reasoning model
    exp = parse_experiment_name(experiment_name)

    # For reasoning models, generate reasoning first
    if exp.reasoning_mode:
        reasoning_path = generate_reasoning(
            experiment_name, probe_variant, eval_jsonl, results_root,
            vllm_tp_size
        )

        # Sanity check: ensure reasoning file exists and is complete before proceeding
        reasoning_file = res_dir / f"{probe_variant}_vllm_reasoning.jsonl"

        if not reasoning_file.exists():
            raise FileNotFoundError(
                f"vLLM reasoning file not found at {reasoning_file}. "
                f"Failed to generate vLLM reasoning for {probe_variant}. "
                f"Please check the vLLM generation logs."
            )
        # Check completeness vs eval file line count
        try:
            with open(eval_jsonl, 'r') as f:
                expected_lines = sum(1 for _ in f)
        except Exception:
            expected_lines = None
        if expected_lines is not None:
            with open(reasoning_file, 'r') as f:
                actual_lines = sum(1 for _ in f)
            if actual_lines != expected_lines:
                raise FileNotFoundError(
                    f"Detected incomplete vLLM reasoning for {probe_variant}: "
                    f"{actual_lines}/{expected_lines} lines at {reasoning_file}. Please regenerate reasoning and retry."
                )

    # Check if this is an OpenAI model (starts with gpt_ or has o1/o4 pattern)
    is_openai_model = (
        exp.model_key.startswith('gpt_') or
        'o1' in exp.hf_model_id or
        'o4' in exp.hf_model_id or
        exp.hf_model_id.startswith('gpt-') or
        exp.hf_model_id.startswith('o1-') or
        exp.hf_model_id.startswith('o4-')
    )

    if is_openai_model:
        compute_script = str(ROOT / "core" / "compute_answers_and_probs_openai.py")
    else:
        compute_script = str(ROOT / "core" / "compute_probs_single_variant.py")

    # Build command based on model type
    if is_openai_model:
        # API-based models don't use accelerate
        cmd = [
            "python", compute_script,
            "--experiment_name", experiment_name,
            "--probe_variant", probe_variant,
            "--eval_jsonl", eval_jsonl,
            "--output_jsonl", str(out_path),
            "--max-concurrent", str(batch_size),
        ]
    else:
        # HuggingFace models: accelerate launch for data parallelism
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(hf_gpus),
            compute_script,
            "--experiment_name", experiment_name,
            "--probe_variant", probe_variant,
            "--eval_jsonl", eval_jsonl,
            "--output_jsonl", str(out_path),
            "--batch_size", str(batch_size)
        ]

    # Add canonical wrong file if needed for non-bare variants
    if probe_variant != "bare":
        canonical_wrong_path = res_dir / "canonical_wrong.jsonl"
        if canonical_wrong_path.exists():
            cmd.extend(["--canonical_wrong_jsonl", str(canonical_wrong_path)])

    # Show phase information
    if exp.reasoning_mode and 'LOCAL_RANK' in os.environ:
        print("\n" + "="*60)
        print("PHASE 2: HF Probing (using accelerate multi-GPU)")
        print("="*60)

    print(f"Running {probe_variant}...")

    # Run without capturing output to show real-time progress
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to run {probe_variant}")

    return str(out_path)


def main():
    """Main entry point for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Run a single probe variant")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--probe_variant", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of examples to process in each batch (default: 8)")
    parser.add_argument("--vllm-tp-size", type=int, default=1,
                        help="Tensor parallel size for vLLM (default: 1)")
    parser.add_argument("--hf-gpus", type=int, default=1,
                        help="Number of GPUs for HF accelerate (default: 1)")

    args = parser.parse_args()

    output_path = run_single_probe(
        args.experiment_name,
        args.probe_variant,
        args.eval_jsonl,
        args.results_root,
        args.batch_size,
        args.vllm_tp_size,
        args.hf_gpus
    )

    print(f"\nGenerated: {output_path}")


if __name__ == "__main__":
    main()
