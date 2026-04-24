# How Large Language Models Balance Internal Knowledge with User and Document Assertions

## News

- **[2026-04]** Paper accepted to **ACL 2026 Findings**.
- **Code**: https://github.com/shuowl/llm-source-balancing
- **Models** (SFT LoRA adapters): https://huggingface.co/codecodebear/llm_source_balancing_sft_lora
- **Paper**: coming soon

## Paper Overview

![Three-source interaction framework](img/main_framework.svg)

Large language models often need to integrate multiple information sources in
real-world systems such as RAG and chat-based assistants. In these settings, a
model may need to balance:
(1) its own parametric knowledge,
(2) a user's assertion, and
(3) information attributed to retrieved documents.

Prior work on knowledge conflict and sycophancy has mostly studied *binary*
conflicts — parametric knowledge vs. retrieved context, or parametric
knowledge vs. user beliefs. However, realistic assistant systems often involve
all three sources at the same time.

To study this setting, we propose a **three-source interaction framework**
that evaluates how LLMs weigh parametric knowledge, user assertions, and
document assertions under controlled source-conflict scenarios.

### Research Questions

1. **Source reliance:** How do LLMs weigh internal parametric knowledge, user assertions, and document assertions?
2. **Discrimination ability:** Can LLMs distinguish helpful external information from harmful or misleading information?
3. **Post-training effects:** How does post-training affect model preferences in multi-source settings?

### Framework

We construct **13 probe variants** for each question, including:

- a bare probe with no external assertion,
- single-source probes with either a user or document assertion,
- double-source probes where both user and document assertions are present,
- cases where external assertions are correct, incorrect, or conflicting.

We evaluate **27 LLMs** from three model families (GPT-4o, Llama 3 / 3.1, and
Qwen3) on **CommonsenseQA** and **GSM8K-MC**.

### Main Findings

- **Document preference:** Most models rely more on document-attributed assertions than user-attributed assertions.
- **Post-training reinforces document reliance:** Instruction-tuned or post-trained models tend to show stronger document preference.
- **Models are often impressionable:** Many models accept helpful external information but also fail to resist harmful or misleading assertions.
- **Fine-tuning helps:** Supervised fine-tuning on diverse source-interaction data improves models' ability to distinguish helpful from harmful external information, while largely preserving general capabilities.

### Why This Matters & Future Directions

Models do not always combine sources rationally — they may over-trust
document-like content, underweight user input, or absorb misleading claims.
We see this work as a first step toward a broader research direction on
multi-source knowledge integration, with natural extensions to:

- **Multi-source knowledge integration:** Extending the framework beyond user and document assertions to other sources, such as tool outputs, search results, memory modules, database entries, environment feedback, or other agents' messages.

- **Multi-modal source interaction:** Studying how external information from different modalities, such as text, images, audio, and video, affects model decisions, especially when different modalities provide conflicting or partially consistent evidence.

- **Agentic settings:** Applying the framework to language agents that perform planning, reasoning, tool use, and interaction with external environments. In these settings, agents may need to reconcile information from users, retrieved documents, tools, sensors, previous actions, and changing world states.

- **Robust training and mitigation:** Developing training methods that help models selectively accept helpful information while resisting misleading or unreliable sources, especially in more realistic long-context, multi-turn, and multi-modal settings.

## Reproducing the Paper

The pipeline has four stages: **(1) Setup** → **(2) Data** → **(3) Probe inference** → **(4) Analysis**, plus optional **SFT** and **standard-benchmark evaluation** of base vs. fine-tuned models.

### Setup

```bash
conda create -n <env> python=3.10 -y
conda activate <env>
pip install -r requirements.txt
```

### Data — prepare CSQA and GSM8K-MC splits

Build the datasets used in the paper (CSQA, GSM8K-MC):

```bash
python data/prepare_datasets.py
```

This produces:
- Per-dataset JSONL files in `data/processed_datasets/{csqa,gsm8k}_default_split/`
- A manifest `data/prepared_datasets.yaml` mapping dataset names to file paths

Dataset and model definitions live in `configs/`: `datasets_config.yaml`
declares each dataset's HF source, split ratios, and seed (consumed by
`prepare_datasets.py`); `models_config.yaml` maps the short model names used
across the pipeline (e.g. `qwen3_8b`, `llama3_8b_instruct`) to their HF
checkpoints.

The exact splits used in the paper are already checked in under
`data/processed_datasets/`, so this step is only needed if you want to
regenerate from scratch.

### Probe inference pipeline

Drives a batch of experiments end-to-end. A recipe expands to many
`(dataset, model, prior_token, instruction, cot_flag)` combinations via a
cartesian product; the pipeline iterates over every combination and, for
each one, runs the bare probe, generates tier sentences, then runs all 8
non-bare probe variants (`upos`/`uneg`, `dpos`/`dneg`, plus 4 double-prior
variants) and aggregates them with the bare probe into a per-question
`merged_results.jsonl`.

```
experiments/build_experiments.py                  # recipe → flat config
       │
       ▼
experiments/generated/<recipe>_config.yaml
       │
       ▼
runner/run_full_pipeline.py                       # top-level driver
       │  for each experiment in the config:
       │
       ├── (1) experiments/run_experiments.py --probe-variant bare
       │        └── experiments/run_single_probe.py
       │             └── core/compute_probs_single_variant.py   (HF models)
       │                  or core/compute_answers_and_probs_openai.py  (OpenAI)
       │                  [reasoning models first call core/generate_with_vllm.py]
       │        then build canonical_wrong.jsonl from bare results
       │
       ├── (2) core/dataset_tier_generator.py
       │        └── T1: deterministic templates (local)
       │        └── T2: GPT-4o paraphrases (via core/llm_client.py)
       │
       └── (3) experiments/run_experiments.py                  # 8 non-bare probes
                └── experiments/run_batch_probes_efficient.py
                     └── core/compute_probs_multi_variant.py   (single model load)
                then experiments/build_merged_results.py for aggregation
       │
       ▼
results/<experiment_name>/
   ├── probs_<variant>.jsonl × 9
   ├── canonical_wrong.jsonl
   └── merged_results.jsonl                       # consumed by sft/extract_sft_data_v5.py
```

#### Run

```bash
# 1. Compose the experiment list from a recipe (cartesian product of fields).
python experiments/build_experiments.py \
    --recipes experiments/recipes/exp1.yaml \
    --out experiments/generated/exp1_config.yaml

# 2. Drive the full 3-step pipeline (bare → tier → all probes) for every experiment.
python runner/run_full_pipeline.py --config experiments/generated/exp1_config.yaml
```


### `analyze/` — metrics & reports

Turns each experiment's `merged_results.jsonl` into per-experiment metric
JSONs (logistic regression, choice-level ratios, distribution statistics),
then rolls everything up across experiments into summary CSVs.

```
results/<experiment_name>/merged_results.jsonl     # produced by the probe pipeline
       │
       ▼
analyze/compute_all_metrics.py                     # orchestrator (one experiment at a time)
       ├── analyze/logistic_regression_analysis.py     → logistic_regression_results.json
       │                                                  (+ logistic_regression_breakdown.json)
       ├── analyze/choice_metrics_analysis.py          → choice_metrics.json
       └── analyze/distribution_metrics_analysis.py    → distribution_metrics.json
       │
       ▼
results/<experiment_name>/ (now also contains the 3 per-experiment metric JSONs)
       │
       ▼
analyze/generate_core_metrics_report.py            # cross-experiment roll-up
```

#### Run

```bash
# 1. Compute per-experiment metrics (logistic / choice / distribution) for every
#    experiment directory. Reads merged_results.jsonl, writes 3 JSONs alongside it.
python analyze/compute_all_metrics.py --config experiments/generated/exp1_config.yaml

# 2. Roll up the per-experiment JSONs into four combined CSVs.
python analyze/generate_core_metrics_report.py \
    --results-dir results \
    --output-dir analyze/exp_results
```

### Supervised fine-tuning — `sft/`

Composes SFT training datasets from pre-computed probe-inference results in
`results/`, mixing helpful and harmful source-conflict examples to teach
models to selectively accept external information. The generated
`*_candidates.jsonl` files serve as input to the actual LoRA fine-tuning,
which we run using
[LlamaFactory](https://github.com/hiyouga/LLaMA-Factory).

The trained LoRA adapters are released on Hugging Face:
https://huggingface.co/codecodebear/llm_source_balancing_sft_lora

### General-capability evaluation of fine-tuned models — `lm_harness_eval/`

Evaluates base vs. SFT models on standard benchmarks (MMLU-Pro, Math Level 5)
to quantify how task-specific SFT on GSM8K / CSQA affects general
capabilities — i.e. whether the source-balancing fine-tuning causes
regression elsewhere. Built as a thin wrapper around
[`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).

#### Contents

- `run.sh` — wrapper around `lm_eval` (supports vLLM and HF backends).
- `analyze_mmlupro.py` / `analyze_mathlvl5.py` — aggregate per-model results
  into a markdown report (overall accuracy + per-sample forgetting/robustness
  analysis).
- `compute_forgetting.py` — generic base-vs-SFT per-sample comparison.

#### Run

```bash
cd lm_harness_eval

# 1. Evaluate each model. Results go under ./results/<bench>/<model_tag>/.
./run.sh --backend vllm --gpu 0 \
    --model Qwen/Qwen3-8B \
    --tasks mmlu_pro --limit 100 \
    --output ./results/mmlupro/qwen3_8b_base

# For merged SFT models, pass the base tokenizer:
./run.sh --backend vllm --gpu 0 \
    --model /path/to/sft_merged_model \
    --tokenizer Qwen/Qwen3-8B \
    --tasks mmlu_pro --limit 100 \
    --output ./results/mmlupro/qwen3_8b_sft_gsm8k

# 2. Aggregate into a markdown report.
python analyze_mmlupro.py   --results-dir ./results/mmlupro   --output mmlupro_analysis.md
python analyze_mathlvl5.py  --results-dir ./results/mathlvl5  --output mathlvl5_analysis.md
```

The analyze scripts expect result subdirectories named
`{qwen3_8b,llama3_8b}_{base,sft_gsm8k,sft_csqa}`; adjust the `families` dict
at the top of each script for other models.

Tasks used in the paper:

| Benchmark    | `--tasks`                | Typical `--limit` |
|--------------|--------------------------|-------------------|
| MMLU-Pro     | `mmlu_pro`               | 100 per subject   |
| Math Level 5 | `leaderboard_math_hard`  | `all` (~1,324)    |

## Citation

If you use this code or our findings in your work, please cite our paper:

```bibtex

```
