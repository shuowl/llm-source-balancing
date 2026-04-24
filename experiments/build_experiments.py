#!/usr/bin/env python3
"""
build_experiments.py
====================

Create a *flat* experiment list from one-or-many "recipe" files and write
it to a YAML file of the form

    experiments:
      - name: csqa__qwen3_8b__d1nu1nin__nocot
      - name: ...

Each recipe is a **YAML mapping** whose keys can be scalars *or* lists.
Legal keys (plural form preferred):

    datasets       –  e.g. csqa   or  [csqa, gsm8k]
    model_tokens   –  e.g. qwen3_8b, qwen3_8br, llama3_8b_instruct
    prior_tokens   –  e.g. d1nu1n or [d1nu1n, u1nd1c, d2cu1w]
                      Format: <chunk><chunk> where each chunk is:
                      - d|u (doc/user) + tier (1|2) + strength (w|n|c)
                      - Must have exactly one doc and one user chunk
                      - Order matters (first chunk comes first in prompt)
    instructions   –  n | d | u | o  (neutral, docs, user, own knowledge)
    cot_flags      –  cot | nocot

All dimensions are cartesian-producted.  Unknown keys are ignored, so you
can add comments or extra fields without breaking the parser.

Examples
--------
```yaml
# experiments/recipes/example.yaml
datasets:       [csqa, gsm8k]
model_tokens:   [qwen3_8b, qwen3_8br, llama3_8b_instruct]
prior_tokens:   [d1nu1n, u1nd1c, d2cu1w]  # doc/user tier+strength combinations
instructions:   [n, d, u, o]               # neutral, based_on_docs, based_on_user, own_knowledge
cot_flags:      [nocot, cot]
```

To specify an output file:
```bash
python experiments/build_experiments.py \
        --recipes experiments/recipes/exp1.yaml \
        --out experiments/generated/exp1_config.yaml
```

If `--out` is omitted, the output path defaults to `experiments/generated/<first_recipe_stem>_config.yaml`.
For example:
```bash
python experiments/build_experiments.py \
        --recipes experiments/recipes/example.yaml
# Default output: experiments/generated/example_config.yaml
```
"""
from __future__ import annotations

import argparse, itertools, pathlib, sys, yaml
from typing import Any, Iterable, List, Dict

# Make utils importable
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.exp_name import parse_experiment_name # validates names

# ─────────────────────── helpers ────────────────────────
def _as_list(v: Any) -> List[Any]:
    """scalar → [scalar]; list/tuple stays unchanged"""
    if v is None:
        return []
    return v if isinstance(v, (list, tuple)) else [v]

def load_recipe(path: pathlib.Path) -> Iterable[str]:
    """Yield experiment-name strings for one recipe file."""
    data = yaml.safe_load(path.read_text()) or {}

    datasets     = _as_list(data.get("datasets")      or data.get("dataset"))
    model_tokens = _as_list(data.get("model_tokens")  or data.get("model_token"))
    prior_tokens = _as_list(data.get("prior_tokens")  or data.get("prior_token"))
    instrs       = _as_list(data.get("instructions")  or data.get("instruction") or "n")
    cot_flags    = _as_list(data.get("cot_flags")     or data.get("cot_flag")    or "nocot")

    if not (datasets and model_tokens and prior_tokens):
        raise SystemExit(f"[ERR] {path}: must specify datasets, model_tokens, prior_tokens")

    for ds, mt, pt, ins, cot in itertools.product(
        datasets, model_tokens, prior_tokens, instrs, cot_flags
    ):
        # Skip invalid combinations: reasoning mode models can only use nocot
        if mt.endswith('r') and cot == 'cot':
            continue
            
        name = f"{ds}__{mt}__{pt}i{ins}__{cot}"
        # quick schema check
        try:
            parse_experiment_name(name)
        except ValueError as e:
            raise SystemExit(f"[ERR] Invalid combination in {path} → {name}\n{e}")
        yield name

# ───────────────────────── CLI ──────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build experiment_config.yaml from recipe files.")
    p.add_argument("--recipes", nargs="+", required=True,
                   help="One or more YAML recipe files.")
    p.add_argument("--out", default=None,
                   help="Path to write the generated config YAML. Defaults to 'experiments/generated/<first_recipe_stem>_config.yaml'.")
    return p.parse_args()

# ───────────────────────── main ─────────────────────────
def main():
    args = cli()

    if args.out is None:
        if not args.recipes:
            # Should not happen due to 'required=True' and 'nargs="+"' for --recipes
            sys.exit("[ERR] No recipes provided to derive default output path.")
        first_recipe_path = pathlib.Path(args.recipes[0])
        default_out_filename = f"{first_recipe_path.stem}_config.yaml"
        # Place the default output in experiments/generated directory
        args.out = pathlib.Path("experiments/generated") / default_out_filename

    names = []
    for rec in args.recipes:
        path = pathlib.Path(rec)
        if not path.exists():
            sys.exit(f"[ERR] Recipe not found: {path}")
        names.extend(load_recipe(path))

    # deduplicate while preserving order
    seen: set[str] = set()
    unique = [n for n in names if not (n in seen or seen.add(n))]

    out_obj: Dict[str, Any] = {"experiments": [{"name": n} for n in unique]}
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(out_obj, sort_keys=False))

    print(f"✓ Wrote {len(unique)} experiments → {out_path}")

if __name__ == "__main__":
    main()
