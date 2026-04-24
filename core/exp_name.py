"""
core/exp_name.py
================
Parse and compose experiment names such as

    csqa__qwen3_8b__dnunin__nocot
    gsm8k__qwen3_8br__uwdcid__cot  # user weak, doc confident, instruction based_on_docs, reasoning mode

Grammar
-------
<dataset>__<model>__<prior-instruction>__<cot-flag>

* <dataset> := alphanumeric dataset name (e.g., csqa, gsm8k)
* <model> := model key from models_config.yaml, optionally with 'r' suffix for reasoning mode
             e.g., qwen3_8b, qwen3_8br, llama3_8b_instruct
* <prior-instruction> := <prior-token>i<instruction>
  where:
  - <prior-token> := (<chunk><chunk>) where each <chunk> is "d|u" + <tier> + <strength>
                     MUST contain exactly one doc chunk and one user chunk, order matters.
                     d = document prior, u = user prior
                     <tier> = 1 | 2 (1 = bare assertion, 2 = contextual phrasing)
                     <strength> = w | n | c (weak, neutral, confident)
                     e.g. d1nu1n (doc tier1 neutral then user tier1 neutral), 
                          d2cu1w (doc tier2 confident → user tier1 weak)
  - <instruction> := n | d | u | o
                     n = neutral
                     d = answer question prioritize info in the doc
                     u = answer question prioritize info in the user statement
                     o = answer question use model's own internal knowledge
* <cot-flag> := cot | nocot

The parser returns an `ExpName` dataclass containing:

    dataset, model_key, reasoning_mode,
    doc_strength, user_strength, user_first (bool), instruction, use_cot

Round-trip safety: `build_experiment_name(parse_experiment_name(x)) == x`.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Literal
import yaml
from pathlib import Path

# ───────────────────────────────── constants ────────────────────────── #
STRENGTH_CHAR_TO_TEXT = {"w": "weak", "n": "neutral", "c": "confident"}
STRENGTH_TEXT_TO_CHAR = {v: k for k, v in STRENGTH_CHAR_TO_TEXT.items()}

InstrType = Literal["n", "d", "u", "o"]

# Mapping for backward compatibility and clarity
INSTRUCTION_MAP = {
    "n": "neutral",
    "d": "based_on_docs", 
    "u": "based_on_user",
    "o": "own_knowledge_only"
}

# ───────────────────────────────── dataclass ─────────────────────────── #
@dataclass(frozen=True, slots=True)
class ExpName:
    dataset: str
    
    # model
    model_key: str  # The key from models_config.yaml
    reasoning_mode: bool
    
    # priors
    doc_strength: str  # weak | neutral | confident
    user_strength: str
    doc_tier: int      # 1 | 2 (1 = bare assertion, 2 = contextual phrasing)
    user_tier: int     # 1 | 2
    user_first: bool   # True ⇒ user chunk came first
    
    # instruction & CoT
    instruction: InstrType
    use_cot: bool
    
    # original tokens for round-trip/debug
    prior_instruction_token: str
    model_token: str
    
    @property
    def hf_model_id(self) -> str:
        """Get the HuggingFace model ID from models_config.yaml."""
        # Find the config file relative to this module
        config_path = Path(__file__).parent.parent / "configs" / "models_config.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        models = config.get("models", {})
        if self.model_key not in models:
            raise ValueError(f"Model key '{self.model_key}' not found in models_config.yaml")
        
        return models[self.model_key]
    
    @property 
    def full_name(self) -> str:
        """Get the full experiment name."""
        return build_experiment_name(self)
    
    @property
    def model_family(self) -> str:
        """Extract the model family from the model key."""
        # Check for specific patterns
        if self.model_key.startswith("llama3_2"):
            return "llama3.2"
        elif self.model_key.startswith("llama3"):
            return "llama3"
        elif self.model_key.startswith("qwen2_5"):
            return "qwen2.5"
        elif self.model_key.startswith("qwen3"):
            return "qwen3"
        elif self.model_key.startswith("gpt"):
            return "gpt"
        elif self.model_key.startswith("gemma3"):
            return "gemma3"
        else:
            # Default to the first part before underscore if no match
            return self.model_key.split("_")[0]

# ───────────────────────────────── regexes ───────────────────────────── #
_DATASET_RE = re.compile(r"^[a-zA-Z0-9]+$")  # Datasets are alphanumeric
_MODEL_RE = re.compile(r"^[a-zA-Z0-9_]+r?$")  # Model keys can have underscores and optional 'r' suffix
_PRIOR_INSTR_RE = re.compile(r"^(?:[du][12][wnc]){2}i[nduo]$")  # Exactly 2 prior chunks with tier + i + instruction

# ───────────────────────────────── parser ─────────────────────────────── #

def parse_experiment_name(name: str) -> ExpName:
    try:
        dataset, model_tok, prior_instr_tok, cot_tok = name.split("__")
    except ValueError:
        raise ValueError("Experiment name must have exactly 4 '__'-separated parts")
    
    # dataset token
    if not _DATASET_RE.fullmatch(dataset):
        raise ValueError(f"Illegal dataset token {dataset!r}. Must be alphanumeric.")
    
    # model token
    if not _MODEL_RE.fullmatch(model_tok):
        raise ValueError(f"Illegal model token {model_tok!r}")
    
    # Check if reasoning mode is enabled (ends with 'r')
    reasoning_mode = model_tok.endswith('r')
    if reasoning_mode:
        model_key = model_tok[:-1]  # Remove the 'r' suffix
    else:
        model_key = model_tok
    
    # prior-instruction token
    if not _PRIOR_INSTR_RE.fullmatch(prior_instr_tok):
        raise ValueError("Prior-instruction token must match pattern like 'd1nu1nin', 'd2cu1wid' with exactly 2 prior chunks")
    
    # Find the 'i' that separates prior from instruction
    i_pos = prior_instr_tok.rfind('i')
    if i_pos == -1 or i_pos == len(prior_instr_tok) - 1:
        raise ValueError("Prior-instruction token must contain 'i' followed by instruction")
    
    prior_tok = prior_instr_tok[:i_pos]
    instr_tok = prior_instr_tok[i_pos+1:]
    
    # Parse prior token - must be exactly 6 chars (both doc and user with tiers)
    if len(prior_tok) != 6:
        raise ValueError(f"Prior token must be exactly 6 characters (both doc and user with tiers), got {len(prior_tok)}")
    
    chunks = [prior_tok[:3], prior_tok[3:]]
    doc_strength = user_strength = None
    doc_tier = user_tier = None
    user_first = chunks[0][0] == "u"
    
    for ch in chunks:
        kind, tier_char, strength_char = ch[0], ch[1], ch[2]
        
        # Validate tier
        if tier_char not in ['1', '2']:
            raise ValueError(f"Invalid tier character: {tier_char}. Must be 1 or 2")
        tier = int(tier_char)
        
        # Validate strength
        strength_txt = STRENGTH_CHAR_TO_TEXT.get(strength_char)
        if not strength_txt:
            raise ValueError(f"Invalid strength character: {strength_char}")
        
        if kind == "d":
            if doc_strength is not None:
                raise ValueError("Two doc chunks found in prior token")
            doc_strength = strength_txt
            doc_tier = tier
        elif kind == "u":
            if user_strength is not None:
                raise ValueError("Two user chunks found in prior token")
            user_strength = strength_txt
            user_tier = tier
        else:
            raise ValueError(f"Invalid source type: {kind}")
    
    if doc_strength is None or user_strength is None:
        raise ValueError("Prior token must contain exactly one d* and one u* chunk")
    
    # instruction
    if instr_tok not in InstrType.__args__:  # type: ignore[attr-defined]
        raise ValueError(f"Unknown instruction type {instr_tok!r}. Must be one of: n, d, u, o")
    instruction: InstrType = instr_tok
    
    # cot flag
    if cot_tok not in {"cot", "nocot"}:
        raise ValueError("Last token must be 'cot' or 'nocot'")
    use_cot = cot_tok == "cot"
    
    # Enforce constraint: reasoning mode models can only use nocot
    if reasoning_mode and use_cot:
        raise ValueError("Reasoning mode models (ending with 'r') can only be used with 'nocot', not 'cot'")
    
    return ExpName(
        dataset=dataset,
        model_key=model_key,
        reasoning_mode=reasoning_mode,
        doc_strength=doc_strength,
        user_strength=user_strength,
        doc_tier=doc_tier,
        user_tier=user_tier,
        user_first=user_first,
        instruction=instruction,
        use_cot=use_cot,
        prior_instruction_token=prior_instr_tok,
        model_token=model_tok,
    )

# ───────────────────────────── serializer ────────────────────────────── #

def build_experiment_name(e: ExpName) -> str:
    # Build model token
    model_tok = e.model_key
    if e.reasoning_mode:
        model_tok += "r"
    
    # Build prior token - always both doc and user with tiers
    d_chunk = "d" + str(e.doc_tier) + STRENGTH_TEXT_TO_CHAR[e.doc_strength]
    u_chunk = "u" + str(e.user_tier) + STRENGTH_TEXT_TO_CHAR[e.user_strength]
    
    if e.user_first:
        prior_tok = u_chunk + d_chunk
    else:
        prior_tok = d_chunk + u_chunk
    
    prior_instr_tok = prior_tok + "i" + e.instruction
    
    return "__".join([
        e.dataset,
        model_tok,
        prior_instr_tok,
        "cot" if e.use_cot else "nocot",
    ])

# ───────────────────────────── smoke tests ───────────────────────────── #
if __name__ == "__main__":
    samples = [
        "csqa__qwen3_8b__d1nu1nin__nocot",
        "gsm8k__qwen3_8br__u1nd2cid__nocot",  # reasoning mode must use nocot
        "csqa__llama3_8b_instruct__d1wu2ciu__nocot",
    ]
    for s in samples:
        obj = parse_experiment_name(s)
        rebuilt = build_experiment_name(obj)
        assert s == rebuilt, f"Round-trip failed for {s}: got {rebuilt}"
        # Pretty-print fields for manual sanity-check
        from dataclasses import asdict
        import pprint
        print(f"\n{s}:")
        pprint.pprint(asdict(obj))
    
    # --- negative cases should raise -----------------------------------
    bad = [
        "csqa__qwen3_8b__d1nin__nocot",     # doc only - not allowed
        "gsm8k__qwen3_8b__u1niu__cot",      # user only - not allowed
        "csqa__qwen3_8b__d1nu1n__nocot",    # missing instruction
        "csqa__qwen3_8b__d1nu1ninn__nocot", # double instruction
        "csqa__qwen3_8b__d1nd1nu1nin__nocot",   # 3 chunks
        "csqa-foo__qwen3_8b__d1nu1nin__nocot", # dataset with hyphen
        "csqa__qwen3_8b__d1du1nin__nocot",  # two doc chunks
        "csqa__qwen3_8b__u1nu1din__nocot",  # two user chunks
        "csqa__qwen3_8b__d1nu1xin__nocot",  # invalid strength char
        "csqa__qwen3_8b__d1nu1nix__nocot",  # invalid instruction
        "csqa__qwen3_8b__d1nu1nin__badcot", # invalid cot flag
        "csqa__qwen3_8br__d1nu1nin__cot",   # reasoning mode with cot - not allowed
        "csqa__qwen3_8b__d3nu1nin__nocot",  # invalid tier (3)
        "csqa__qwen3_8b__dnunin__nocot",    # old format without tiers
        "toofewparts",
        "too__many__parts__in__this__name",
    ]
    for s in bad:
        try:
            parse_experiment_name(s)
            raise AssertionError(f"Should have failed: {s}")
        except ValueError:
            pass   # expected
    print("\n✓ exp_name parser round-trips & error-checks OK")