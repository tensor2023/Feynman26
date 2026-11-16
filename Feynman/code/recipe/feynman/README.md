# Feynman26 on OdysSim

This task implements the source-grounded behavioral simulation in `how.md`. It does not claim that the policy is Richard
Feynman and does not reward quotation recall. Scientific correctness and source grounding gate the reward; copying,
fabricated quotations, historical leakage, and evaluator manipulation are explicit penalties or hard failures.

## Quick start

```bash
python recipe/feynman/prepare_data.py recipe/feynman/scenarios.example.jsonl data/feynman/all.parquet
# Split real data by complete source/event/template groups, not by passage.
TRAIN_FILES=data/feynman/train.parquet VAL_FILES=data/feynman/val.parquet \
  OPENAI_API_KEY=... bash recipe/feynman/run_rl.sh
```

Generate the catalog-level schema fixture:

```bash
python recipe/feynman/generate_seed_scenarios.py recipe/feynman/scenarios.seed100.jsonl
python recipe/feynman/prepare_data.py recipe/feynman/scenarios.seed100.jsonl data/feynman/seed100.parquet
```

This fixture is synthetic and is **not training data**. It contains four
derivative tasks for each of 25 catalog cards. Keep all
tasks sharing a source ID in the same split. Its evidence is paraphrased at the
catalog-card level; verify each card against the cited primary source before
promoting the pilot to a production training dataset.

## Real 100-row source-grounded dataset

The completed first batch is:

```text
data/feynman/feynman_real100.jsonl
data/feynman/feynman_real100.parquet
```

It was produced without an external API key. The local authenticated Codex CLI
was used for separate generation and verification passes. Cached intermediate
prompts and results are under `data/feynman/work/`.

The reproducible stages are:

```bash
# 1. Extract page-located, hashed source episodes from primary-source files.
python recipe/feynman/extract_source_episodes.py \
  ../../data/files data/feynman/work/source_episodes.jsonl --limit 140

# 2. Build generation prompts. Run each prompt through `codex exec` with
#    codex_generation.schema.json and cache its JSON response.
python recipe/feynman/prepare_codex_batches.py \
  data/feynman/work/source_episodes.jsonl data/feynman/work/prompts

# 3. Join accepted cards back to their private source episodes and prepare an
#    independent verification pass.
python recipe/feynman/prepare_verification_batches.py \
  data/feynman/work/source_episodes.jsonl \
  data/feynman/work/source_episodes_additional.jsonl \
  "data/feynman/work/generation_*.json" \
  "data/feynman/work/additional_generation_*.json" \
  data/feynman/work/verification_prompts

# 4. After the verifier outputs are cached, select the best 100 verified rows
#    and convert the validated JSONL to OdysSim-compatible Parquet.
python recipe/feynman/assemble_verified_dataset.py \
  data/feynman/feynman_real100.jsonl --limit 100
python recipe/feynman/prepare_data.py \
  data/feynman/feynman_real100.jsonl \
  data/feynman/feynman_real100.parquet
```

Generation admitted 125 of 200 candidates. The separate verifier retained 104;
the assembler selected the top 100 using source support, prompt independence,
episode completeness, and copying risk. Each delivered row contains the private
source episode, page and file provenance, source SHA-256, structured reference
card, generation-output hash, verifier scores, and private distinctive phrases.
The policy prompt contains none of the private reference text or phrases.

The runner assumes the official OdysSim/verl 0.7.0 container and GPU stack. `agent_version=copy` generates the original,
feedback-free refined, and feedback-conditioned refined sequences used by FoldGRPO. Set `FEYNMAN_JUDGE_MODEL` and
`FEYNMAN_JUDGE_REASONING` to override the independent structured judge.

## Required scenario fields

Use `scenarios.example.jsonl` as the canonical schema. Evidence records should include source IDs, dates, locations when
available, and clearly marked quotations versus paraphrases. Populate `source_passages` only for deterministic overlap
checking; these passages are not placed in the policy prompt.

For production, keep private evaluation scenarios, use a separate final judge, and report each dimension independently.
