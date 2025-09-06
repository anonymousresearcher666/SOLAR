<meta name="robots" content="noindex">
# SOLAR — Schema-Only Learning And Rule-extraction with Large Language Models

SOLAR learns and ranks logical rules directly from knowledge‑graph schemas using LLMs. It supports two complementary ranking paths: a multi‑agent consensus debate and an independent evaluator pool aggregated via Borda. SOLAR is model‑agnostic — any LLM wired into `llm/call_models.py` can be used (OpenAI, Anthropic, Google, or local via Ollama).

## Core Components

### Main Entry (`main_SOLAR.py`)
Generation and SPCA ranking orchestrator:
- Generates rules via a coordinator LLM
- Runs SPCA^{consensus} (debate) or SPCA^{pool} (independent ranking)

### LLM Integration (`llm/`)
- `call_models.py`: model wrappers; add your model here to use it everywhere (OpenAI, Anthropic, Google, Ollama).
- `consensus_llms.py`: consensus coordinator/agents scaffolding.

## Usage

### SOLAR Modes: Consensus vs Pool

- SPCA^{consensus} (multi‑agent consensus)
  - A coordinator model generates rules, then multiple agents debate and the coordinator summarizes a consensus.
  - Run: `python main_SOLAR.py --dataset yago --schema_type line --spca_mode consensus --coordinator_model ollama_qwen2.5:latest`
  - Outputs: `gen_rules/<dataset>/<prompt>/<schema>/<k>/consensus_<coordinator>`

- SPCA^{pool} (independent evaluators + Borda)
  - Phase 1 (generation): a SINGLE coordinator generates the candidate set R.
    - `python main_SOLAR.py --dataset yago --schema_type line --spca_mode pool --coordinator_model ollama_qwen2.5:latest`
  - Phase 2 (evaluation + aggregation): a pool of LLMs ranks R; Borda aggregates.
    - `python -m quality_llm.pool.pool_llm_quality_full gen_rules/<ds>/<prompt>/<schema>/<k>/pool_<coord> --dataset-root dataset --agent-models <m1> <m2> ... --out aggregated_pool.csv`
  - Layout:
    - `<pool_dir>/Coordinator_<coord>/*_candidates.rules`, `*_schema.txt`
    - `<pool_dir>/Agent*/*_pool.query`, `<pool_dir>/Agent*/*_pool.txt`
    - `<pool_dir>/aggregated_pool.csv`


## Output

Generated rules are saved in structured directories:
```
gen_rules/
├── dataset_name/
│   ├── prompt_type/
│   │   ├── schema_type/
│   │   │   ├── numbodyatoms/
│   │   │   │   └── model_name/
│   │   │   │       ├── predicate.txt (generated rules)
│   │   │   │       └── predicate.query (input prompt)
```

## Quality (quality_llm)

The `quality_llm` package contains evaluation tools for both SOLAR modes. Paths are resolved relative to the repo root; prompts are read from `prompt/`.

Consensus (one‑shot)
- Entrypoint: `python -m quality_llm.consensus.consensus_llm_quality_full <consensus_dir>`
- Input: `gen_rules/<ds>/<prompt>/<schema>/<k>/consensus_<coord>/` (contains `<predicate>_consensus.txt`)
- Actions:
  - Extracts rules into JSON (`consensus_rules.json`).
  - Computes SCS (Schema Consistency), SeCS (Semantic Coherence), CovS (Coverage) per predicate.
  - Writes CSV (`--csv-out`, default `consensus_quality.csv`).
- Multi‑model SeCS:
  - `--semantic --sim-models <m1> <m2> ... --multi-out consensus_multi_models.csv`
  - If `--sim-models` omitted, uses a version‑aware recommended set from `get_default_similarity_models()`.

Pool (two‑phase)
- Phase 1 is generation done by `main_SOLAR.py --spca_mode pool`.
- Phase 2 Entrypoint: `python -m quality_llm.pool.pool_llm_quality_full <pool_dir> --dataset-root dataset --agent-models <m1> <m2> ... --out aggregated_pool.csv`
- Input structure expected under `<pool_dir>`:
  - `Coordinator_<coord>/<predicate>_candidates.rules` and `<predicate>_schema.txt` (produced in Phase 1)
  - The runner creates per‑agent folders: `Agent*/*_pool.query` and `Agent*/*_pool.txt`
- Evaluator prompt and normalization:
  - Candidates are shown as a numbered list `1) <rule>`, `2) <rule>`, ...
  - Each evaluator returns ONLY the ranking of indices (e.g., `3 1 2` or `[3,1,2]`).
  - Outputs are normalized and completed to a full permutation of the candidate set.
- Aggregation:
  - Borda scoring combines independent rankings; writes `<pool_dir>/aggregated_pool.csv`.
  - Cleaning maps agent output back to canonical candidate rules; spurious lines are ignored.

Shared utilities
- `quality_llm/common/SchemaQualityMeasures_full.py`:
  - `SchemaLevelEvaluator`: SCS/SeCS/CovS (single model) and rule parsing.
  - `evaluate_rules_multi_models(schema, rules, target, models)`: multi‑model SeCS with shared SCS/CovS.
  - `get_default_similarity_models()`: version‑aware recommended SentenceTransformer list.
- `quality_llm/common/RuleExtractor.py`: robust parsing of rules from text files.

Running from subdirectories
- Tools auto‑detect repo root and accept absolute or relative paths.
- If you run from `quality_llm/` subfolders, imports fall back to sibling/package paths automatically.

Path tips:
- Works as module or script; resolves relative paths against repo root.
- Prompts live under `<repo_root>/prompt/` (not per dataset).
  - Pool: `prompt/spca_pool_prompt.txt`
  - Consensus: `prompt/consensus_prompt.txt`, `prompt/consensus_user_instruction.txt`
- Override repo root via `REPO_ROOT=/path/to/SOLAR` if needed.


## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set API keys in `llm/call_models.py`:
```python
os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["ANTHROPIC_API_KEY"] = "your_key" 
os.environ["GEMINI_API_KEY"] = "your_key"
```
