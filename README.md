# SOLAR — Schema-Only Learning And Rule-extraction with Large Language Models

SOLAR learns and ranks logical rules directly from knowledge graph schemas using Large Language Models, without requiring instance data. The system supports schema-only rule discovery for scenarios like privacy-restricted environments, federated settings, and cold-start domains where instance data is unavailable.

## Key Features

- **Schema-only rule learning**: Generates logical rules from knowledge graph schemas without instance data
- **Multi-agent consensus**: Uses debate between LLM agents to validate and rank rules
- **Model-agnostic**: Supports OpenAI, Anthropic, Google, and local models via Ollama
- **Two ranking approaches**: Consensus debate (SPCA^consensus) and independent evaluation pool (SPCA^pool)
- **Few-shot support**: Optional injection of structural examples to guide generation
- **Comprehensive evaluation**: Schema-level metrics (SCS, SeCS, CovS) and standard link prediction metrics
- **AMIE integration (optional)**: Seed AMIE’s search with SOLAR rules via a custom MiningAssistant (see `amie_integration/PIPELINE.md`)

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

For local models, ensure Ollama is installed and running.

## Quick Start

### Basic Consensus Mode
```bash
python main_SOLAR.py \
  --dataset yago \
  --schema_type line \
  --spca_mode consensus \
  --coordinator_model ollama_qwen2.5:latest
```

### Basic Pool Mode
```bash
# Phase 1: Generate candidates
python main_SOLAR.py \
  --dataset yago \
  --schema_type line \
  --spca_mode pool \
  --coordinator_model ollama_qwen2.5:latest

# Phase 2: Evaluate and aggregate
python -m quality_llm.pool.pool_llm_quality_full \
  gen_rules/yago/c2r_new/line/2/pool_ollama_qwen2.5:latest \
  --dataset-root dataset \
  --agent-models model1 model2 model3 \
  --out aggregated_pool.csv
```

## Core Architecture

### Main Components

- **`main_SOLAR.py`**: Main orchestrator for rule generation and SPCA ranking
- **`llm/call_models.py`**: Model wrappers for different LLM providers
- **`llm/consensus_llms.py`**: Multi-agent consensus coordination
- **`quality_llm/`**: Evaluation tools and metrics
- **`amie_integration/`**: Exporters, assistant, and runners to bias AMIE with SOLAR seeds

### Two Ranking Modes

#### SPCA^consensus (Multi-agent Debate)
A coordinator model generates candidate rules, then multiple agents debate their validity through structured rounds. The coordinator synthesizes a final consensus ranking.

**Features:**
- Full traceability of debate process
- Per-agent voting (-1/0/+1) on each rule
- Consensus-driven final ranking

#### SPCA^pool (Independent Evaluation)
A coordinator generates candidates, then independent evaluator models rank them separately. Rankings are aggregated using Borda counting.

**Features:**
- Phase 1: Candidate generation
- Phase 2: Independent evaluation and Borda aggregation
- Parallel evaluation across multiple models

## Detailed Usage

### Command Line Options

- `--dataset`: Target dataset (yago, family, etc.)
- `--schema_type`: Schema encoding type (line, domain_range, etc.)
- `--spca_mode`: Ranking mode (consensus or pool)
- `--coordinator_model`: Model for coordination/generation
- `--prompt_type`: Prompt style (c2r_new, fs for few-shot)
- `--numex`: Number of examples for few-shot mode
- `--dry_run`: Preview prompts without execution

### Few-shot Mode

Enable few-shot learning by providing structural examples:

```bash
python main_SOLAR.py \
  --dataset family \
  --schema_type line \
  --spca_mode consensus \
  --coordinator_model ollama_qwen2.5:latest \
  --prompt_type fs \
  --numex 3
```

Examples are drawn from `sampled_path/<dataset>/closed_rel_paths.jsonl` and converted into rule-shaped patterns.

### Model Integration

Add new models in `llm/call_models.py`. The system supports:

- **OpenAI**: GPT-3.5, GPT-4, etc.
- **Anthropic**: Claude models
- **Google**: Gemini models  
- **Local**: Any model via Ollama

## Output Structure

Generated rules and metadata are organized hierarchically:

```
gen_rules/
├── dataset_name/
│   ├── prompt_type/
│   │   ├── schema_type/
│   │   │   ├── numbodyatoms/
│   │   │   │   └── model_name/
│   │   │   │       ├── predicate.txt              # Generated rules
│   │   │   │       ├── predicate.query            # Input prompts
│   │   │   │       ├── Coordinator_<model>/       # Coordinator artifacts
│   │   │   │       ├── AgentN_<model>/            # Agent responses
│   │   │   │       ├── experiment_config.json     # Run configuration
│   │   │   │       └── predicate_timings.json     # Performance metrics
```

### Consensus Output
- `<predicate>_consensus.txt`: Final consensus rules
- `Coordinator_<model>/`: Prompts, outputs, votes
- `AgentN_<model>/`: Per-round agent responses and votes

### Pool Output
- `<predicate>_candidates.rules`: Generated candidates
- `Agent*/`: Individual agent rankings
- `aggregated_pool.csv`: Borda-aggregated final ranking

## Evaluation Tools

The `quality_llm` package provides comprehensive evaluation capabilities.

### Consensus Evaluation

```bash
python -m quality_llm.consensus.consensus_llm_quality_full <consensus_dir>
```

**Metrics computed:**
- **SCS (Schema Consistency)**: Adherence to schema constraints
- **SeCS (Semantic Coherence)**: Semantic plausibility using sentence embeddings
- **CovS (Coverage)**: Schema coverage breadth

**Multi-model evaluation:**
```bash
python -m quality_llm.consensus.consensus_llm_quality_full <consensus_dir> \
  --semantic \
  --sim-models model1 model2 model3 \
  --multi-out consensus_multi_models.csv
```

### Pool Evaluation

```bash
python -m quality_llm.pool.pool_llm_quality_full <pool_dir> \
  --dataset-root dataset \
  --agent-models model1 model2 model3 \
  --out aggregated_pool.csv
```

**Process:**
1. Present candidates as numbered lists to evaluators
2. Collect ranking preferences from each agent
3. Normalize outputs to full permutations
4. Apply Borda counting for aggregation

### Vote Analysis

Extract and analyze consensus votes:

```bash
python -m quality_llm.consensus.run_consensus_votes \
  gen_rules/<ds>/<prompt>/<schema>/<k>/consensus_<coord>_N
```

**Output:**
- `votes_raw.csv`: Per-participant, per-rule votes
- `votes_agg.csv`: Aggregated votes with SPCA_consensus scores

## AMIE Integration

Bias AMIE’s search with SOLAR seeds via a custom MiningAssistant.

Full guide: `amie_integration/PIPELINE.md`.

Quick steps
- Export seeds (2-column TSV: `head\tbody`):
  ```bash
  python amie_integration/export_solar_seeds.py \
    --solar-run gen_rules/<ds>/<prompt>/<schema>/<k>/consensus_<coord>_N
  ```
  Output: `amie_integration/out/<dataset>_amie_seeds.tsv`

- Build assistant (choose one):
  - Maven (includes assistant in AMIE jar):
    ```bash
    cd amie_integration/AMIE && mvn -DskipTests package
    ```
  - Tiny jar (compile assistant only):
    ```bash
    AMIE_JAR=amie_integration/AMIE/amie-dev.jar \
    ./amie_integration/build_solar_assistant.sh
    ```

- Run AMIE with seeds:
  - Python subprocess (recommended):
    ```bash
    export AMIE_JAR=amie_integration/AMIE/target/amie.jar  # or amie_integration/AMIE/amie-dev.jar
    python amie_integration/run_amie_subprocess.py \
      --facts dataset/<ds>/facts.txt \
      --seeds amie_integration/out/<dataset>_amie_seeds.tsv \
      --amie-args -minhc 0.01 -mins 5 -minpca 0.01
    ```
  - Shell wrapper:
    ```bash
    AMIE_ROOT=$(pwd)/amie_integration/AMIE \
    ./amie_integration/run_amie_with_seeds.sh \
      amie_integration/out/<dataset>_amie_seeds.tsv \
      dataset/<ds>/facts.txt
    ```

## Release Instructions

Use this checklist to prepare and publish a clean release of SOLAR with optional AMIE integration.

- Prerequisites
  - Python 3.9+ and `pip`
  - Optional (for AMIE integration): Java JDK (javac/java) and Maven if rebuilding AMIE

- Install dependencies
  - `pip install -r requirements.txt`

- Basic runs (schema-only)
  - Consensus (example):
    - `python main_SOLAR.py --dataset family --schema_type line --spca_mode consensus --coordinator_model ollama_qwen2.5:latest`
  - Pool (phase 1):
    - `python main_SOLAR.py --dataset family --schema_type line --spca_mode pool --coordinator_model ollama_qwen2.5:latest`
  - Few-shot examples (fs):
    - `python main_SOLAR.py --dataset family --schema_type line --spca_mode consensus --coordinator_model ollama_qwen2.5:latest --prompt_type fs --numex 3`

- AMIE integration (external, not vendored)
  - Export seeds (2-column TSV):
    - `python amie_integration/export_solar_seeds.py --solar-run gen_rules/<ds>/<prompt>/<schema>/<k>/consensus_<coord>_N`
    - Output: `amie_integration/out/<dataset>_amie_seeds.tsv`
  - Fetch AMIE locally (kept out of git):
    - `./amie_integration/fetch_amie.sh`
  - Build assistant into AMIE (option A):
    - `cd amie_integration/AMIE && mvn -DskipTests package`
  - Or build a small assistant jar (option B):
    - `AMIE_JAR=amie_integration/AMIE/amie-dev.jar ./amie_integration/build_solar_assistant.sh`
  - Run AMIE with seeds (Python subprocess):
    - `export AMIE_JAR=amie_integration/AMIE/target/amie.jar`
    - `python amie_integration/run_amie_subprocess.py --facts dataset/<ds>/facts.txt --seeds amie_integration/out/<dataset>_amie_seeds.tsv --amie-args -minhc 0.01 -mins 5 -minpca 0.01`

- Packaging notes
  - The AMIE repository is not included in this repo. The helper `amie_integration/fetch_amie.sh` clones it locally (ignored by git).
  - Do not commit large datasets or generated artifacts. The repo ignores `amie_integration/AMIE/` and build outputs.
  - Provide a short “Getting Started” in your release description with the commands above.

## Advanced Features

### Sub-schema Extraction

SOLAR uses bounded BFS to extract relevant sub-schemas around head predicates, improving efficiency:

- Polynomial complexity in schema size
- Type-compatible path discovery
- Configurable radius parameter `k`

### RAG Integration

For domain-specific knowledge, SOLAR supports RAG-style background document integration:

- Local embedding indices over domain corpora
- Relevant snippet retrieval during rule generation
- Enhanced performance on niche domains

### Multilingual Support

SOLAR supports multilingual rule generation:

- Schema translation capabilities
- Cross-lingual semantic consistency evaluation
- Preserved rule quality across languages

## Prompts and Templates

Prompts are centrally managed in `prompt/`:

- `spca_pool_prompt.txt`: Pool evaluation prompt
- `consensus_user_instruction.txt`: Consensus coordination
- `fs.txt`: Few-shot template with examples section

## Path Resolution

The system automatically detects repository root and resolves relative paths. Override with:

```bash
export REPO_ROOT=/path/to/SOLAR
```

## Troubleshooting

**Common issues:**

1. **Missing API keys**: Ensure all required keys are set in `llm/call_models.py`
2. **Path errors**: Verify repository structure and REPO_ROOT if needed
3. **Model timeouts**: Adjust timeout settings for slower local models
4. **Memory issues**: Use sub-schema extraction for large schemas

## Contributing

When adding new models:
1. Add wrapper function in `llm/call_models.py`
2. Update model lists in quality evaluation tools
3. Test with both consensus and pool modes
