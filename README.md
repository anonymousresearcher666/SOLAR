<meta name="robots" content="noindex">
# SOLAR

A Knowledge Graph Rule Learning system that generates logical rules from knowledge graph schemas using Large Language Models (LLMs).

## Overview

This project provides tools for:
- Extracting subschemas from knowledge graphs
- Generating logical rules using various LLM models  
- Consensus-based rule evaluation across multiple LLMs
- Schema-aware rule generation with type checking

## Core Components

### Main Script (`main_SOLAR.py`)
The primary entry point for rule generation that:
- Loads datasets and schemas from specified paths
- Supports multiple prompt types and schema formats
- Generates rules using various LLM models in parallel
- Saves generated rules and queries to organized directories

**Supported Models:**
- OpenAI: `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- Anthropic: `claude-3-5-sonnet-20240620`, `claude-3-opus-20240229`
- Google: `gemini-1.5-flash`, `gemini-1.5-pro`
- Ollama: `llama3`, `deepseek-r1`, `qwen2.5`, `gemma3`

### Schema Rule Generator (`SchemaRuleGenerator.py`)
Advanced rule generation with schema awareness:
- Extracts subschemas centered on target predicates
- Generates rules with proper variable chaining
- Performs type checking using domain/range information
- Supports different rule formatting options

### Data Handling (`data.py`)
Dataset management utilities:
- Reads various schema formats (domain_range, graph, line)
- Loads predicates and prompt templates
- Parses schema nodes and edges
- Extracts subschemas for focused rule generation

### LLM Integration (`llm/`)

#### Model Interface (`call_models.py`)
- Unified interface for multiple LLM providers
- Token counting and truncation handling
- Timeout configuration for reliable API calls
- Support for local (Ollama) and cloud-based models

#### Consensus System (`consensus_llms.py`)
Multi-agent rule evaluation system:
- Simulates debates between multiple LLM agents
- Coordinator-based consensus building
- Ranking rules by plausibility and confidence
- Configurable discussion rounds

### Utilities (`utils.py`)
Ranking and evaluation functions for knowledge graph completion tasks.

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

## Usage

### Basic Rule Generation
```bash
python main_SOLAR.py --dataset yago --prompt_type c2r_new --model_name gpt-4o
```

### Parameters
- `--dataset`: Dataset name (yago, dbpedia, family, etc.)
- `--prompt_type`: Prompt strategy (c2r_new, base, cot, etc.)
- `--schema_type`: Schema format (graph, line, domain_range)
- `--model_name`: LLM model to use
- `--numbodyatoms`: Number of atoms in rule body (default: 2)
- `-m`: Number of rules to generate (default: 10)
- `--dry_run`: Generate queries without calling LLM

### Schema-Aware Generation
```python
from SchemaRuleGenerator import RuleGenerator

generator = RuleGenerator("path/to/domain_and_range.txt")
subgraph = generator.extract_subschema(schema, "target_predicate", max_length=2)
rules = generator.generate_rules(subgraph, "target_predicate", path_length=2)
```

### Consensus Evaluation
```python
from llm.consensus_llms import simulate_consensus_discussion

consensus = simulate_consensus_discussion(
    coordinator_llm=coordinator,
    agents=agent_dict,
    rules=rule_string,
    schema=schema_string,
    rounds=3
)
```

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

## Dependencies

Key requirements:
- `torch>=2.4.1` - Deep learning framework
- `transformers>=4.44.1` - Hugging Face transformers
- `llama-index-core>=0.11.10` - LLM framework
- `openai>=1.46.0` - OpenAI API
- `anthropic>=0.28.1` - Anthropic API
- `tiktoken>=0.7.0` - Token counting
- `tqdm>=4.66.1` - Progress bars

Requirements

Python 3.8+

Lllamaindex (to query multiple LLM models with the same interface)

OpenAI API key (for GPT models)

Ollama (for open source models)

Google API key (for Gemini models)

Anthropic API key (for Claude models)

--
Configuration

Set up API keys in environment variables:

export OPENAI_API_KEY="your-key"

export GOOGLE_API_KEY="your-key"

export ANTHROPIC_API_KEY="your-key"
--
Install Ollama and required models:

curl https://ollama.ai/install.sh | sh

ollama pull llama3.2:latest

ollama pull deepseek-r1:70b

ollama pull qwen2.5

