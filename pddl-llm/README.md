## PDDL-LLM: LLM-based PDDL Planning

This module evaluates the ability of Large Language Models to generate valid PDDL (Planning Domain Definition Language) domain and problem files from natural language task descriptions.

### Overview

Given a natural language description of a planning task, the LLM generates:
1. A **PDDL domain file** (actions, predicates, types)
2. A **PDDL problem file** (initial state, goal state)

The generated files are then validated using classical AI planners (via [Unified Planning](https://github.com/aiplan4eu/unified-planning)). If planning fails, the LLM receives feedback and iteratively refines its output.

---

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### API Keys

Set your OpenAI API key in the scripts or via environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

For local models, ensure [Ollama](https://ollama.ai/) is running on `localhost:11434`.

---

## Main Scripts

### 1. `chatgpt_pddl5.py` — Interactive PDDL Generation

Interactively generate PDDL from a task description using a selected LLM model.

```bash
python chatgpt_pddl5.py
```

You will be prompted for:
- **Task name**: A short identifier (e.g., "cupcake")
- **Task description**: Natural language description of the planning problem

The script will:
- Generate domain and problem PDDL files
- Attempt to solve using a classical planner
- If errors occur, iteratively refine with LLM feedback (up to 10 attempts)
- Save logs to `./logs/`

### 2. `test_all_models.py` — Benchmark Multiple Models

Run comprehensive tests across multiple LLM models on classical PDDL benchmark problems.

```bash
python test_all_models.py
```

**Features:**
- Tests models against 11 classical planning domains (blocksworld, gripper, logistics, etc.)
- Tracks solve rates, attempts needed, and plan correctness
- Supports both OpenAI models and local Ollama models
- Saves detailed results to `./test_results/`

**Modify the `MODELS` list** in the script to select which models to test.

### 3. `classical-examples/solve_all.py` — Generate Reference Plans

Solve all classical PDDL problems using traditional planners and save reference plans.

```bash
cd classical-examples
python solve_all.py
```

This generates `plan.txt` files in each problem directory, used as ground truth for benchmarking.

---

## Directory Structure

```
pddl-llm/
├── chatgpt_pddl5.py          # Interactive PDDL generation
├── test_all_models.py        # Multi-model benchmark
├── classical-examples/       # Classical PDDL benchmark problems
│   ├── blocksworld/
│   ├── gripper/
│   ├── logistics/
│   ├── ... (11 domains)
│   └── solve_all.py          # Generate reference plans
├── llm-cfg/                  # Constrained decoding experiments (Lark grammars)
├── logs/                     # Experiment logs
├── test_results/             # Benchmark results and plots
└── requirements.txt
```

---

## Supported Models

**OpenAI (API):**
- gpt-4o, gpt-4o-mini, gpt-4.5-preview
- o1, o1-mini, o3-mini (reasoning models)
- gpt-4.1, o4-mini

**Ollama (Local):**
- llama3.2, qwen3, gemma3, deepseek-r1, etc.

