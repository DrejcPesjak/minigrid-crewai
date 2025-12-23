# Sense-Plan-Code-Act (SPCA) System

An LLM-powered autonomous agent that solves OpenAI Gymnasium MiniGrid/BabyAI tasks through a closed-loop architecture combining classical AI planning (PDDL) with LLM-generated code. The system progressively learns new capabilities by generating Python code for new high-level actions as it encounters new challenge categories.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [The SPCA Loop](#the-spca-loop)
3. [Environment & Perception](#environment--perception)
4. [Planning Subsystem](#planning-subsystem)
5. [Code Generation Subsystem](#code-generation-subsystem)
6. [Agent & Action Library](#agent--action-library)
7. [Curriculum & Task Progression](#curriculum--task-progression)
8. [Error Recovery & Repair](#error-recovery--repair)
9. [Metrics & Logging](#metrics--logging)
10. [LLM Client Wrapper](#llm-client-wrapper)

---

## System Overview

The SPCA system solves grid-world navigation and manipulation tasks by combining:

| Component | Responsibility |
|-----------|----------------|
| **Sense** | Capture current state snapshot (partial SLAM map, mission, inventory, orientation) |
| **Plan** | Generate PDDL domain/problem and compute a symbolic plan using classical planners |
| **Code** | Generate or patch Python methods to implement high-level actions |
| **Act** | Execute the plan as primitive actions in the simulator |

The key innovation is that the **agent's action codebase grows over time**—the Coder LLM writes new Python methods whenever the Planner introduces actions that don't yet exist in the Agent class.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MAIN CONTROL LOOP                                  │
│                                                                              │
│  ┌──────────┐   ┌──────────┐   ┌────────────────────────────────────────┐    │
│  │  SENSE   │ → │   PLAN   │ → │         INNER RETRY LOOP               │    │
│  │ snapshot │   │ PDDL+UP  │   │  ┌────────┐   ┌─────────────────────┐  │    │
│  └──────────┘   └──────────┘   │  │  CODE  │ → │        ACT          │  │    │
│       ↑                        │  │  LLM   │   │  execute in env     │  │    │
│       │                        │  └────────┘   └─────────────────────┘  │    │
│       │                        │       ↑              │                 │    │
│       │                        │       └── on failure ┘                 │    │
│       │                        │           (rollback + re-code,         │    │
│       │                        │            up to 5 semantic retries)   │    │
│       │                        └────────────────────────────────────────┘    │
│       │                                       │                              │
│       └─── if semantic retries exhausted ─────┘                              │
│            (new SPCA round, up to 5)                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The SPCA Loop

The main loop (`main.py`) orchestrates the four phases for each task in the curriculum:

### Phase Breakdown

1. **SENSE**: Capture a snapshot of the current environment state including mission, agent orientation, inventory, and the **accumulated SLAM map** (which may contain `"unseen"` cells for unexplored areas).

2. **PLAN**: The `PlannerLLM` generates a PDDL domain and problem, then uses **Unified Planning** (a classical AI planner) to find a valid action sequence. If the PDDL has syntax errors, it self-repairs via LLM feedback.

3. **CODE**: For any actions in the plan that the Agent class lacks, `CoderLLM` generates the missing Python methods. It uses AST-based merging to inject new methods into the existing Agent.

4. **ACT**: The plan string (e.g., `[pick_up_obj(a1, key_blue), open_door(a1, door_blue)]`) is executed step-by-step. Each high-level action is a generator yielding primitive codes (0–6).

### Two-Level Retry Structure

**Inner Loop (same plan):** When execution fails, the system first attempts to fix the code:
1. Rollback environment to pre-execution checkpoint
2. Re-invoke Coder with error trace for **all** high-level actions in the plan
3. Retry execution with patched code
4. Repeat up to `MAX_SEMANTIC_RETRY = 5` times

**Outer Loop (new plan):** Only when the inner loop is exhausted:
1. Mark `plan_failed = True`
2. Start a new SPCA round (SENSE → PLAN → CODE → ACT)
3. Planner receives the failure signal and generates new/different PDDL
4. Repeat up to `MAX_SPA_ROUNDS = 5` times

| Condition | Recovery |
|-----------|----------|
| **Syntax/merge error** in Coder | Coder self-repairs with error feedback (up to 5 internal rounds) |
| **Execution failure** (runtime crash, stuck, wrong outcome) | Inner loop: rollback + re-code + retry (same plan) |
| **Inner retries exhausted** | Outer loop: new SPCA round with replanning |
| **All SPCA rounds exhausted** | Mark level as failed, proceed to next config |

---

## Environment & Perception

The `MiniGridEnv` class (`minigridenv.py`) wraps the Gymnasium environment and provides:

### Incremental Map Building via SLAM

MiniGrid environments only provide a **7×7 egocentric view** centered on the agent. The wrapper maintains an **accumulated exploration map** (`agent.full_grid`) that grows as the agent moves:

- **Not the true environment grid** — only contains what the agent has observed so far
- Unexplored regions are marked as `"unseen"`
- After each primitive action, `SLAM()` integrates the new local view into the accumulated map
- The grid is padded automatically when the agent moves beyond known boundaries
- Cells contain space-separated strings: `"door red closed"`, `"key blue"`, `"wall"`, etc.

**Important:** The Planner and Coder receive this partial map (named `visible_grid` in the snapshot), which may have large `"unseen"` regions. High-level actions must handle exploration (e.g., `reach_goal` includes frontier exploration logic to find objects not yet visible).

**When SLAM updates occur:** The map is updated during the **ACT phase**, not SENSE. Each primitive action (move, turn, etc.) triggers `SLAM()` which integrates the new 7×7 view. The SENSE phase merely captures the current accumulated state.

### Observation Conversion

Raw observations are 3D arrays of `(object_idx, color_idx, state_idx)`. The `convert_observation()` method translates these into human-readable strings and rotates the view based on agent orientation.

### Checkpoint & Replay

Every primitive action code is recorded. On semantic failure:
1. The environment resets to initial state
2. All recorded primitives are replayed rapidly
3. The agent resumes from the exact pre-failure state

This enables the Coder to patch code and re-attempt without losing progress.

### Outcome Classification

Plan execution returns an `Outcome` with status:

| Status | Meaning |
|--------|---------|
| `success` | Goal reached with positive reward |
| `goal_not_reached` | Plan completed but mission unsatisfied |
| `stuck` | No grid changes for N steps (agent looping) |
| `runtime_error` | Exception during action execution |
| `syntax_error` | Method raised an error |
| `missing_method` | Plan references non-existent Agent method |
| `reward_failed` | Episode terminated with zero/negative reward |

---

## Planning Subsystem

The `PlannerLLM` class (`plannerLLM.py`) is responsible for symbolic planning:

### PDDL Generation

The LLM receives:
- Current snapshot containing:
  - `mission` — Task description (e.g., "pick up the blue key")
  - `direction` — Agent orientation
  - `inventory` — Currently held object
  - `visible_grid` — The **accumulated SLAM map** (may contain `"unseen"` cells)
  - `visible_objects` — Objects in the current 7×7 view
- Category metadata (skill description, level info)
- Current Agent code (method signatures + docstrings)

It outputs a JSON with:
```json
{
  "domain": "(define (domain ...) ...)",
  "problem": "(define (problem ...) ...)"
}
```

**Note:** The planner sees partial information. It must generate plans that account for the fact that targets may be in unexplored regions, delegating exploration to the high-level action implementations.

### Classical Planning

The generated PDDL is fed to **Unified Planning** (`unified_planning`), which:
1. Parses domain + problem files
2. Runs a one-shot planner (auto-selected by problem kind)
3. Validates the plan with the `tamer` validator

The resulting plan is converted to a function call string:
```
[pick_up_obj(a1, key_blue), nav_to(a1, door_blue), done(a1, door_blue)]
```

### Model Selection Strategy

| Scenario | Model Used |
|----------|------------|
| Fresh planning (no prior PDDL) | `o3` (large model) |
| Reusing trusted PDDL from prior success | `o4-mini` (small model) |
| Replanning after plan failure | `o3` (large model) |
| Syntax repair after PDDL parse error | `o3` (large model) |

### Repair Loop

If the PDDL fails to parse or the planner returns `UNSOLVABLE`, the LLM receives the error log and must produce corrected PDDL. This repeats up to `MAX_RETRIES = 8`.

### Action Schema Caching

After a successful plan, the planner extracts `(:action ...)` blocks from the domain. These schemas are passed to the Coder so it knows the PDDL semantics (parameters, preconditions, effects) for each action it must implement.

---

## Code Generation Subsystem

The `CoderLLM` class (`coderLLM.py`) generates Python implementations for high-level actions:

### Usage

When the plan contains actions like `reach_goal(a1, key_blue)` that the Agent class lacks, the main loop calls:
```python
coder.implement_actions(
    actions       = {"reach_goal"},
    pddl_schemas  = {"reach_goal": "(:action reach_goal :parameters ...)"},
    plan_str      = "[reach_goal(a1, key_blue), ...]",
    agent_state   = {...current grid, inventory, etc...},
    past_error_log= None  # or error from prior attempt
)
```

### Prompt Structure

The Coder LLM receives:
1. **System prompt**: Rules for writing Agent methods (yield primitives, read-only grid, naming conventions)
2. **Current Agent code**: The full `agent_tmp.py` source
3. **Agent state**: Grid snapshot for context
4. **PDDL schemas**: So the method signature matches parameter count/order
5. **Plan string**: To understand usage context

### AST-Based Merging

The LLM outputs raw Python `def` blocks (no class wrapper). The coder:
1. Parses the patch with `ast.parse()`
2. Locates the `Agent` class in the base AST
3. Replaces existing methods or appends new ones
4. Merges any new imports
5. Writes back via `ast.unparse()`

### Syntax Repair

If the merged code fails to reload (`importlib.reload(agent_tmp)`), the error + traceback is fed back to the LLM for correction. Up to `MAX_ROUNDS = 5` repair attempts.

---

## Agent & Action Library

The `Agent` class (`agent.py`) starts minimal and grows over the curriculum:

### Primitive Actions (Codes 0–6)

| Code | Action | Method |
|------|--------|--------|
| 0 | Turn left | `turn_left()` |
| 1 | Turn right | `turn_right()` |
| 2 | Move forward | `move_forward()` |
| 3 | Pick up | `pick_up()` |
| 4 | Drop | `drop()` |
| 5 | Toggle (open/close) | `toggle()` |
| 6 | Done | `done()` |

### High-Level Actions (LLM-Generated)

These are generators that `yield` or `yield from` primitive codes:

```python
def reach_goal(self, _a: str, g: str):
    """Navigate to target object g, exploring as needed."""
    # BFS pathfinding, door handling, frontier exploration...
    yield from self._navigate_to_cell(target)
    yield from self._face_direction(dr, dc)
```

Examples from the evolved Agent:
- `pick_up_obj(_a, _k)` — Navigate to and pick up an object
- `nav_to(_a, _d)` — Navigate to a door, open if closed
- `reach_target(_a, _d)` — Find and approach a door
- `cross_stream(_a, _from, _to)` — Traverse stepping stones over lava
- `complete_memory_task(_ag)` — Solve the memory matching puzzle

### Helper Methods

The Agent accumulates reusable utilities:
- `_agent_coords()` — Find agent position in grid
- `_front_coords()` — Get cell in front of agent
- `_cell_passable(r, c)` — Check if cell is walkable
- `_navigate_to_cell(dest)` — BFS pathfinding + execution
- `_face_direction(dr, dc)` — Rotate to face a direction
- `_find_frontier()` — Find nearest unexplored cell

### State Variables

The environment updates these read-only (from Agent's perspective):
- `mission` — Task description string
- `current_dir` — Orientation: `"East"`, `"South"`, `"West"`, `"North"`
- `current_observation` — Current 7×7 egocentric view (refreshed each step)
- `full_grid` — **Accumulated SLAM map** (numpy array of strings, contains `"unseen"` for unexplored cells)
- `inventory` — Currently held object name or `None`
- `previous_actions` — List of all executed primitive codes

**Note:** `full_grid` is NOT the true environment grid—it only contains cells the agent has observed. Actions like `reach_goal` must implement exploration logic to handle cases where the target is in an `"unseen"` region.

---

## Curriculum & Task Progression

The system trains on a structured curriculum (`merged_curriculum2.json`) organized by skill categories:

### Categories (12 total)

1. **goal_navigation** — Reach green goal squares (Empty, Crossing)
2. **static_obstacle_navigation** — Navigate around objects (GoToLocal, GoToDoor)
3. **hazard_avoidance** — Avoid lava tiles (LavaGap, LavaCrossing, DistShift)
4. **pickup_only** — Pick up specified objects (OneRoom, Fetch, PickupDist)
5. **open_door** — Open unlocked doors (OpenRedDoor, MultiRoom)
6. **memory_ordering** — Execute actions in sequence (RedBlueDoors, MemoryEnv)
7. **unlock_door** — Fetch key and unlock (LockedRoom, KeyInBox)
8. **unlock_pickup** — Unlock door then pick up item (UnlockPickup, KeyCorridor)
9. **obstacle_blocking** — Move blockers, then unlock (BlockedUnlockPickup)
10. **object_placement** — Put object next to another (PutNext, PutNear)
11. **composite_skills** — Multi-step combined tasks (Synth, BossLevel)
12. **dynamic_obstacle_avoidance** — Moving obstacles (DynamicObstacles)

### Progression Logic

1. Each category resets PDDL but **inherits the accumulated Agent code**
2. Within a category, successful PDDL is "trusted" and reused with a smaller model
3. Failed plans trigger replanning with the large model
4. The `--keep-agent` and `--keep-pddl` flags allow resuming mid-curriculum

### Environment Configurations

Each level has multiple configs of increasing difficulty:
```json
{
  "name": "Empty",
  "configs": [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-Random-5x5-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0"
  ]
}
```

---

## Error Recovery & Repair

The system has multiple layers of error handling:

### Coder Syntax Repair

When AST merge or module reload fails:
1. Error message + traceback is appended to conversation
2. LLM produces corrected code
3. Retry up to 5 times

### Semantic Retry with Rollback

When execution fails (stuck, runtime error, wrong outcome):
1. Checkpoint is restored (env reset + replay saved primitives)
2. Error trace is passed to Coder
3. All plan actions are re-implemented
4. Retry up to 5 times per plan

### Plan-Level Retry

When semantic retries are exhausted:
1. Increment SPCA round counter
2. Planner receives `plan_failed=True`
3. New PDDL is generated with "drastic changes" guidance
4. Up to 5 full SPCA rounds per configuration

### Logging on Failure

Each agent version and PDDL pair is saved with `fail_` prefix when the run ends unsuccessfully, enabling post-mortem analysis.

---

## Metrics & Logging

The `MetricsLogger` class (`metrics_logger.py`) provides structured telemetry:

### Per-Level Metrics

```json
{
  "level_name": "BabyAI-Unlock-v0",
  "outcome": "success",
  "elapsed_min": 1.42,
  "counters": {
    "spca_rounds": 1,
    "simulation_runs": 2,
    "planner_calls": 2,
    "coder_calls": 1
  }
}
```

### Counter Categories

| Counter | Description |
|---------|-------------|
| `spca_rounds` | Outer Sense-Plan-Code-Act iterations |
| `simulation_runs` | Total `env.run_sim()` calls |
| `planner_calls` | Total LLM calls for PDDL |
| `planner_fresh/reuse/replan/syntax` | Planning mode breakdown |
| `coder_calls` | Total LLM calls for code |
| `coder_first/semantic/syntax` | Coder invocation breakdown |

### Output Files

Per run, the system generates:
- `metrics.jsonl` — Streaming event log (level starts, counter increments, summaries)
- `summary_<category>_<level>_<run_id>.json` — Final summary per level
- `agent_v{N}.py` — Agent code at each successful level
- `domain_v{N}.pddl` / `problem_v{N}.pddl` — PDDL snapshots
- `prompt_log.md` — All LLM prompt templates used

### Signal Handling

`register_signal_handlers()` ensures that Ctrl-C writes an "aborted" summary rather than losing metrics.

---

## LLM Client Wrapper

The `ChatGPTClient` class (`llmclient.py`) provides a unified interface across:

| Provider | Model Examples |
|----------|----------------|
| OpenAI | `gpt-4o`, `o3`, `o4-mini`, `codex-mini-latest` |
| Ollama (local) | `llama4` |
| Google | `gemini-2.5-flash` |

### Structured Output

Uses Pydantic models for response parsing:
```python
class PDDLResp(BaseModel):
    domain: str
    problem: str

client = ChatGPTClient("openai/o3", PDDLResp)
response = client.chat_completion(messages)  # returns PDDLResp
```

The client automatically routes to the correct API endpoint based on the model prefix.

---

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set API Key

Create a `.env` file in this directory (or export the variable):

```bash
# .env
OPENAI_API_KEY=sk-...
```

Or export directly:
```bash
export OPENAI_API_KEY="sk-..."
```

For local models via Ollama, ensure the server is running on `localhost:11434`.

---

## Running the System

### Basic Execution

```bash
python main.py
```

### Resume from a Category

```bash
python main.py --start-category unlock_door --keep-agent --keep-pddl
```

### Arguments

| Flag | Effect |
|------|--------|
| `--start-category NAME` | Skip to this category (case-insensitive) |
| `--keep-agent` | Don't reset agent.py to initial state |
| `--keep-pddl` | Seed PDDL cache from existing domain/problem files |

---

## Architecture Summary

```
main.py                  # Orchestrates the SPCA loop & curriculum
├── minigridenv.py       # Gym wrapper + SLAM + checkpoint + execution
├── plannerLLM.py        # PDDL generation + Unified Planning + repair
├── coderLLM.py          # Code synthesis + AST merge + syntax repair
├── agent.py             # Growing action library (source of truth)
├── agent_tmp.py         # Hot-reloaded working copy
├── llmclient.py         # LLM API abstraction
├── metrics_logger.py    # Structured telemetry
└── merged_curriculum2.json  # Task progression definition
```

The system demonstrates that **LLM-generated code can be composed with classical AI planning** to achieve zero-shot generalization across diverse grid-world tasks, with the agent's capabilities expanding as it encounters new challenge types.

