# pddl_cfg_ollama_openai.py
from pathlib import Path
from outlines.types import CFG
from outlines import from_ollama
import ollama

# # TypeError: CFG-based structured outputs are not available with OpenAI. Use an open source model or dottxt instead.
# from outlines import from_openai
# from openai import OpenAI
# # 1) OpenAI-compatible client pointing to Ollama
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# # 2) Wrap it with Outlines (so we can pass a CFG)
# model = from_openai(client, "qwen3")

# 1) OpenAI-compatible client pointing to Ollama
client = ollama.Client()
# 2) Wrap it with Outlines (so we can pass a CFG)
model = from_ollama(client, "qwen3")

# 3) Load the CFG
dom_cfg = open("pddl_dom.lark").read()
prb_cfg = open("pddl_prob.lark").read()

DOMAIN_CFG  = CFG(dom_cfg)
PROBLEM_CFG = CFG(prb_cfg)

# 4) Prompts (raw PDDL only; no markdown, no JSON)
task = (
    "Blocks A B C D are clear and on the table. Hand is empty. "
    "Goal: (and (on D C) (on C B) (on B A))."
)

domain_prompt = (
    "Output ONLY a valid PDDL DOMAIN s-expression for a typed blocks world with a hand. "
    "Use :requirements :strips :typing :negation :equality. "
    "Types: block hand table. Predicates: on, on-table, clear, holding, handempty. "
    "Actions: pickup, putdown, stack, unstack (typed, consistent arities). "
    "Domain name: blocksworld-hand."
)

problem_prompt = (
    "Output ONLY a valid PDDL PROBLEM s-expression. "
    "Problem name: bw-stack-d-on-c-on-b-on-a. Domain: blocksworld-hand. "
    "Objects: A B C D - block, h - hand, t - table. "
    "Init: A B C D on table and clear; handempty h. "
    f"Goal: (and (on D C) (on C B) (on B A)). Task: {task}"
)

# 5) Generate under CFG constraints (pure PDDL out)
domain_pddl  = model(domain_prompt,  DOMAIN_CFG,  max_tokens=800).strip()
problem_pddl = model(problem_prompt, PROBLEM_CFG, max_tokens=400).strip()

print(";; ---- DOMAIN ----")
print(domain_pddl)
print("\n;; ---- PROBLEM ----")
print(problem_pddl)
