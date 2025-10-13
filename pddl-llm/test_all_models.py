#!/usr/bin/env python3
"""
Comprehensive test script for all LLM models on classical PDDL problems.
Tests models against ranked problems and tracks detailed statistics.
"""

import time
import os
import json
import logging
import datetime
import signal
import sys
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus
from collections import defaultdict

API_KEY = "sk-proj-1VvkSU..."

# List of models to test
# Format: "provider/model-name" or "provider/model-name*" (star = no structured output support)
# Supported providers: "openai", "ollama"
MODELS = [
    ## comparison
    "openai/gpt-4o", # https://aclanthology.org/2025.naacl-long.560.pdf  # paid
    "openai/o1-mini*", # https://aclanthology.org/2025.naacl-long.560.pdf (no structured output) 
    "openai/gpt-4-0125-preview*",  # https://arxiv.org/pdf/2404.07751 (no structured output)  # paid
    ## frontier
    "openai/gpt-5",
    "openai/gpt-5-mini",
    ## thesis
    "openai/o3", # paid
    "openai/gpt-4.1", # paid
    "openai/o4-mini",
    # ## ollama
    # "ollama/llama3.2",
    # "ollama/granite4:tiny-h", # not working
    # "ollama/phi4-mini-reasoning",
    # "ollama/qwen3",
    # "ollama/gemma3",
    # "ollama/deepseek-r1:8b",

    ## other
    # "openai/o1",
    # "openai/o1-pro",
    # "openai/o3-mini",
    # "openai/gpt-4.5-preview",
    # "openai/gpt-4.1-mini",
    # "ollama/mistral",
    # "ollama/codellama",
]

# Classical problems ranked from easiest to hardest (approximate)
PROBLEMS = [
    "blocksworld",     # Simple stacking
    "gripper",         # Simple transport
    "depot",           # Medium logistics
    "driverlog",       # Medium coordination
    "satellite",       # Medium sequencing
    "rovers",          # Medium exploration
    "tyreworld",       # Medium sequential tasks
    "storage",         # Complex logistics
    "logistics",       # Complex multi-city
    "termes",          # Complex construction
    "floortile",       # Complex coordination
]

# Pydantic model for LLM response
class PDDLResponse(BaseModel):
    domain: str
    problem: str

class ModelTester:
    def __init__(self, model_spec, api_key):
        """
        Initialize model tester.
        
        Args:
            model_spec: Full model specification (e.g., "openai/gpt-4o", "openai/o1-mini*")
                       Star suffix (*) indicates no structured output support
            api_key: API key for OpenAI models (unused for Ollama)
        """
        self.model_spec = model_spec
        
        # Check if model supports structured output (no star at end)
        self.supports_structured_output = not model_spec.endswith("*")
        clean_spec = model_spec.rstrip("*")
        
        # Parse provider and model name
        if "/" in clean_spec:
            self.provider, self.model_name = clean_spec.split("/", 1)
        else:
            # Default to openai if no prefix
            self.provider = "openai"
            self.model_name = clean_spec
        
        # Initialize client based on provider
        if self.provider == "ollama":
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            self.use_ollama = True
        elif self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.use_ollama = False
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'openai' or 'ollama'.")
        
        self.stats = defaultdict(lambda: defaultdict(int))
        self.consecutive_failures = 0
        
    def extract_json_from_text(self, text):
        """
        Extract JSON from text that may contain additional content.
        Looks for JSON object with 'domain' and 'problem' keys.
        """
        import re
        
        # Try to find JSON object in the text
        # Look for patterns like {...}
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                
                # Check if it has the required keys
                if 'domain' in data and 'problem' in data:
                    # Create PDDLResponse object
                    return PDDLResponse(domain=data['domain'], problem=data['problem'])
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to extract by looking for markers
        # Sometimes LLMs wrap JSON in ```json ... ```
        json_code_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        code_match = re.search(json_code_pattern, text, re.DOTALL)
        if code_match:
            try:
                data = json.loads(code_match.group(1))
                if 'domain' in data and 'problem' in data:
                    return PDDLResponse(domain=data['domain'], problem=data['problem'])
            except json.JSONDecodeError:
                pass
        
        raise ValueError("Could not extract valid JSON with 'domain' and 'problem' keys from response")
    
    def get_completion(self, messages):
        """Get completion from the model."""
        # print(f"Messages: {messages}")
        if self.supports_structured_output:
            # Use structured output for compatible models
            result = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=PDDLResponse,
            )
            # print(f"Result: {result}")
            return result.choices[0].message.parsed
        else:
            # Use regular chat completion and parse JSON manually
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            response_text = result.choices[0].message.content
            # print(f"Response text: {response_text}")
            # Extract and parse JSON from the response
            return self.extract_json_from_text(response_text)
    
    def save_pddl_files(self, domain_text, problem_text, output_dir):
        """Save PDDL files to specified directory."""
        domain_path = output_dir / "domain_generated.pddl"
        problem_path = output_dir / "problem_generated.pddl"
        
        with open(domain_path, "w") as f:
            f.write(domain_text)
        with open(problem_path, "w") as f:
            f.write(problem_text)
            
        return domain_path, problem_path
    
    def solve_pddl(self, domain_file, problem_file):
        """Attempt to parse and solve PDDL files."""
        try:
            reader = PDDLReader()
            problem = reader.parse_problem(str(domain_file), str(problem_file))
            
            with OneshotPlanner(problem_kind=problem.kind) as planner:
                result = planner.solve(problem)
            
            if result.status in [PlanGenerationResultStatus.SOLVED_SATISFICING, 
                                PlanGenerationResultStatus.SOLVED_OPTIMALLY]:
                return True, result.plan, None
            else:
                return False, None, f"Plan status: {result.status}"
        except Exception as e:
            return False, None, str(e)
    
    def extract_action_sequence(self, plan):
        """Extract action names from a plan."""
        if plan is None:
            return []
        
        actions = []
        plan_str = str(plan)
        
        # Parse the plan string to extract action names
        for line in plan_str.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract action name (first word before parenthesis or space)
                if '(' in line:
                    action = line.split('(')[0].strip()
                else:
                    action = line.split()[0] if line.split() else ""
                
                if action:
                    actions.append(action.lower())
        
        return actions
    
    def load_reference_plan(self, problem_dir):
        """Load reference plan from plan.txt."""
        plan_file = problem_dir / "plan.txt"
        if not plan_file.exists():
            return []
        
        with open(plan_file, 'r') as f:
            content = f.read()
        
        # Create a dummy plan object and extract actions
        return self.extract_action_sequence(content)
    
    def compare_plans(self, generated_plan, reference_actions):
        """Compare generated plan with reference plan (action sequence only)."""
        if not reference_actions:
            return None  # No reference to compare
        
        generated_actions = self.extract_action_sequence(generated_plan)
        
        # Compare action sequences
        return generated_actions == reference_actions
    
    def test_problem(self, problem_name, problem_dir, output_dir, logger, max_attempts=10):
        """Test a single problem with the model."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {problem_name}")
        logger.info(f"{'='*60}")
        
        # Check if we should skip due to consecutive failures
        if self.consecutive_failures >= 3:
            logger.warning(f"⚠ Skipping {problem_name} due to 3+ consecutive failures")
            return {
                'parsed': False,
                'solved': False,
                'correct': None,
                'attempts': 0,
                'error': 'Skipped due to consecutive failures',
                'skipped': True
            }
        
        # Load instructions
        instructions_file = problem_dir / "instructions.txt"
        if not instructions_file.exists():
            logger.error(f"No instructions.txt found for {problem_name}")
            return {
                'parsed': False,
                'solved': False,
                'correct': None,
                'attempts': 0,
                'error': 'No instructions file'
            }
        
        with open(instructions_file, 'r') as f:
            task_description = f.read().strip()
        
        # Load reference plan
        reference_actions = self.load_reference_plan(problem_dir)
        
        # Initial prompt
        initial_prompt = (
            "You are an expert in PDDL planning. Given the following task description:\n\n"
            f"{task_description}\n\n"
            "Please generate a JSON object with exactly two keys: 'domain' and 'problem'. "
            "The value of 'domain' should be a complete PDDL DOMAIN file, and the value of 'problem' "
            "should be a complete PDDL PROBLEM file. Ensure that predicates and variable names match between the two files."
            # "Use exactly this action header grammar: "
            # "(:action <name>\n"
            # " :parameters (<typed vars>)\n"
            # " :precondition (<formula>)\n"
            # " :effect (<formula>)\n"
            # ")\n"
            # "Ensure that the PDDL files are valid. Be careful with the number of parameters for actions and predicates."
            # "Object names should be unique and different from their types."
        )
        
        conversation_history = [{"role": "user", "content": initial_prompt}]
        logger.info("Initial prompt sent to LLM")
        
        domain_text = ""
        problem_text = ""
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"\nAttempt {attempt}/{max_attempts}")
            
            try:
                # Get LLM response
                response = self.get_completion(conversation_history.copy())
                domain_text = response.domain.strip()
                problem_text = response.problem.strip()
                
                # Save generated files
                domain_path, problem_path = self.save_pddl_files(
                    domain_text, problem_text, output_dir
                )
                logger.info("PDDL files generated and saved")
                
                # Try to solve
                solved, plan, error = self.solve_pddl(domain_path, problem_path)
                
                if solved:
                    logger.info(f"✓ Problem SOLVED on attempt {attempt}")
                    logger.info(f"Plan: {plan}")
                    
                    # Compare with reference
                    plan_correct = self.compare_plans(plan, reference_actions)
                    
                    if plan_correct is not None:
                        if plan_correct:
                            logger.info("✓ Plan matches reference plan!")
                        else:
                            logger.info("⚠ Plan differs from reference plan")
                            logger.info(f"Generated: {self.extract_action_sequence(plan)}")
                            logger.info(f"Reference: {reference_actions}")
                    
                    # Reset consecutive failures on success
                    self.consecutive_failures = 0
                    
                    return {
                        'parsed': True,
                        'solved': True,
                        'correct': plan_correct,
                        'attempts': attempt,
                        'plan': str(plan),
                        'skipped': False
                    }
                else:
                    logger.warning(f"✗ Solving failed: {error}")
                    
                    # Build refinement prompt
                    refinement_prompt = (
                        "The following error occurred during planning/validation of the PDDL files:\n"
                        f"{error}\n\n"
                        "Here are the previously generated PDDL files:\n\n"
                        "DOMAIN file:\n" + domain_text + "\n\n"
                        "PROBLEM file:\n" + problem_text + "\n\n"
                        "Please generate a revised JSON object with keys 'domain' and 'problem' "
                        "containing updated PDDL files that fix the error."
                    )
                    
                    conversation_history.append({"role": "user", "content": refinement_prompt})
                    logger.info("Refinement prompt sent to LLM")
                    
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                logger.error(f"Exception during attempt {attempt}: {str(e)}")
                
                if attempt < max_attempts:
                    refinement_prompt = (
                        f"An error occurred: {str(e)}\n\n"
                        "Please generate a valid JSON object with keys 'domain' and 'problem' "
                        "containing correct PDDL files."
                    )
                    conversation_history.append({"role": "user", "content": refinement_prompt})
                    time.sleep(1)
        
        # Failed after all attempts
        logger.error(f"✗ Failed to solve {problem_name} after {max_attempts} attempts")
        
        # Increment consecutive failures
        self.consecutive_failures += 1
        logger.warning(f"Consecutive failures: {self.consecutive_failures}")
        
        return {
            'parsed': False,
            'solved': False,
            'correct': None,
            'attempts': max_attempts,
            'error': 'Max attempts reached',
            'skipped': False
        }

def calculate_statistics(results):
    """Calculate aggregate statistics from results."""
    stats = {
        'total_problems': len(results),
        'parsed_count': 0,
        'solved_count': 0,
        'correct_count': 0,
        'compared_count': 0,
        'skipped_count': 0,
        'solved_at_k': defaultdict(int),
        'total_attempts': 0,
        'avg_attempts': 0,
    }
    
    for problem, result in results.items():
        if result.get('skipped', False):
            stats['skipped_count'] += 1
            continue
        
        if result['parsed']:
            stats['parsed_count'] += 1
        
        if result['solved']:
            stats['solved_count'] += 1
            attempts = result['attempts']
            stats['total_attempts'] += attempts
            
            # Track solved@k
            for k in range(1, 11):
                if attempts <= k:
                    stats['solved_at_k'][k] += 1
        
        if result['correct'] is not None:
            stats['compared_count'] += 1
            if result['correct']:
                stats['correct_count'] += 1
    
    # Calculate percentages
    stats['parsed_pct'] = (stats['parsed_count'] / stats['total_problems'] * 100) if stats['total_problems'] > 0 else 0
    stats['solved_pct'] = (stats['solved_count'] / stats['total_problems'] * 100) if stats['total_problems'] > 0 else 0
    stats['correct_pct'] = (stats['correct_count'] / stats['compared_count'] * 100) if stats['compared_count'] > 0 else 0
    stats['avg_attempts'] = (stats['total_attempts'] / stats['solved_count']) if stats['solved_count'] > 0 else 0
    
    return stats

def save_results(all_model_results, results_dir, dt_stamp):
    """Save results to JSON file."""
    if not all_model_results:
        print("No results to save.")
        return
    
    results_file = results_dir / f"all_results_{dt_stamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_model_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    
    # Also print a quick summary
    print(f"\n{'='*115}")
    print("PARTIAL RESULTS SUMMARY")
    print(f"{'='*115}")
    print(f"{'Model':<30} {'Status':<15} {'Skipped':<10} {'Parsed':<12} {'Solved':<12} {'Correct':<12} {'Avg Attempts':<15}")
    print(f"{'-'*115}")
    
    for model_spec, data in all_model_results.items():
        stats = data['stats']
        incomplete_marker = " (INCOMPLETE)" if data.get('incomplete', False) else ""
        print(f"{model_spec:<30}{incomplete_marker:<15} "
              f"{stats['skipped_count']:<10} "
              f"{stats['parsed_count']}/{stats['total_problems']} ({stats['parsed_pct']:>5.1f}%)  "
              f"{stats['solved_count']}/{stats['total_problems']} ({stats['solved_pct']:>5.1f}%)  "
              f"{stats['correct_count']}/{stats['compared_count']} ({stats['correct_pct']:>5.1f}%)  "
              f"{stats['avg_attempts']:>6.2f}")

def main():
    # Setup directories
    base_dir = Path(__file__).parent / "classical-examples"
    logs_dir = Path(__file__).parent / "logs"
    results_dir = Path(__file__).parent / "test_results"
    
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Timestamp for this run
    dt_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Store all results
    all_model_results = {}
    current_model_results = {}  # Track current model's partial results
    current_model_spec = None
    
    # Setup signal handler for graceful exit on Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n⚠️  Interrupted! Saving results before exit...")
        
        # Save partial results of current model if any
        if current_model_spec and current_model_results:
            stats = calculate_statistics(current_model_results)
            all_model_results[current_model_spec] = {
                'results': current_model_results,
                'stats': stats,
                'incomplete': True  # Mark as incomplete
            }
            print(f"  (Including partial results for {current_model_spec})")
        
        save_results(all_model_results, results_dir, dt_stamp)
        print("✓ Results saved. Exiting.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Test each model
    for model_spec in MODELS:
        # Update current model being tested (for signal handler)
        current_model_spec = model_spec
        current_model_results.clear()
        
        # Parse model spec for display and logging
        clean_spec = model_spec.rstrip("*")
        has_star = model_spec.endswith("*")
        
        if "/" in clean_spec:
            provider, model_name = clean_spec.split("/", 1)
        else:
            provider = "openai"
            model_name = clean_spec
        
        print(f"\n{'#'*70}")
        print(f"# Testing Model: {model_spec}")
        print(f"{'#'*70}\n")
        
        # Setup logging for this model (use sanitized name for filename)
        log_filename = f"test_{provider}_{model_name}_{dt_stamp}.log".replace("/", "_")
        log_file = logs_dir / log_filename
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True  # Reset logging config for each model
        )
        logger = logging.getLogger()
        
        logger.info(f"{'='*70}")
        logger.info(f"Testing Model: {model_spec}")
        logger.info(f"Provider: {provider}")
        logger.info(f"Model Name: {model_name}")
        logger.info(f"Structured Output: {'No' if has_star else 'Yes'}")
        logger.info(f"Timestamp: {dt_stamp}")
        logger.info(f"{'='*70}")
        
        # Initialize tester
        tester = ModelTester(model_spec, API_KEY)
        model_results = current_model_results  # Use the shared reference for signal handler
        
        # Test each problem
        for problem_name in PROBLEMS:
            problem_dir = base_dir / problem_name
            
            if not problem_dir.exists():
                logger.warning(f"Problem directory not found: {problem_name}")
                continue
            
            # Create output directory for this model+problem (sanitize name)
            sanitized_model = f"{provider}_{model_name}".replace("/", "_")
            output_dir = results_dir / f"{sanitized_model}_{problem_name}_{dt_stamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test the problem
            result = tester.test_problem(problem_name, problem_dir, output_dir, logger)
            model_results[problem_name] = result
            
            # Brief pause between problems
            time.sleep(2)
        
        # Calculate statistics for this model
        stats = calculate_statistics(model_results)
        
        # Save results
        all_model_results[model_spec] = {
            'results': model_results,
            'stats': stats,
            'incomplete': False  # Mark as complete
        }
        
        # Clear current model tracking since this one is complete
        current_model_spec = None
        current_model_results.clear()
        
        # Log summary for this model
        logger.info(f"\n{'='*70}")
        logger.info(f"Summary for {model_spec}")
        logger.info(f"{'='*70}")
        logger.info(f"Problems Tested: {stats['total_problems']}")
        logger.info(f"Skipped: {stats['skipped_count']}")
        logger.info(f"Parsed: {stats['parsed_count']} ({stats['parsed_pct']:.1f}%)")
        logger.info(f"Solved: {stats['solved_count']} ({stats['solved_pct']:.1f}%)")
        logger.info(f"Correct Plans: {stats['correct_count']}/{stats['compared_count']} ({stats['correct_pct']:.1f}%)")
        logger.info(f"Average Attempts (when solved): {stats['avg_attempts']:.2f}")
        logger.info(f"\nSolved@k:")
        for k in range(1, 11):
            count = stats['solved_at_k'][k]
            pct = (count / stats['total_problems'] * 100) if stats['total_problems'] > 0 else 0
            logger.info(f"  k={k:2d}: {count:2d} ({pct:.1f}%)")
    
    # Save aggregate results to JSON
    save_results(all_model_results, results_dir, dt_stamp)

if __name__ == "__main__":
    main()

