#!/usr/bin/env python3
"""
Script to solve all classical PDDL problems and save plans to plan.txt in each folder.
"""

import os
import sys
from pathlib import Path
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.engines import PlanGenerationResultStatus

def solve_pddl_problem(domain_file, problem_file):
    """
    Solve a PDDL problem and return the result.
    
    Args:
        domain_file: Path to domain PDDL file
        problem_file: Path to problem PDDL file
        
    Returns:
        tuple: (result, success_flag)
    """
    try:
        reader = PDDLReader()
        print(f"  Parsing PDDL files...")
        problem = reader.parse_problem(str(domain_file), str(problem_file))
        
        print(f"  Starting planner...")
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            result = planner.solve(problem)
        
        if result.status in [PlanGenerationResultStatus.SOLVED_SATISFICING, 
                            PlanGenerationResultStatus.SOLVED_OPTIMALLY]:
            return result, True
        else:
            print(f"  Plan status: {result.status}")
            return result, False
            
    except Exception as e:
        print(f"  Error: {e}")
        return None, False

def save_plan(plan, output_file):
    """Save the plan to a text file."""
    with open(output_file, 'w') as f:
        f.write(str(plan))
        f.write('\n')

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # List all subdirectories (each is a problem domain)
    problem_dirs = [d for d in script_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(problem_dirs)} problem directories")
    print("=" * 60)
    
    results_summary = []
    
    for problem_dir in sorted(problem_dirs):
        problem_name = problem_dir.name
        print(f"\nProcessing: {problem_name}")
        print("-" * 60)
        
        # Find domain.pddl file
        domain_files = list(problem_dir.glob("domain.pddl"))
        if not domain_files:
            print(f"  No domain.pddl found, skipping...")
            results_summary.append((problem_name, "SKIP", "No domain.pddl"))
            continue
        
        domain_file = domain_files[0]
        
        # Find problem file (various naming conventions)
        problem_files = [f for f in problem_dir.glob("*.pddl") if f.name != "domain.pddl"]
        if not problem_files:
            print(f"  No problem file found, skipping...")
            results_summary.append((problem_name, "SKIP", "No problem file"))
            continue
        
        # Use the first problem file found
        problem_file = problem_files[0]
        print(f"  Domain: {domain_file.name}")
        print(f"  Problem: {problem_file.name}")
        
        # Solve the problem
        result, success = solve_pddl_problem(domain_file, problem_file)
        
        if success and result:
            print(f"  ✓ Plan found successfully!")
            print(f"  Plan: {result.plan}")
            
            # Save plan to plan.txt in the problem directory
            plan_file = problem_dir / "plan.txt"
            save_plan(result.plan, plan_file)
            print(f"  Plan saved to: {plan_file}")
            
            results_summary.append((problem_name, "SUCCESS", str(result.status.name)))
        else:
            status = result.status.name if result else "ERROR"
            print(f"  ✗ Failed to find plan")
            results_summary.append((problem_name, "FAILED", status))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status, detail in results_summary:
        status_symbol = "✓" if status == "SUCCESS" else "✗" if status == "FAILED" else "○"
        print(f"{status_symbol} {name:20s} - {status:10s} ({detail})")
    
    print("\n" + "=" * 60)
    success_count = sum(1 for _, s, _ in results_summary if s == "SUCCESS")
    print(f"Successfully solved: {success_count}/{len(results_summary)} problems")

if __name__ == "__main__":
    main()

