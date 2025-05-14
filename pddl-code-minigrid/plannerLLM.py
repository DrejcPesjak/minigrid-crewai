###
# chatgpt_pddl3.py with logging
###

import time
import os
import logging
import datetime
from openai import OpenAI
from pydantic import BaseModel
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner, PlanValidator
from unified_planning.engines import PlanGenerationResultStatus

from llmclient import ChatGPTClient

# --- Define a Pydantic model for ChatGPT's response ---
class PDDLResponse(BaseModel):
    domain: str
    problem: str

# --- Function to save PDDL files ---
def save_pddl_files(domain_text, problem_text):
    with open("domain.pddl", "w") as f:
        f.write(domain_text)
    with open("problem.pddl", "w") as f:
        f.write(problem_text)

# --- Function to parse, plan, and validate ---
def plan_with_unified_planning():
    reader = PDDLReader()
    logging.info("Parsing PDDL files...")
    problem = reader.parse_problem("domain.pddl", "problem.pddl")
    logging.info("PDDL files parsed successfully.")
    
    logging.info("Starting planner...")
    with OneshotPlanner(problem_kind=problem.kind) as planner:
        result = planner.solve(problem)
    logging.info("Planner finished execution.")
    
    logging.info("Validating plan...")
    with PlanValidator(name="tamer") as validator:
        validation_result = validator.validate(problem, result.plan)
    logging.info("Plan validation finished.")
    
    return result, validation_result

# --- Main pipeline ---
def main():
    # Input for task name (sanitized for file naming) and task description
    task_name = input("Enter planning task name: ").strip()
    task_description = input("Enter planning task description: ").strip()
    task_name_sanitized = task_name.replace(" ", "_")
    
    # Create a datetime stamp
    dt_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize ChatGPT client (model name will be used for logging as well)
    chat_client = ChatGPTClient(model_name="o1", output_format=PDDLResponse)
    model_name = chat_client.model_name
    
    # Setup logging to file in the ./logs directory
    logs_dir = "./logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_filename = os.path.join(logs_dir, f"{task_name_sanitized}_{model_name}_{dt_stamp}.log")
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Log initial details
    logging.info("=== SESSION START ===")
    logging.info(f"Task Name: {task_name}")
    logging.info(f"Task Description: {task_description}")
    logging.info(f"Model Name: {model_name}")
    logging.info(f"Datetime Stamp: {dt_stamp}")
    
    conversation_history = []  # Holds the conversation context
    
    # Build initial prompt for PDDL generation
    initial_prompt = (
        "You are an expert in PDDL planning. Given the following task description:\n\n"
        f"{task_description}\n\n"
        "Please generate a JSON object with exactly two keys: 'domain' and 'problem'. "
        "The value of 'domain' should be a complete PDDL DOMAIN file, and the value of 'problem' "
        "should be a complete PDDL PROBLEM file. Ensure that predicates and variable names match between the two files."
    )
    conversation_history.append({"role": "user", "content": initial_prompt})
    
    # Log the prompt before sending it to the LLM
    logging.info("Sending initial prompt to LLM:")
    logging.info(initial_prompt)
    
    max_attempts = 10
    attempt = 1
    success = False
    domain_text = ""
    problem_text = ""
    final_plan_status = "N/A"
    
    while attempt <= max_attempts and not success:
        try:
            # Get response from the ChatGPT client
            response = chat_client.chat_completion(conversation_history.copy())
            # Log the LLM's output
            response_json = response.model_dump_json()
            logging.info("Received response from LLM:")
            logging.info(response_json)
            
            # Retrieve domain and problem texts from the response
            domain_text = response.domain.strip()
            problem_text = response.problem.strip()
            save_pddl_files(domain_text, problem_text)
            logging.info("PDDL files saved.")
            
            # Attempt to plan with the generated PDDL files
            result, validation_result = plan_with_unified_planning()
            
            if result.status in [PlanGenerationResultStatus.SOLVED_SATISFICING, PlanGenerationResultStatus.SOLVED_OPTIMALLY]:
                logging.info("Plan found successfully!")
                logging.info(f"Domain:\n{domain_text}")
                logging.info(f"Problem:\n{problem_text}")
                logging.info(f"Plan: {result.plan}")
                success = True
                final_plan_status = result.status.name
                break
            else:
                raise Exception(f"Plan status not successful: {result.status}")
        
        except Exception as e:
            error_msg = str(e)
            logging.error("Error during planning/validation:")
            logging.error(error_msg)
            print("\nError during planning/validation:", error_msg)
            
            # Build refinement prompt based on whether PDDL files are available
            if domain_text and problem_text:
                refinement_prompt = (
                    "The following error occurred during planning/validation of the PDDL files:\n"
                    f"{error_msg}\n\n"
                    "Here are the previously generated PDDL files:\n\n"
                    "DOMAIN file:\n" + domain_text + "\n\n"
                    "PROBLEM file:\n" + problem_text + "\n\n"
                    "Please generate a revised JSON object with keys 'domain' and 'problem' containing updated PDDL files that fix the error."
                )
            else:
                refinement_prompt = (
                    "The following error occurred during planning/validation of the PDDL files:\n"
                    f"{error_msg}\n\n"
                    "Please generate a revised JSON object with keys 'domain' and 'problem' containing updated PDDL files that fix the error."
                )
            
            conversation_history.append({"role": "user", "content": refinement_prompt})
            # Log the refinement prompt for the next attempt
            logging.info("Sending refinement prompt to LLM:")
            logging.info(refinement_prompt)
            attempt += 1
            time.sleep(1)  # To avoid potential rate limit issues

    # Log final status details
    if success:
        if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            plan_status_detail = "Satisficing"
        elif result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
            plan_status_detail = "Optimal"
        else:
            plan_status_detail = "Unknown"
        logging.info("=== FINAL PLAN STATUS ===")
        logging.info(f"Plan is viable: True")
        logging.info(f"Plan is feasible: True")
        logging.info(f"Plan is satisficing: {result.status == PlanGenerationResultStatus.SOLVED_SATISFICING}")
        logging.info(f"Plan is optimal: {result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY}")
        logging.info(f"Overall Result: Success ({plan_status_detail})")
    else:
        logging.info("=== FINAL PLAN STATUS ===")
        logging.info("Plan generation failed after maximum attempts.")
        logging.info(f"Overall Result: Failure")
    
    logging.info("=== SESSION END ===")


if __name__ == "__main__":
    main()
