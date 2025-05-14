import os
import logging
import datetime
import random
import yaml

import gymnasium as gym
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

from constants import LEVELS
from convert_space import convert_observation
from output_types import AgentAction

# -------------------------------------------------------------------------------------------------
# State definition
# -------------------------------------------------------------------------------------------------
class RobotGymFlowState(BaseModel):
    """State for the RobotGymFlow"""
    observation: Optional[dict] = None
    mission: Optional[str] = None
    terminated: Optional[bool] = False
    truncated: Optional[bool] = False
    reward: Optional[float] = 0.0
    info: Optional[dict] = None
    last_action: Optional[int] = None
    observation_history: Optional[list] = []
    memory_history: Optional[list] = []

# -------------------------------------------------------------------------------------------------
# Agent class
# -------------------------------------------------------------------------------------------------
class LlmRobotGymAgent:
    def __init__(self, agent_name: str):
        # Load agent configuration
        with open("agents_config3.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        agent_config = config["agents"].get(agent_name)
        if not agent_config:
            raise ValueError(f"Agent '{agent_name}' not found in configuration.")

        # This message will be logged later once logging is configured properly.
        logging.info(f"Loaded agent config for {agent_name}: "
                     f"{ {k: v for k, v in agent_config.items() if k != 'api_key'} }")

        self.agent_name = agent_config["agent_name"]
        self.system_message = agent_config["system_message"]
        self.task_message = agent_config["task_message"]
        self.output_format = eval(agent_config["output_format"])  # Convert class name to Pydantic class

        # Uncomment an appropriate client & model based on your deployment
        self.client = OpenAI(api_key="sk-proj-1VvkSU")
        self.model_name = "o4-mini"
        # self.client = OpenAI(api_key="none", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        # self.model_name = "gemini-1.5-flash"
        # self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # self.model_name = "llama3.2"
        # self.model_name = "deepseek-r1:8b"

    def parse_input(self, inputs):
        system_message = {"role": "system", "content": self.system_message}
        task_message = {"role": "user", "content": self.task_message.format(**inputs)}
        return [system_message, task_message]

    def completion(self, messages):
        # Assuming the API returns an object with both 'action' and 'explanation' attributes.
        return self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=self.output_format,
        ).choices[0].message.parsed


# -------------------------------------------------------------------------------------------------
# Workflow class with improved logging initialization
# -------------------------------------------------------------------------------------------------
class MinigridWorkFlow():
    """Flow for running Minigrid environments with CrewAI agents"""

    def run(self):
        # Ask user for environment name first so we can use it in the log filename.
        env_name = input("Enter Minigrid environment name (default: Empty-5x5, or 'random'): ").strip()
        if env_name == 'random':
            env_name = random.choice(LEVELS)
        elif env_name == '':
            env_name = 'MiniGrid-Empty-5x5-v0'  # Default environment
        elif env_name not in LEVELS:
            err_msg = f"Invalid environment name: {env_name}. Please choose from available environments."
            print(err_msg)
            return

        self.level_name = env_name

        # Create an agent early (only to retrieve model name for logging purposes).
        self.agent = LlmRobotGymAgent(agent_name="main_agent")

        # Setup logging BEFORE any further actions.
        self.dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = "./logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        log_filename = f"{logs_dir}/{self.agent.model_name}_{self.level_name}_{self.dt_str}.log"
        # The force=True parameter ensures that logging is reconfigured even if basicConfig was called before.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
            force=True
        )
        logging.info("Logging initialized.")
        
        # Now initialize state and the environment.
        self.state = RobotGymFlowState()
        self.env = gym.make(env_name, render_mode="human")
        obs, info = self.env.reset()
        # Modify observation for demonstration purposes
        obs['image'][3][6][0] = 10  
        self.state.observation = obs
        self.state.mission = obs['mission']
        logging.info(f"Environment '{env_name}' initialized.")
        logging.info(f"Initial mission: {self.state.mission}")

        # Log initial session details.
        logging.info("=== Session Start ===")
        logging.info("Agent configuration loaded from agents_config.yaml")
        logging.info(f"Model Name: {self.agent.model_name}")
        logging.info(f"Level Name: {self.level_name}")
        logging.info(f"Datetime Stamp: {self.dt_str}")
        logging.info(f"Mission: {self.state.mission}")

        # Main loop.
        while True:
            self.agent_action()
            if self.play_episode() == "end":
                break
        self.terminate()
        logging.info("=== Session End ===")

    def agent_action(self):
        """Get action from agent and log details"""
        logging.info("Starting agent action step.")
        cnv_obs = convert_observation(self.state.observation)
        # logging.info(f"Converted observation: {cnv_obs}")
        inputs = {
            'mission': str(self.state.mission),
            'memory_history': str(self.state.memory_history),
            'observation': str(cnv_obs),
            # 'observation_history': str(self.state.observation_history[-5:]),  # last 5 observations
        }
        self.state.observation_history.append(cnv_obs)
        logging.info(f"Inputs for agent: {inputs}")

        msg = self.agent.parse_input(inputs)
        result = self.agent.completion(msg)  # Assuming result has 'action' and 'explanation'
        logging.info(f"Agent result: {result}")

        # Decide on the action.
        self.state.last_action = 6  # Default/fallback action
        try:
            if result.action in range(7):
                self.state.last_action = result.action
            else:
                err_msg = f"Invalid action received: {result.action}"
                logging.error(err_msg)
                print(err_msg)
            if result.memory:
                self.state.memory_history.append(result.memory)
            else:
                self.state.memory_history.append(" ")
        except KeyError:
            err_msg = "Key 'action' not found in agent's output. Check agent response format."
            logging.error(err_msg)
            print(err_msg)

    def play_episode(self):
        """Execute one step in the environment and log the outcome"""
        logging.info("Taking a step in the environment.")
        obs, reward, terminated, truncated, info = self.env.step(self.state.last_action)
        obs['image'][3][6][0] = 10
        self.state.observation = obs
        self.state.mission = obs['mission']
        self.state.reward = reward
        self.state.terminated = terminated
        self.state.truncated = truncated
        self.state.info = info

        if terminated or truncated:
            end_msg = f"Episode ended - terminated: {terminated}, truncated: {truncated}"
            logging.info(end_msg)
            print(end_msg)
            return "end"
        else:
            return "play"

    def terminate(self):
        """Clean up environment"""
        logging.info("Terminating environment.")
        if self.env:
            self.env.close()
        logging.info("Environment closed.")


if __name__ == "__main__":
    workflow = MinigridWorkFlow()
    workflow.run()
