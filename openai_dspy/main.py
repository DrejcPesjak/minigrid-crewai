import random
import yaml

import gymnasium as gym
from openai import OpenAI

from pydantic import BaseModel
from typing import Optional

from constants import LEVELS
from convert_space import convert_observation
from output_types import AgentAction

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


class LlmRobotGymAgent:
    def __init__(self, agent_name: str):
        # Load agent configuration
        with open("agents_config.yaml", "r") as file:
            config = yaml.safe_load(file)
        
        agent_config = config["agents"].get(agent_name)
        if not agent_config:
            raise ValueError(f"Agent '{agent_name}' not found in configuration.")

        self.agent_name = agent_config["agent_name"]
        self.system_message = agent_config["system_message"]
        self.task_message = agent_config["task_message"]
        self.output_format = eval(agent_config["output_format"])  # Convert class name to Pydantic class
        
        # self.client = OpenAI(
        #     api_key="sk-proj-1VvkSUwvyvF"
        # )
        # self.model_name = "gpt-4o"

        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model_name = "llama3.2"
        # self.model_name = "deepseek-r1:8b"

        # self.client = OpenAI(
        #     api_key="none",
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        # )
        # self.model_name="gemini-1.5-flash"

    def parse_input(self, inputs):
        system_message = {"role": "system", "content": self.system_message}
        task_message = {"role": "user", "content": self.task_message.format(**inputs)}
        return [system_message, task_message]

    def completion(self, messages):
        return self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=self.output_format,
        ).choices[0].message.parsed


class MinigridWorkFlow():
    """Flow for running Minigrid environments with CrewAI agents"""

    def init_env(self):
        """Initialize Minigrid environment"""
        
        env_name = input("Enter Minigrid environment name (default: Empty-5x5, or 'random'): ")
            
        if env_name == 'random':
            env_name = random.choice(LEVELS)
        elif env_name == '':
            env_name = 'MiniGrid-Empty-5x5-v0' # Default environment
        elif env_name not in LEVELS:
            print(f"Invalid environment name: {env_name}")
            print("Please choose from the available environments, press Enter for default, or type 'random'")
            return "end"
            
        self.env = gym.make(env_name, render_mode="human")
        self.agent = LlmRobotGymAgent(agent_name="main_agent")
        obs, info = self.env.reset() # info is empty
        print("init_env")
        obs['image'][3][6][0] = 10  
        self.state.observation = obs # image 7x7x3, direction, mission
        self.state.mission = obs['mission']

    def agent_action(self):
        """Get action from agent"""
        print("agent_action")
        cnv_obs = convert_observation(self.state.observation)
        inputs = {
            'mission': str(self.state.mission),
            'observation': str(cnv_obs),
            'observation_history': str(self.state.observation_history[-5:]), # last 5 observations
        }
        self.state.observation_history.append(cnv_obs)
        msg = self.agent.parse_input(inputs)
        result = self.agent.completion(msg) # AgentAction
        print(f"result: {result}")

        self.state.last_action = 6 # done
        try:
            if result.action in range(7):
                self.state.last_action = result.action
            else:
                print(f"Invalid action: {result.action}")
        except KeyError:
            print("Key 'action' not found in CrewOutput.")
            print("Please make sure the agent returns a valid action.")


    def play_episode(self): 
        """Route based on action"""
        print("play_episode")
        obs, reward, terminated, truncated, info = self.env.step(self.state.last_action)
        obs['image'][3][6][0] = 10
        self.state.observation = obs
        self.state.mission = obs['mission']
        self.state.reward = reward
        self.state.terminated = terminated
        self.state.truncated = truncated
        self.state.info = info # info is empty

        if self.state.terminated or self.state.truncated:
            print(f"Episode ended, because of terminated: {terminated} or truncated: {truncated}")
            return "end"
        else:
            return "play"

    def terminate(self):
        """Clean up environment"""
        print("terminate")
        if self.env:
            self.env.close()
    
    def run(self):
        """Run the workflow"""
        self.state = RobotGymFlowState()
        self.init_env()
        while True:
            self.agent_action()
            # if self.state.last_action == 6:
            #     break
            if self.play_episode() == "end":
                break
        self.terminate()

if __name__ == "__main__":
    workflow = MinigridWorkFlow()
    workflow.run()
