import random
import sys

import gymnasium as gym
from openai import OpenAI

from pydantic import BaseModel, Field
from typing import Optional

from constants import LEVELS
from convert_space import convert_observation

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


class AgentAction(BaseModel):
    explanation: str = Field(..., description="The explanation why should the action be taken")
    action: int = Field(..., description="The action to take")

class LlmRobotGymAgent():

    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.model_name = "llama3.2"
    
    def parse_input(self, inputs):
        system_message = {"role": "system", "content": """
                          You are an intelligent agent designed to play Minigrid games. 
                          Navigate through the Minigrid environment efficiently by taking optimal actions to complete objectives. 
                          Make decisions based on the dynamic 7x7 observation grid relative to your position and orientation.
                          """}
        task_message = """
        Play the Minigrid environment by taking optimal actions based on observations.

        Observation Grid:
        - A 7x7 grid dynamically generated relative to your position and orientation.
        - Your position is marked as "agent_X", where X is your orientation with respect to current observation grid.
        - The direction is according to the global map.
        - The grid uses the format <object>_<color>_<state>.
        - Tiles labeled unseen are outside your visible range.
        - And "lava" tiles will kill you if you step on them.

        Actions:
        - 0: Turn left
        - 1: Turn right
        - 2: Move forward
        - 3: Pickup object
        - 4: Drop object
        - 5: Toggle (e.g., open doors)
        - 6: Done (e.g., finish the game)

        Rules:
        - Use only valid actions (0-6).
        - PICKUP (3) works only if the target object is directly in front of you.
        - TOGGLE (5) works only for interactable objects (e.g., doors).

        Plan:
        - Step 1: Identify your current position and direction.
        - Step 2: Locate key objects that are relevant to the mission (e.g., keys, doors, goal).
        - Step 3: Plan optimal movements to achieve the mission.

        Mission:
        {mission}

        Observation:
        {observation}

        Observation history:
        {observation_history}

        Your Answer:
        Explain the reasoning for your action and output a valid action number (0-6). You are allowed to execute only one action.
        """
        user_message = {"role": "user", "content": task_message.format(**inputs)}

        return [system_message, user_message]
    
    def completion(self, messages) -> AgentAction:
        return self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=AgentAction,
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
        self.agent = LlmRobotGymAgent()
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
            'observation_history': str(self.state.observation_history[-5:]), # last 5 observations
            'observation': str(cnv_obs),
            'mission': str(self.state.mission)
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
            if self.state.last_action == 6:
                break
            if self.play_episode() == "end":
                break
        self.terminate()

if __name__ == "__main__":
    workflow = MinigridWorkFlow()
    workflow.run()
