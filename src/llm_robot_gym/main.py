#!/usr/bin/env python
import random
import sys

import gymnasium as gym
from crewai.flow.flow import Flow, listen, router, start, or_
from pydantic import BaseModel
from typing import Optional

from constants import LEVELS
from crews.main_crew.crew import LlmRobotGymCrew
from utils.convert_space import convert_observation

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

class MinigridFlow(Flow[RobotGymFlowState]):
    """Flow for running Minigrid environments with CrewAI agents"""
    @start()
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
        self.crew = LlmRobotGymCrew().crew()
        obs, info = self.env.reset() # info is empty
        print("init_env")
        obs['image'][3][6][0] = 10  
        self.state.observation = obs # image 7x7x3, direction, mission
        self.state.mission = obs['mission']

    @listen(or_(init_env, "play"))
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
        result = self.crew.kickoff(inputs=inputs)
        print(f"result: {result}")

        self.state.last_action = 6 # done
        try:
            if result['action'] in range(7):
                self.state.last_action = result['action']
            else:
                print(f"Invalid action: {result['action']}")
        except KeyError:
            print("Key 'action' not found in CrewOutput.")
            print("Please make sure the agent returns a valid action.")


    @router(agent_action)
    def play_episode(self): 
        """Route based on action"""
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

    @listen(play_episode) 
    def terminate(self):
        """Clean up environment"""
        print("terminate")
        if self.env:
            self.env.close()
    
def kickoff():
    """
    Run the flow.
    """
    minigrid_flow = MinigridFlow()
    minigrid_flow.kickoff()


def plot():
    """
    Plot the flow.
    """
    minigrid_flow = MinigridFlow() 
    minigrid_flow.plot()

if __name__ == "__main__":
    kickoff()
    
