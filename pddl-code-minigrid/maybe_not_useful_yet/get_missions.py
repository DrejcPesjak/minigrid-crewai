import random
import yaml
import copy
import gymnasium as gym
from pydantic import BaseModel
from typing import Optional

# Assume LEVELS and convert_observation are defined in your project.
from constants import LEVELS
from convert_space import convert_observation

# Define a simple state to hold the game information.
class RobotGymFlowState(BaseModel):
    observation: Optional[dict] = None
    mission: Optional[str] = None
    terminated: Optional[bool] = False
    truncated: Optional[bool] = False
    reward: Optional[float] = 0.0
    info: Optional[dict] = None
    last_action: Optional[int] = None
    observation_history: Optional[list] = []

class MinigridWorkFlow():
    """Workflow to let a human play Minigrid and record training examples."""

    def init_env(self, env_name):
        """Initialize Minigrid environment."""
        # env_name = input("Enter Minigrid environment name (default: Empty-5x5, or 'random'): ")
        if env_name == 'random':
            env_name = random.choice(LEVELS)
        elif env_name == '':
            env_name = 'MiniGrid-Empty-5x5-v0'  # default
        elif env_name not in LEVELS:
            print(f"Invalid environment name: {env_name}")
            print("Please choose from the available environments, press Enter for default, or type 'random'")
            return "end"
        
        self.env = gym.make(env_name)#, render_mode="human")
        self.env_name = env_name
        obs, info = self.env.reset()
        obs['image'][3][6][0] = 10
        self.state = RobotGymFlowState()
        self.state.observation = obs
        self.state.mission = obs.get('mission', '')
        print(f'"{self.env_name}", "{self.state.mission}"')
        # print("Environment initialized.")
    
    def play_episode_human(self):
        """Let the user play, recording each state-action pair."""
        training_examples = []
        while True:
            # Print current observation (you might want to format it nicely)
            # print("\nCurrent observation:")
            # print(self.state.observation)
            
            action_input = input("Enter an action number (0-6) or 'q' to quit: ")
            if action_input.lower() == 'q':
                break
            try:
                action = int(action_input)
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 6 or 'q' to quit.")
                continue
            if action not in range(7):
                print("Action not in valid range (0-6).")
                continue

            # Optionally, convert observation if needed for better readability
            cnv_obs = convert_observation(self.state.observation)
            
            # Record the current state and chosen action as a training example
            example = {
                "mission": self.state.mission,
                "observation": cnv_obs,
                "observation_history": copy.deepcopy(self.state.observation_history[-5:]),  # last 5 for context
                "action": action
            }
            training_examples.append(example)
            
            # Append the current observation to history
            self.state.observation_history.append(copy.deepcopy(cnv_obs))
            
            # Take the action in the environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs['image'][3][6][0] = 10
            
            # Update the state with the new observation and mission
            self.state.observation = obs
            self.state.mission = obs.get('mission', '')
            self.state.reward = reward
            self.state.terminated = terminated
            self.state.truncated = truncated
            self.state.info = info
            
            if terminated or truncated:
                # print(self.state)
                print("Episode ended.")
                break
        
        # Save the collected training examples to a YAML file
        game_id = random.randint(0, 100000)
        file_name = f"human_play_{self.env_name}_{game_id}.yaml"
        with open(file_name, "w") as f:
            yaml.dump(training_examples, f)
        print(f"\nTraining examples saved to '{file_name}'.")
    
    def terminate(self):
        """Clean up the environment."""
        if self.env:
            self.env.close()
    
    def run_human(self):
        """Run the manual play workflow."""
        for l in LEVELS:        
            self.init_env(l)
            # self.play_episode_human()
            self.terminate()

if __name__ == "__main__":
    workflow = MinigridWorkFlow()
    workflow.run_human()
