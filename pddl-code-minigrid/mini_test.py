from minigrid import MiniGridEnv

def test_minigrid():
    # Initialize the agent and environment
    env = MiniGridEnv(level_name="MiniGrid-Empty-5x5-v0")

    env.start_env()
    
    # Define a sequence of actions
    action_sequence = [
        "move-forward()",
        "turn-left()",
        "move-forward()",
        "turn-right()",
        "move-forward()"
    ]

    a = env.parse_actions(action_sequence)

    for action in a:
        # Execute the action in the environment
        obs, reward, terminated, truncated, info = env.play_episode(action)
        
        # Check if the episode is terminated or truncated
        if terminated or truncated:
            print("Episode ended.")
            break
        else:
            print(f"Action executed: {action}, Observation: {obs}, Reward: {reward}")
        
        obs_converted = env.convert_observation(obs)
        env.SLAM(obs_converted)
        print(f"SLAM updated with observation: {obs_converted}")
        