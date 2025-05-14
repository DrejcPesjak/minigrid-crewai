from agent import Agent

# self.env = gym.make(env_name, render_mode="human")
# obs, info = self.env.reset()
# # Modify observation for demonstration purposes
# obs['image'][3][6][0] = 10  
# self.state.observation = obs
# self.state.mission = obs['mission']
# cnv_obs = convert_observation(self.state.observation)


# def execute_actions():
#     agent = Agent()
#     state = "idk"
#     actions = [
#         agent.move_forward(),
#         agent.move_forward(),
#         agent.turn_right(),
#         agent.move_forward(),
#         agent.move_forward()
#     ]
#     for action in actions:
        
#         obs, reward, terminated, truncated, info = env.step(self.state.last_action)
        

def main():
    # take first mission type "get to the green goal square"
    # take Agent signiture
    # PlannerLLM(mission, agent_signature) > domain, problem, result.plan
    # for each action in result.plan not in agent_signature:
        # CoderLLM(action) > updated_agent_code
        #  - inside coder: we infinite loop new_code->test.env->error->llm
    pass