# Code generation LLM
from llmclient import ChatGPTClient
from pydantic import BaseModel
from minigrid import MiniGridEnv

class CoderResponse(BaseModel):
    code: str

class CodeGeneratorLLM:
    def __init__(self):
        self.client = ChatGPTClient(
            model_name="o1",
            output_format=CoderResponse
        )
    
    def get_agent_code():
        # load agent.py
        pass

    def generate_code():
        # only new code : predicates (simple if func), and actions
        # provided previous agent code, and pddl for 1 new action, (and error if this is not first loop iter)
        # return code
        pass
    
    def append_code_tmp():
        # append code to agent_tmp.py
        pass

    def save_code():
        # save code from agent_tmp.py to agent.py
        pass

    def main():
        # prompt="prev code, pddl action"
        # loop X times
        #     generate_code(prompt)
        #     append_code_tmp()
        #     err = MiniGridEnv(agent_tmp.py, level_name).run_sim(action_sequence)
        #     if err == "success":
        #         save_code()
        #     else:
        #         # err is [code error] or [env goal not reached]
        #         prompt="fix code: prev code, pddl action, new code, err"
        pass


        

    
