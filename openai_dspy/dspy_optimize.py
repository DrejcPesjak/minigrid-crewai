import yaml
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

class RobotPlanningSignature(dspy.Signature):
    # """
    # Generate a plan with reasoning and a valid action for the current state.
    # This signature now includes the system and task instructions for full context.
    # """
    # system_message: str = dspy.InputField(desc="Full system instructions")
    # task_message: str = dspy.InputField(desc="Detailed task instructions")
    mission: str = dspy.InputField(desc="Current mission or goal")
    observation: str = dspy.InputField(desc="Readable description of the current observation")
    observation_history: str = dspy.InputField(desc="Recent observations for context")

    reasoning: str = dspy.OutputField(desc="Chain-of-thought reasoning for the decision")
    action: int = dspy.OutputField(desc="The selected action number (0-6)")


def load_config(filename):
    # Load agent configuration from YAML (agents_config.yaml)
    with open(filename, "r") as f:
        agent_config = yaml.safe_load(f)["agents"]["main_agent"]

    system_message = agent_config["system_message"]
    task_message = agent_config["task_message"]
    task_message = task_message.split("Mission:")[0] 

    return system_message, task_message

def load_dataset(filename):
    # Load training data recorded during human play
    with open(filename, "r") as f:
        training_data = yaml.safe_load(f)

    trainset = []
    for example in training_data:
        ex = dspy.Example(
            mission=example["mission"],
            observation=example["observation"],
            observation_history=example["observation_history"],
            action=example["action"]
        ).with_inputs("mission", "observation", "observation_history")
        trainset.append(ex)

    print(f"Loaded {len(trainset)} training examples.")
    return trainset


def planning_metric(truth, prediction, trace=None):
    # Here, simply check if the predicted action matches the recorded action.
    return truth.action == prediction.action


if __name__ == "__main__":
    # LOAD LLM -------------
    lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
    dspy.configure(lm=lm)
    # lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
    # dspy.configure(lm=lm)

    # SET INSTRUCTIONS -------------
    system_message, task_message = load_config("agents_config.yaml")
    RobotPlanningSignature.__doc__ = f"""
    Generate a plan with reasoning and a valid action for the current state.\n System Message: {system_message} \n Task Message: {task_message}
    """

    # LOAD DATASET -------------
    # cat optimal_path_sim/*.yaml > human_play_trainset_combined.yaml
    dataset = load_dataset("human_play_trainset_combined.yaml")
    # DO NOT SHUFFLE, as the examples are taken from a sequential game play
    # split=0.83 # 197 * 0.83 = 163, 164 has empty observation_history aka is new sequence
    split = 0.803 # 472 * 0.803 = 379
    split_index = int(len(dataset) * split)
    trainset = dataset[:split_index]
    testset = dataset[split_index:]
    print(f"Training set size: {len(trainset)}")
    print(f"Testing set size: {len(testset)}")
    # should split at where observation_history==[]

    # MAIN MODULE -------------
    # Create the planning module using a chain-of-thought approach.
    planner_module = dspy.ChainOfThought(RobotPlanningSignature)

    # Evaluate the innitial planner on testset
    evaluator = Evaluate(devset=testset, num_threads=1, display_progress=True, display_table=5)
    evaluator(planner_module, metric=planning_metric)


    # OPTIMIZATION -------------
    # optimizer = MIPROv2(metric=planning_metric, max_bootstrapped_demos=5, max_labeled_demos=5)
    optimizer = dspy.MIPROv2(metric=planning_metric, auto="heavy", num_threads=24)#, verbose=True)
    optimized_planner = optimizer.compile(planner_module, trainset=trainset)

    print("Optimized planner module:", repr(optimized_planner))
    timestamp = dspy.dsp.utils.utils.timestamp()
    optimized_planner.save(f"opti_model_{timestamp}.json")

    evaluator = Evaluate(devset=testset, num_threads=1, display_progress=True, display_table=5)
    evaluator(optimized_planner, metric=planning_metric)




