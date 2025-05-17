import time
import numpy as np
np.set_printoptions(linewidth=200)
import gymnasium as gym
from agent_tmp import Agent

class MiniGridEnv:
    def __init__(self, level_name: str= "MiniGrid-Empty-5x5-v0"):
        self.agent = Agent()
        self.level_name = level_name
        self.env = None
        # Track agent global position
        self.agent_pos = None  # (row, col) in full_grid
        # Underlying cell at agent (for cleaning up agent marker)
        self.prev_underlying = None
        self.start_env()

    def start_env(self):
        # Initialize the environment
        self.env = gym.make(self.level_name, render_mode="human")
        obs, info = self.env.reset(seed=42) # later will be random
        obs['image'][3][6][0] = 10
        obs_converted = self.convert_observation(obs)
        self.init_full_grid(obs_converted)        
    
    def end_env(self):
        # Close the environment
        if self.env:
            self.env.close()
        self.env = None
        self.agent = None
    
    def convert_observation(self, input_dict):
        # Convert the observation to a format suitable for the agent

        # Define mappings
        COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
        IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

        OBJECT_TO_IDX = {
            "unseen": 0,
            "empty": 1,
            "wall": 2,
            "floor": 3,
            "door": 4,
            "key": 5,
            "ball": 6,
            "box": 7,
            "goal": 8,
            "lava": 9,
            "agent": 10,
        }
        IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

        STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}
        IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

        # Extract components
        image = input_dict['image']
        direction_idx = input_dict['direction']
        mission = input_dict['mission']

        # Corrected direction mapping
        DIRECTION_MAP = {0: "East", 1: "South", 2: "West", 3: "North"}
        # DIRECTION_ARROW = {0: ">", 1: "v", 2: "<", 3: "^"}
        # DIRECTION_ARROW = {0: "right", 1: "down", 2: "left", 3: "up"}
        direction = DIRECTION_MAP.get(direction_idx, "Unknown")

        # Convert image into object_color_state format
        grid = []
        for row in image:
            grid_row = []
            for cell in row:
                object_idx, color_idx, state_idx = cell
                object_name = IDX_TO_OBJECT.get(object_idx, "unknown")
                color_name = IDX_TO_COLOR.get(color_idx, "unknown")
                state_name = IDX_TO_STATE.get(state_idx, "unknown")
                c = ""
                if object_name in ["door","key","ball","box", "goal"]:
                    c = f" {color_name}"
                s = ""
                if object_name in ["door"]:
                    s = f" {state_name}"
                # if object_name == "agent":
                #     object_name += f" {DIRECTION_ARROW[direction_idx]}"
                grid_row.append(f"{object_name}{c}{s}")
            grid.append(grid_row)

        # Flip the grid 
        if direction == "East":
            grid = [row[::-1] for row in grid]
        elif direction == "West":
            # flip the grid upside down
            grid = [row for row in grid[::-1]]
        elif direction == "North":
            grid = list(zip(*grid))
        elif direction == "South":
            grid = list(zip(*grid[::-1]))[::-1]
        
        # Add "Row i" to each row
        # grid2 = [f"Row {i}: " + ", ".join(row) for i, row in enumerate(grid)]
        # grid_string = " \n ".join(grid2)

        # # Convert grid to a string with newlines for better visualization
        # grid_string = " \n ".join(" ".join(cell for cell in row) for row in grid)

        # Create the formatted output
        formatted_observation = {
            "mission": mission,
            "direction": direction,
            "grid": grid,  # 2D list of strings
            # "observation_grid_string": grid_string,  # String representation of the grid
        }

        return formatted_observation
    
    def init_full_grid(self, obs: dict):
        newg          = np.array([list(r) for r in obs['grid']], dtype=object)
        h, w          = newg.shape
        orient        = obs['direction']          # "East" | "South" | "West" | "North"

        # local agent coordinates inside 7×7 view
        loc_pos = {
            "East":  (h // 2, 0),
            "South": (0,        w // 2),
            "West":  (h // 2, w - 1),
            "North": (h - 1,    w // 2)
        }
        la_r, la_c = loc_pos[orient]

        # global map starts exactly as the first view
        self.agent.full_grid = np.full((h, w), "unseen", dtype=object)
        mask                 = newg != "unseen"
        self.agent.full_grid[mask] = newg[mask]

        # store agent position & ground-under-feet
        self.agent_pos       = (la_r, la_c)
        self.prev_underlying = "empty"
        self.agent.full_grid[la_r, la_c] = "agent"

        # sync agent state
        self.agent.current_observation = newg
        self.agent.current_dir         = orient

        # agent_state = {
        #     "current_dir": self.agent.current_dir,
        #     "current_observation": self.agent.current_observation,
        #     "full_grid": self.agent.full_grid,
        #     "previous_actions": self.agent.previous_actions,
        #     "inventory": self.agent.inventory,
        # }
        # print(agent_state)
        print(self.agent.full_grid)
    

    def SLAM(self, obs: dict, action: int):
        dir_vec = {"East": (0, 1), "South": (1, 0), "West": (0, -1), "North": (-1, 0)}
        left    = {"East": "North", "North": "West", "West": "South", "South": "East"}
        right   = {v: k for k, v in left.items()}

        # bookkeeping
        self.agent.previous_actions.append(action)
        prev_obs   = self.agent.current_observation
        prev_dir   = self.agent.current_dir
        prev_r, prev_c = self.agent_pos
        H, W       = self.agent.full_grid.shape

        # heading after this action
        new_dir = right[prev_dir] if action == 1 else left.get(prev_dir, prev_dir) if action == 0 else prev_dir
        dr, dc  = dir_vec[new_dir]

        # new 7×7 view
        newg = np.asarray([list(r) for r in obs["grid"]], dtype=object)
        h, w = newg.shape
        loc_pos = {"East":(h//2,0), "South":(0,w//2), "West":(h//2,w-1), "North":(h-1,w//2)}
        la_r, la_c = loc_pos[new_dir]

        # forward blocked?
        blocked = False
        if action == 2:
            pr, pc = {"East":(prev_obs.shape[0]//2,1),
                    "West":(prev_obs.shape[0]//2,prev_obs.shape[1]-2),
                    "South":(1,prev_obs.shape[1]//2),
                    "North":(prev_obs.shape[0]-2,prev_obs.shape[1]//2)}[prev_dir]
            blocked = (prev_obs[pr, pc] == "wall")

        # remove old agent marker
        self.agent.full_grid[prev_r, prev_c] = self.prev_underlying

        # tentative global pose
        new_r, new_c = (prev_r + dr, prev_c + dc) if (action == 2 and not blocked) else (prev_r, prev_c)

        # padding check
        min_r = new_r - la_r;           max_r = new_r + (h-1-la_r)
        min_c = new_c - la_c;           max_c = new_c + (w-1-la_c)
        pad_top    = max(0, -min_r)
        pad_left   = max(0, -min_c)
        pad_bottom = max(0, max_r - (H-1))
        pad_right  = max(0, max_c - (W-1))
        if pad_top or pad_left or pad_bottom or pad_right:
            self.agent.full_grid = np.pad(self.agent.full_grid,
                                        ((pad_top, pad_bottom),(pad_left,pad_right)),
                                        constant_values="unseen")
            new_r += pad_top; new_c += pad_left
            prev_r += pad_top; prev_c += pad_left

        # ------------ INVENTORY SNAPSHOT (OLD) -------------
        front_r, front_c = new_r + dr, new_c + dc
        if 0 <= front_r < self.agent.full_grid.shape[0] and 0 <= front_c < self.agent.full_grid.shape[1]:
            before_front = self.agent.full_grid[front_r, front_c]
        else:
            before_front = "unseen"

        # ------------ STAMP LOCAL VIEW ---------------------
        for i in range(h):
            for j in range(w):
                if newg[i,j] in ("unseen", "agent"):      # keep unknowns & skip agent
                    continue
                gr = new_r + (i-la_r)
                gc = new_c + (j-la_c)
                self.agent.full_grid[gr, gc] = newg[i, j]

        # ------------ INVENTORY AFTER STAMP ---------------
        after_front = self.agent.full_grid[front_r, front_c] if (
            0 <= front_r < self.agent.full_grid.shape[0] and 0 <= front_c < self.agent.full_grid.shape[1]) else "unseen"

        if action == 3:                               # pick-up
            if (before_front not in ("empty","floor","wall","unseen") and
                    after_front   in ("empty","floor")):
                self.agent.inventory = before_front
        elif action == 4:                             # drop
            if (self.agent.inventory and
                    before_front in ("empty","floor") and
                    after_front not in ("empty","floor")):
                self.agent.inventory = None

        # finalise pose & vars
        self.prev_underlying               = self.agent.full_grid[new_r, new_c]
        self.agent.full_grid[new_r, new_c] = "agent"
        self.agent_pos                     = (new_r, new_c)
        self.agent.current_observation     = newg
        self.agent.current_dir             = new_dir

    
    def parse_actions(self, actions_str: str):
        # Parse the action string and convert it to a list of Agent function calls
        # Example: "[move-forward(), turn-left()]" -> ["agent.move_forward()", "agent.turn_left()"]
        import re
        pattern = re.compile(r'([a-zA-Z][\w-]*)\s*\(\s*([^)]*?)\s*\)')
        calls = []
        for m in pattern.finditer(actions_str):
            name = m.group(1).replace('-', '_')
            args = [a.strip() for a in m.group(2).split(',')] if m.group(2) else []
            calls.append((name, args))
        return calls

    def play_episode(self, action: int):
        # Execute one step in the environment with the given action
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs['image'][3][6][0] = 10

        obs_converted = self.convert_observation(obs)
        self.SLAM(obs_converted, action)

        return obs, reward, terminated, truncated, info

    def run_sim(self, action_sequence: list):
        # Simulate the environment with the given agent code and action sequence
        # Return "success" if the goal is reached, otherwise return an error message

        try:
            calls = self.parse_actions(action_sequence)

            for method_name, args in calls:
                if not hasattr(self.agent, method_name):
                    raise AttributeError(f"No Agent method {method_name}")
                # run the high-level action
                codes = getattr(self.agent, method_name)(*args)  # -> List[int]
                print(f"→ {method_name}({', '.join(args)}) returned {codes}")
                # actually execute those primitive codes in your env
                for code in codes:
                    obs, reward, terminated, truncated, info = self.play_episode(code)
                    agent_state = {
                        "current_observation": self.agent.current_observation,
                        # "full_grid": self.agent.full_grid,
                        "current_dir": self.agent.current_dir,
                        "previous_actions": self.agent.previous_actions,
                        "inventory": self.agent.inventory,
                    }
                    # print(agent_state)
                    import pandas as pd
                    df = pd.DataFrame(self.agent.full_grid)
                    print(df.to_string(index=False, header=False))
                    # print(self.agent.full_grid)
                    print(self.agent.full_grid.shape)
                    print(self.agent.current_observation)
                    print(self.agent.current_dir)
                    print(self.agent.previous_actions)
                    print(self.agent.inventory)
                    time.sleep(2) # for demo purposes
                    if (terminated or truncated):
                        if reward > 0:
                            return "success"
                        else:
                            agent_state = {
                                "current_dir": self.agent.current_dir,
                                "current_observation": self.agent.current_observation,
                                "full_grid": self.agent.full_grid,
                                "previous_actions": self.agent.previous_actions,
                                "inventory": self.agent.inventory,
                            }
                            return f"Failed with reward {reward}, last action {code}, agent state {agent_state}"
                    #elif stuck: # 5 actions and still at the same place
                    #     return f"stuck...."
    
        except Exception as e:
            return f"Error: {str(e)}"


if __name__ == "__main__":
    # env = MiniGridEnv("MiniGrid-Empty-5x5-v0")
    # action_sequence = "[move-forward(), move-forward(), turn-right(), move-forward(), move-forward()]"
    # env = MiniGridEnv("MiniGrid-Empty-5x5-v0") # solved
    # action_sequence = "[turn-right(), move-forward(), move-forward(), move-forward(), pick-up(), turn-left(), move-forward(), move-forward()]"
    # env = MiniGridEnv("MiniGrid-MemoryS11-v0") # solved
    # action_sequence = "[turn-right(), turn-right(), turn-right(), turn-right(), move-forward(), move-forward(), move-forward(), move-forward(), move-forward(), move-forward(), move-forward(), move-forward(), move-forward(), move-forward(), turn-left(), move-forward(), move-forward(), move-forward(), pick-up()]"
    # env = MiniGridEnv("BabyAI-UnlockPickup-v0") # solved
    # action_sequence = "[turn-left(), move-forward(), pick-up(), turn-left(), move-forward(), move-forward(), turn-right(), move-forward(), toggle(), move-forward(), move-forward(), move-forward(), move-forward(), drop(), turn-right(), move-forward(), move-forward(), move-forward(), turn-left(), pick-up()]"
    # env = MiniGridEnv("MiniGrid-LavaGapS6-v0") # solved
    # env = MiniGridEnv("MiniGrid-LavaCrossingS9N2-v0") # solved
    env = MiniGridEnv("MiniGrid-LavaCrossingS11N5-v0") # solved
    action_sequence = "[cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava()]"
    result = env.run_sim(action_sequence)
    print(result)
    env.end_env()