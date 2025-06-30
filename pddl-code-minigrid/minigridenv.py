import time
import numpy as np
np.set_printoptions(linewidth=200)
import gymnasium as gym
import agent_tmp

class MiniGridEnv:
    def __init__(self, level_name: str= "MiniGrid-Empty-5x5-v0"):
        self.agent = agent_tmp.Agent()
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
            print(self.level_name)
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
        self.agent.mission            = obs['mission']

        # agent_state = {
        #     "current_dir": self.agent.current_dir,
        #     "current_observation": self.agent.current_observation,
        #     "full_grid": self.agent.full_grid,
        #     "previous_actions": self.agent.previous_actions,
        #     "inventory": self.agent.inventory,
        # }
        # print(agent_state)
        # print(self.agent.full_grid)
    

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

        # --- forward blocked? -----------------------------------------
        def _is_passable_str(cell_str: str) -> bool:
            """
            True ↦ the agent can occupy this tile
            """
            if cell_str is None:
                return False
            words = cell_str.split()
            obj = words[0] if words else "unseen"

            if obj in {"unseen", "empty", "floor", "goal", "lava"}:
                return True
            if obj == "door":
                return "open" in words          # only open doors are walkable
            return False
        
        blocked = False
        if action == 2:
            pr, pc = {"East":(prev_obs.shape[0]//2,1),
                    "West":(prev_obs.shape[0]//2,prev_obs.shape[1]-2),
                    "South":(1,prev_obs.shape[1]//2),
                    "North":(prev_obs.shape[0]-2,prev_obs.shape[1]//2)}[prev_dir]
            blocked = not _is_passable_str(prev_obs[pr, pc])

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
    
    def _map_without_agent(self, grid: np.ndarray) -> np.ndarray:
        g = grid.copy()
        g[g == "agent"] = self.prev_underlying
        return g
   
    def _two_cycle(self, actions: list, min_actions:int=10) -> bool:
        """Only few specific patterns are considered cycles"""
        if len(actions) < min_actions or min_actions % 2:
            return False
        # 0) Half-length cycle full (if first half == second half)
        if len(actions) % 2 == 0:
            l = len(actions)
            if actions[:l//2] == actions[l//2:l]:
                return True
        window = actions[-min_actions:]
        # 1) Half-length cycle (e.g., XXXXXYYYYY)
        half = min_actions // 2
        if window[:half] == window[half:]:
            return True
        # 2) 2-element cycle (e.g., ABABABABAB)
        return all(window[i] == window[i % 2] for i in range(min_actions))

    def run_sim(self, action_sequence: list):
        # Simulate the environment with the given agent code and action sequence
        # Return "success" if the goal is reached, otherwise return an error message

        try:
            # print(action_sequence)
            calls = self.parse_actions(action_sequence)
            # print(calls)
            # print(dir(self.agent))
            agent_state = vars(self.agent)
            grid_history = []
            # history_limit = 6 # allowed 4 turns plus 1 forward move, so 5 actions in total
            history_limit = 15 # allows backtracking

            for method_name, args in calls:
                if not hasattr(self.agent, method_name):
                    raise AttributeError(f"No Agent method {method_name}")
                # run the high-level action
                codes = getattr(self.agent, method_name)(*args)  # -> List[int]
                # print(codes)
                # print(f"→ {method_name}({', '.join(args)}) returned {codes}")
                # actually execute those primitive codes in your env
                for code in codes:
                    obs, reward, terminated, truncated, info = self.play_episode(code)

                    # ------------- for debugging purposes ------------- 
                    # a = {
                    #     # "current_observation": self.agent.current_observation,
                    #     # "full_grid": self.agent.full_grid,
                    #     "current_dir": self.agent.current_dir,
                    #     "previous_actions": self.agent.previous_actions,
                    #     "inventory": self.agent.inventory,
                    # }
                    # print(a)
                    # import pandas as pd
                    # df = pd.DataFrame(self.agent.full_grid)
                    # print(df.to_string(index=False, header=False))
                    time.sleep(0.2) # for demo purposes
                    # ---------------------------------------------------

                    agent_state = {
                        "mission": self.agent.mission,
                        "current_dir": self.agent.current_dir,
                        "current_observation": self.agent.current_observation,
                        "full_grid": self.agent.full_grid,
                        "previous_actions": self.agent.previous_actions,
                        "inventory": self.agent.inventory,
                    }

                    # for determining if the agent is stuck
                    if len(grid_history) >= history_limit:
                        grid_history.pop(0)
                    grid_history.append(self._map_without_agent(self.agent.full_grid))
                    # without agent, if he walks in circles, the grid will not change

                    if (terminated or truncated):
                        if reward > 0:
                            return "success"
                        else:
                            return f"Failed with reward {reward}, last action {code}, agent state {agent_state}"
                        
                    #stuck: # 5 actions and still at the same place , aka grid_history the same
                    elif (len(grid_history) >= history_limit and \
                        all(np.array_equal(grid_history[-1], grid) for grid in grid_history)) or \
                        (self._two_cycle(self.agent.previous_actions, min_actions=10) and \
                        all(g.shape == grid_history[-1].shape for g in grid_history)):
                        # if all grids are the same as the last one, then they are all the same
                        return f"Agent is stuck, last action {code}, agent state {agent_state}"
            
            return f"Executed all actions successfully, but did not reach the goal. Last agent state: {agent_state}"
    
        except Exception as e:
            return f"Error: {str(e)}"
        
        finally:
            # Clean up the environment
            self.end_env()
            print("Environment closed.")
            self.agent = None


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
    # env = MiniGridEnv("MiniGrid-LavaCrossingS11N5-v0") # solved
    # action_sequence = "[cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava(), cross-lava()]"
    # result = env.run_sim(action_sequence)
    # print(result)
    # env.end_env()

    # first_set = [
    #     "MiniGrid-Empty-5x5-v0",
    #     "MiniGrid-Empty-Random-5x5-v0",
    #     "MiniGrid-Empty-6x6-v0",
    #     "MiniGrid-Empty-Random-6x6-v0",
    #     "MiniGrid-Empty-8x8-v0",
    #     "MiniGrid-Empty-16x16-v0",
    #     # "MiniGrid-FourRooms-v0",
    #     # "BabyAI-GoToLocalS6N4-v0",
    #     # "BabyAI-GoToLocalS6N2-v0",
    #     # "BabyAI-MiniBossLevel-v0"
    # ]
    # for level_name in first_set:
    #     print(f"Running level: {level_name}")
    #     env = MiniGridEnv(level_name)
    #     # action_sequence = "[move_to_goal(agent1, goal1)]"
    #     # action_sequence = "[turn-left(), move-forward(), turn-right(), move-forward(), move-forward(), move-forward(), move-forward()]"
    #     action_sequence = "[move-forward(), move-forward(), move-forward(),move-forward(),move-forward(),move-forward(),move-forward(),move-forward(),move-forward(),move-forward(), turn-right(), move-forward(), move-forward()]"
    #     # action_sequence = "[move-to-goal-v5(a,t)]"
    #     result = env.run_sim(action_sequence)
    #     print(result)
    #     env.end_env()
    
    from pathlib import Path
    import json
    CURRIC_FILE = Path(__file__).with_name("merged_curriculum2.json")
    curriculum = json.loads(CURRIC_FILE.read_text())
    i = 0
    for cat in curriculum:
        if cat["category_name"] == "open_door":
            for lvl in cat["levels"]:
                for env_name in lvl["configs"]:
                    print(f"\n=== {env_name} ===")
                    i += 1
                    env = MiniGridEnv(env_name)
                    action_sequence = "[move-forward()]"
                    result = env.run_sim(action_sequence)
                    # print(result)
                    env.end_env()