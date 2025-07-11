"""
MiniGridEnv wrapper with
• SLAM-like global map
• primitive-action checkpoint & replay
• rich Outcome status class for the new SPA loop
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import numpy as np

import agent_tmp                                               # hot-reloaded from main

np.set_printoptions(linewidth=200)

# --------------------------------------------------------------------------- #
#  OUTCOME DATACLASS
# --------------------------------------------------------------------------- #

@dataclass
class Outcome:
    status      : str           # success | goal_not_reached | stuck | …
    msg         : str           # human-readable one-liner
    agent_state : dict          # final state snapshot
    trace       : str = ""      # optional traceback for errors


# --------------------------------------------------------------------------- #
#  ENV  WRAPPER
# --------------------------------------------------------------------------- #

class MiniGridEnv:
    """Owns the live Gym env, the Agent instance and the checkpoint logic."""

    # ------------- construction / reset ----------------------------------

    def __init__(self, level_name: str, seed: int = 42):
        self.level_name   = level_name
        self.seed         = seed
        self.agent        = agent_tmp.Agent()
        self.env          = None

        self.agent_pos        : Tuple[int, int] | None = None
        self.prev_underlying  : str | None = None
        self._checkpoint      : List[int] = []       # primitive codes executed so far

        self.start_env()

    def start_env(self):
        self.env = gym.make(self.level_name, render_mode="human")
        obs, _   = self.env.reset(seed=self.seed)
        obs["image"][3][6][0] = 10                   # cosmetic marker for agent pos
        obs_conv = self.convert_observation(obs)
        self.init_full_grid(obs_conv)

    def end_env(self):
        if self.env:
            self.env.close()
        self.env   = None
        self.agent = None

    # ------------- snapshot for planner ----------------------------------

    def snapshot(self) -> dict:
        """Return the fields needed by PlannerLLM."""
        def _cells_to_objs(grid):
            objs = set()
            for row in grid:
                for cell in row:
                    if cell not in ("unseen", "empty", "floor", "agent"):
                        objs.add(cell)      # e.g. "door red closed"
            return sorted(objs)
        snap = {
            "mission"      : self.agent.mission,
            "direction"    : self.agent.current_dir,
            "inventory"    : self.agent.inventory,
            "visible_grid" : self.agent.full_grid.tolist(),
            "visible_objects": _cells_to_objs(self.agent.current_observation),
        }
        return snap

    # ------------- checkpoint helpers ------------------------------------

    def save_primitive(self, code: int):
        self._checkpoint.append(code)

    def replay_checkpoint(self):
        """Reset env and fast-forward all previously executed primitive codes."""
        # if len(self._checkpoint) == 0:
        #     print("No checkpoint to replay")
        #     return
        
        print(f"Replaying {len(self._checkpoint)} checkpoint codes")
        saved_codes = self._checkpoint.copy()
        self.__init__(self.level_name, seed=self.seed)    # fresh env + agent
        for c in saved_codes:
            self.play_episode(c)                          # ignore rewards etc.
            time.sleep(0.1)  # for demo purposes, remove in production

    # --------------------------------------------------------------------- #
    #  OBS & MAPPING 
    # --------------------------------------------------------------------- #

    def convert_observation(self, input_dict):
        COLOR_TO_IDX = {"red":0,"green":1,"blue":2,"purple":3,"yellow":4,"grey":5}
        IDX_TO_COLOR = {v:k for k,v in COLOR_TO_IDX.items()}
        OBJECT_TO_IDX = {
            "unseen":0,"empty":1,"wall":2,"floor":3,"door":4,"key":5,
            "ball":6,"box":7,"goal":8,"lava":9,"agent":10,
        }
        IDX_TO_OBJECT = {v:k for k,v in OBJECT_TO_IDX.items()}
        STATE_TO_IDX = {"open":0,"closed":1,"locked":2}
        IDX_TO_STATE = {v:k for k,v in STATE_TO_IDX.items()}

        image         = input_dict["image"]
        direction_idx = input_dict["direction"]
        mission       = input_dict["mission"]
        DIRECTION_MAP = {0:"East",1:"South",2:"West",3:"North"}
        direction     = DIRECTION_MAP.get(direction_idx,"Unknown")

        grid = []
        for row in image:
            row_out = []
            for obj_idx,col_idx,sta_idx in row:
                obj  = IDX_TO_OBJECT.get(obj_idx,"unknown")
                col  = IDX_TO_COLOR.get(col_idx,"unknown")
                sta  = IDX_TO_STATE.get(sta_idx,"unknown")
                parts = [obj]
                if obj in {"door","key","ball","box","goal"}:
                    parts.append(col)
                if obj == "door":
                    parts.append(sta)
                row_out.append(" ".join(parts))
            grid.append(row_out)

        # rotate so that agent always in centre-bottom
        if direction == "East":
            grid = [r[::-1] for r in grid]
        elif direction == "West":
            grid = grid[::-1]
        elif direction == "North":
            grid = list(zip(*grid))
        elif direction == "South":
            grid = list(zip(*grid[::-1]))[::-1]

        return {"mission":mission, "direction":direction, "grid":grid}

    def init_full_grid(self, obs):
        newg   = np.array([list(r) for r in obs["grid"]], dtype=object)
        h, w   = newg.shape
        orient = obs["direction"]
        loc_pos = {"East":(h//2,0),"South":(0,w//2),
                   "West":(h//2,w-1),"North":(h-1,w//2)}
        la_r,la_c = loc_pos[orient]

        self.agent.full_grid = np.full((h,w),"unseen",dtype=object)
        mask                 = newg != "unseen"
        self.agent.full_grid[mask] = newg[mask]

        self.agent_pos       = (la_r,la_c)
        self.prev_underlying = "empty"
        self.agent.full_grid[la_r,la_c] = "agent"

        self.agent.current_observation = newg
        self.agent.current_dir         = orient
        self.agent.mission             = obs["mission"]

    # ----------------- SLAM -----------------------------------
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

    # ---------------- primitive execution ---------------------------------

    def play_episode(self, action:int):
        obs,reward,terminated,truncated,_ = self.env.step(action)
        obs["image"][3][6][0] = 10
        obs_conv = self.convert_observation(obs)
        self.SLAM(obs_conv, action)
        return obs_conv, reward, terminated, truncated

    # ---------------- utility ---------------------------------------------

    def _agent_state(self) -> dict:
        return {
            "mission"            : self.agent.mission,
            "direction"          : self.agent.current_dir,
            "inventory"          : self.agent.inventory,
            "previous_primitives": self.agent.previous_actions.copy(),
            "full_grid"          : self.agent.full_grid.copy(),
        }

    def _map_without_agent(self, grid: np.ndarray) -> np.ndarray:
        g = grid.copy()
        g[g == "agent"] = self.prev_underlying
        return g

    def _two_cycle(self, actions:list, min_actions:int=10) -> bool:
        if len(actions) < min_actions or min_actions % 2:
            return False
        if len(actions) % 2 == 0 and actions[:len(actions)//2] == actions[len(actions)//2:]:
            return True
        window = actions[-min_actions:]
        half   = min_actions//2
        if window[:half] == window[half:]:
            return True
        return all(window[i] == window[i%2] for i in range(min_actions))

    # --------------------------------------------------------------------- #
    #  PLAN EXECUTION for SPA loop
    # --------------------------------------------------------------------- #

    def parse_actions(self, plan_str:str):
        import re
        pat = re.compile(r'([a-zA-Z][\w-]*)\s*\(\s*([^)]*?)\s*\)')
        for m in pat.finditer(plan_str):
            name = m.group(1).replace("-","_")
            args = [a.strip() for a in m.group(2).split(",")] if m.group(2) else []
            yield name, args

    def run_sim(self, plan_str:str) -> Outcome:
        try:
            calls          = self.parse_actions(plan_str)
            grid_history   : List[np.ndarray] = []
            history_limit  = 15
            
            for meth, args in calls:
                if not hasattr(self.agent, meth):
                    return Outcome("missing_method",
                                   f"Agent has no method {meth}",
                                   self._agent_state())

                try:
                    codes = getattr(self.agent, meth)(*args)
                except Exception as e:
                    return Outcome("syntax_error",
                                   f"{meth} raised {e}",
                                   self._agent_state(),
                                   traceback.format_exc())

                for code in codes:
                    self.save_primitive(code)
                    _, reward, terminated, truncated = self.play_episode(code)

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

                    # progress / stuck detection
                    if len(grid_history) >= history_limit:
                        grid_history.pop(0)
                    grid_history.append(self._map_without_agent(self.agent.full_grid))

                    if terminated or truncated:
                        if reward > 0:
                            return Outcome("success",
                                           "Goal reached!",
                                           self._agent_state())
                        return Outcome("reward_failed",
                                       f"Terminated with reward {reward}",
                                       self._agent_state())

                    if (len(grid_history) >= history_limit and
                        all(np.array_equal(grid_history[-1], g) for g in grid_history)) or \
                        (self._two_cycle(self.agent.previous_actions, 10) and
                         all(g.shape == grid_history[-1].shape for g in grid_history)):
                        return Outcome("stuck",
                                       f"Agent is stuck. No progress after {history_limit} primitives",
                                       self._agent_state())

            # loop finished with no termination
            return Outcome("goal_not_reached",
                           "Plan finished succesfully, but did not reach the goal.",
                           self._agent_state())

        except Exception as e:
            return Outcome("runtime_error",
                           str(e),
                           self._agent_state(),
                           traceback.format_exc())

    # --------------------------------------------------------------------- #
    #  MAIN DEBUG
    # --------------------------------------------------------------------- #

if __name__ == "__main__":
    env = MiniGridEnv("MiniGrid-Empty-5x5-v0", seed=42)
    action_sequence = "[move-forward(), move-forward(), turn-right(), move-forward(), move-forward()]"
    outcome = env.run_sim(action_sequence)
    print(outcome)
    env.end_env()
