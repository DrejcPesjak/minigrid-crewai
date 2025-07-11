import re
from collections import deque
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Set

class Agent():
    def __init__(self):
        # Mission: a string describing the task
        self.mission: str = ""
        # Orientation: one of "East", "South", "West", "North"
        self.current_dir: str = ""
        # Local view & global map just placeholders (updated externally)
        self.current_observation: Optional[np.ndarray] = None
        self.full_grid: Optional[np.ndarray] = None
        # Tracks all primitive action codes executed
        self.previous_actions: List[int] = []
        # Inventory holds at most one object name
        self.inventory: Optional[str] = None

    # ---------- primitive actions ----------
    def turn_left(self) -> List[int]:
        return [0]
    
    def turn_right(self) -> List[int]:
        return [1]
    
    def move_forward(self) -> List[int]:
        return [2]
    
    def pick_up(self) -> List[int]:
        return [3]
    
    def drop(self) -> List[int]:
        return [4]
    
    def toggle(self) -> List[int]:
        return [5]
    
    def done(self) -> List[int]:
        return [6]
    
    # ---------- helpers ----------
    def _agent_coords(self) -> tuple[int, int]:
        """Return (row, col) of the agent inside full_grid."""
        (r,), (c,) = np.where(self.full_grid == "agent")
        return int(r), int(c)

    def _front_coords(self) -> tuple[int, int]:
        """Cell directly in front of the agent."""
        DIR = {
            "East":  ( 0,  1), 
            "South": ( 1,  0), 
            "West":  ( 0, -1), 
            "North": (-1,  0)
        }
        r, c = self._agent_coords()
        dr, dc = DIR[self.current_dir]
        return r + dr, c + dc
    
    def _parse_name(self, name: str) -> str:
        """Convert a name like 'door_red_locked' to "door red locked".
        Convert snake_case or kebab-case to space-separated words.
        Also removes numbers.
        """
        name = re.sub(r"[_-]", " ", name)
        name = re.sub(r"\d+", "", name)  # remove numbers
        name = name.strip()  # remove leading/trailing spaces
        return name

    # ---------- predicates ----------
    def am_next_to(self, obj: str) -> bool:
        fr, fc = self._front_coords()
        R, C = self.full_grid.shape
        return 0 <= fr < R and 0 <= fc < C and self.full_grid[fr, fc] == obj

    def lava_ahead(self) -> bool:
        fr, fc = self._front_coords()
        R, C = self.full_grid.shape
        return 0 <= fr < R and 0 <= fc < C and self.full_grid[fr, fc] == "lava"
    
    # ---------- new actions ----------
    def safe_forward_infinite(self):
        """Warning: this is an infinite loop! Do not call directly."""
        while not self.lava_ahead():
            yield from self.move_forward()
    
    def pick_up_obj(self, obj: str)  -> List[int]:
        if self.am_next_to(obj):
            return self.pick_up()
        return []
        
