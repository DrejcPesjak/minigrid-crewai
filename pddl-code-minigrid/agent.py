import numpy as np
from typing import List, Optional

class Agent():
    def __init__(self):
        # Orientation: one of "East", "South", "West", "North"
        self.current_dir: str = ""
        # Local view & global map just placeholders (updated externally)
        self.current_observation: Optional[np.ndarray] = None
        self.full_grid: Optional[np.ndarray] = None
        # Tracks all primitive action codes executed
        self.previous_actions: List[int] = []
        # Inventory holds at most one object name
        self.inventory: Optional[str] = None
    
    # ---------- helpers ----------
    def _agent_coords(self) -> tuple[int, int]:
        """Return (row, col) of the agent inside current_observation."""
        R, C = self.current_observation.shape
        return {
            "East":  (R // 2, 0),         # left-middle
            "South": (0,       C // 2),   # top-middle
            "West":  (R // 2, C - 1),     # right-middle
            "North": (R - 1,   C // 2),   # bottom-middle
        }[self.current_dir]

    def _front_coords(self) -> tuple[int, int]:
        """Cell directly in front of the agent."""
        r, c = self._agent_coords()
        dr, dc = {
            "East":  (0,  1),
            "South": (1,  0),
            "West":  (0, -1),
            "North": (-1, 0),
        }[self.current_dir]
        return r + dr, c + dc

    # ---------- predicates ----------
    def am_next_to(self, obj: str) -> bool:
        fr, fc = self._front_coords()
        R, C = self.current_observation.shape
        return 0 <= fr < R and 0 <= fc < C and self.current_observation[fr, fc] == obj

    def lava_ahead(self) -> bool:
        fr, fc = self._front_coords()
        R, C = self.current_observation.shape
        return 0 <= fr < R and 0 <= fc < C and self.current_observation[fr, fc] == "lava"

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
    
    # ---------- new actions ----------
    def safe_forward(self):
        if not self.lava_ahead():
            return self.move_forward()
        return []
    
    def pick_up_obj(self, obj: str)  -> List[int]:
        if self.am_next_to(obj):
            return self.pick_up()
        return []
        
