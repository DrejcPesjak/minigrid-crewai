import re
from collections import deque
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Set

class Agent:

    def __init__(self):
        self.mission: str = ''
        self.current_dir: str = ''
        self.current_observation: Optional[np.ndarray] = None
        self.full_grid: Optional[np.ndarray] = None
        self.previous_actions: List[int] = []
        self.inventory: Optional[str] = None

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

    def _agent_coords(self) -> tuple[int, int]:
        """Return (row, col) of the agent inside full_grid."""
        ((r,), (c,)) = np.where(self.full_grid == 'agent')
        return (int(r), int(c))

    def _front_coords(self) -> tuple[int, int]:
        """Cell directly in front of the agent."""
        DIR = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        (r, c) = self._agent_coords()
        (dr, dc) = DIR[self.current_dir]
        return (r + dr, c + dc)

    def _parse_name(self, name: str) -> str:
        """Convert a name like 'door_red_locked' to "door red locked".
        Convert snake_case or kebab-case to space-separated words.
        Also removes numbers.
        """
        name = re.sub('[_-]', ' ', name)
        name = re.sub('\\d+', '', name)
        name = name.strip()
        return name

    def am_next_to(self, obj: str) -> bool:
        (fr, fc) = self._front_coords()
        (R, C) = self.full_grid.shape
        return 0 <= fr < R and 0 <= fc < C and (self.full_grid[fr, fc] == obj)

    def lava_ahead(self) -> bool:
        (fr, fc) = self._front_coords()
        (R, C) = self.full_grid.shape
        return 0 <= fr < R and 0 <= fc < C and (self.full_grid[fr, fc] == 'lava')

    def safe_forward_infinite(self):
        """Warning: this is an infinite loop! Do not call directly."""
        while not self.lava_ahead():
            yield from self.move_forward()

    def pick_up_obj(self, obj: str) -> List[int]:
        if self.am_next_to(obj):
            return self.pick_up()
        return []

    def _cell_passable(self, r: int, c: int) -> bool:
        """Return True if the agent can step onto the cell at (r, c)."""
        (R, C) = self.full_grid.shape
        if not (0 <= r < R and 0 <= c < C):
            return False
        cell = self.full_grid[r, c]
        parts = cell.split()
        obj = parts[0]
        if obj == 'door':
            state = parts[-1]
            return state == 'open'
        return obj in ('empty', 'floor', 'goal', 'agent')

    def reach_goal(self, a: str, g: str) -> List[int]:
        """Navigate the agent to the specified goal object g."""
        target = self._parse_name(g)
        (rows, cols) = np.where(self.full_grid == target)
        if rows.size == 0:
            raise RuntimeError(f"reach_goal: target '{target}' not found in full_grid")
        goal = (int(rows[0]), int(cols[0]))
        start = self._agent_coords()
        frontier = deque([start])
        came_from: Dict[tuple[int, int], Union[tuple[int, int], None]] = {start: None}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while frontier:
            current = frontier.popleft()
            if current == goal:
                break
            for (dr, dc) in directions:
                nbr = (current[0] + dr, current[1] + dc)
                if nbr not in came_from and self._cell_passable(nbr[0], nbr[1]):
                    came_from[nbr] = current
                    frontier.append(nbr)
        if goal not in came_from:
            raise RuntimeError(f"reach_goal: no path to goal '{target}'")
        path: List[tuple[int, int]] = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        if len(path) <= 1:
            return []
        dirs = ['East', 'South', 'West', 'North']
        vectors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        cur_dir_idx = dirs.index(self.current_dir)
        prims: List[int] = []
        cur_pos = start
        for next_pos in path[1:]:
            dr = next_pos[0] - cur_pos[0]
            dc = next_pos[1] - cur_pos[1]
            try:
                desired_idx = vectors.index((dr, dc))
            except ValueError:
                raise RuntimeError(f'reach_goal: invalid step from {cur_pos} to {next_pos}')
            delta = (desired_idx - cur_dir_idx) % 4
            if delta == 1:
                prims.extend(self.turn_right())
            elif delta == 2:
                prims.extend(self.turn_right())
                prims.extend(self.turn_right())
            elif delta == 3:
                prims.extend(self.turn_left())
            prims.extend(self.move_forward())
            cur_dir_idx = desired_idx
            cur_pos = next_pos
        return prims