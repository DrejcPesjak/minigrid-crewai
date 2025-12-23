import re
from collections import deque
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Set
from typing import Tuple, List

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

    def reach_goal(self, _a: str, g: str):
        """Navigate the agent to the specified goal object g, exploring as needed."""
        target = self._parse_name(g)
        while True:
            (rows, cols) = np.where(self.full_grid == target)
            if rows.size > 0:
                goal_pos = (int(rows[0]), int(cols[0]))
                break
            (cell, unseen) = self._find_frontier()
            yield from self._navigate_to_cell(cell)
            dr = unseen[0] - cell[0]
            dc = unseen[1] - cell[1]
            yield from self._face_direction(dr, dc)
            yield from self.move_forward()
        yield from self._navigate_to_cell(goal_pos)

    def _find_frontier(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Find the nearest cell adjacent to 'unseen', treating out-of-bounds as unseen."""
        start = self._agent_coords()
        (R, C) = self.full_grid.shape
        frontier = deque([start])
        came_from = {start: None}
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while frontier:
            curr = frontier.popleft()
            for (dr, dc) in directions:
                nbr = (curr[0] + dr, curr[1] + dc)
                if not (0 <= nbr[0] < R and 0 <= nbr[1] < C):
                    return (curr, nbr)
                if self.full_grid[nbr[0], nbr[1]] == 'unseen':
                    return (curr, nbr)
            for (dr, dc) in directions:
                nbr = (curr[0] + dr, curr[1] + dc)
                if nbr not in came_from and self._cell_passable(nbr[0], nbr[1]):
                    came_from[nbr] = curr
                    frontier.append(nbr)
        raise RuntimeError('_find_frontier: no frontier found; entire known map is closed or goal unreachable')

    def _face_direction(self, dr: int, dc: int):
        """Rotate the agent to face the direction vector (dr, dc)."""
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir_names = ['East', 'South', 'West', 'North']
        desired = dirs.index((dr, dc))
        cur = dir_names.index(self.current_dir)
        delta = (desired - cur) % 4
        if delta == 1:
            yield from self.turn_right()
        elif delta == 2:
            yield from self.turn_right()
            yield from self.turn_right()
        elif delta == 3:
            yield from self.turn_left()

    def _navigate_to_cell(self, dest: Tuple[int, int]):
        """Pathfind to dest on the known map and execute turns and moves."""
        start = self._agent_coords()
        if start == dest:
            return []
        (R, C) = self.full_grid.shape
        frontier = deque([start])
        came_from = {start: None}
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while frontier:
            curr = frontier.popleft()
            if curr == dest:
                break
            for (dr, dc) in directions:
                nbr = (curr[0] + dr, curr[1] + dc)
                if 0 <= nbr[0] < R and 0 <= nbr[1] < C and (nbr not in came_from) and self._cell_passable(nbr[0], nbr[1]):
                    came_from[nbr] = curr
                    frontier.append(nbr)
        if dest not in came_from:
            raise RuntimeError(f'_navigate_to_cell: no path to {dest}')
        path: List[Tuple[int, int]] = []
        node = dest
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir_names = ['East', 'South', 'West', 'North']
        cur_dir_idx = dir_names.index(self.current_dir)
        cur_pos = start
        for next_pos in path[1:]:
            dr = next_pos[0] - cur_pos[0]
            dc = next_pos[1] - cur_pos[1]
            desired_idx = dirs.index((dr, dc))
            delta = (desired_idx - cur_dir_idx) % 4
            if delta == 1:
                yield from self.turn_right()
            elif delta == 2:
                yield from self.turn_right()
                yield from self.turn_right()
            elif delta == 3:
                yield from self.turn_left()
            yield from self.move_forward()
            cur_dir_idx = desired_idx
            cur_pos = next_pos

    def goto_crossing(self, _a: str):
        """Navigate to the nearest corridor crossing (an intersection or opening in a wall)."""

        def is_crossing(r: int, c: int) -> bool:
            if not self._cell_passable(r, c):
                return False
            dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            passable_dirs: list[tuple[int, int]] = []
            (R, C) = self.full_grid.shape
            for (dr, dc) in dirs:
                (nr, nc) = (r + dr, c + dc)
                if 0 <= nr < R and 0 <= nc < C and self._cell_passable(nr, nc):
                    passable_dirs.append((dr, dc))
            if len(passable_dirs) < 3:
                return False
            for i in range(len(passable_dirs)):
                for j in range(i + 1, len(passable_dirs)):
                    d1 = passable_dirs[i]
                    d2 = passable_dirs[j]
                    if d1[0] != -d2[0] or d1[1] != -d2[1]:
                        return True
            return False
        while True:
            (R, C) = self.full_grid.shape
            crosses = np.array([[is_crossing(r, c) for c in range(C)] for r in range(R)])
            idx = np.argwhere(crosses)
            if idx.size > 0:
                target = tuple(idx[0])
                break
            (cell, unseen) = self._find_frontier()
            yield from self._navigate_to_cell(cell)
            dr = unseen[0] - cell[0]
            dc = unseen[1] - cell[1]
            yield from self._face_direction(dr, dc)
            yield from self.move_forward()
        yield from self._navigate_to_cell(target)

    def goto_goal(self, _a: str):
        """Navigate to the green goal square."""
        yield from self.reach_goal(_a, 'goal_green')