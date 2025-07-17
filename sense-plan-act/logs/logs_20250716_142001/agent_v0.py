import re
from collections import deque
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Set
from typing import List, Optional, Tuple

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

    def is_goal(self, cell: str) -> bool:
        """Return True if the cell string names a goal (any color)."""
        tokens = cell.split()
        return bool(tokens) and tokens[0] == 'goal'

    def _is_passable(self, cell: str) -> bool:
        """Return True if the agent can move into the given cell."""
        tokens = cell.split()
        obj = tokens[0] if tokens else ''
        if obj in ('wall', 'lava', 'unseen'):
            return False
        if obj == 'door' and 'open' not in tokens:
            return False
        return True

    def _shortest_path(self, start: Tuple[int, int], goals: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """Breadth-first search from start to the nearest goal in goals, returns list of coords or None."""
        (R, C) = self.full_grid.shape
        goal_set = set(goals)
        queue = deque([start])
        parent: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        for (r, c) in queue:
            pass
        while queue:
            (r, c) = queue.popleft()
            if (r, c) in goal_set:
                path: List[Tuple[int, int]] = []
                cur = (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1]
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                (nr, nc) = (r + dr, c + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                if (nr, nc) in parent:
                    continue
                cell = self.full_grid[nr, nc]
                if not self._is_passable(cell):
                    continue
                parent[nr, nc] = (r, c)
                queue.append((nr, nc))
        return None

    def _turn_towards(self, target_dir: str) -> List[int]:
        """Return primitive turn actions to rotate from current_dir to target_dir."""
        DIRS = ['East', 'South', 'West', 'North']
        idx = DIRS.index(self.current_dir)
        tidx = DIRS.index(target_dir)
        diff = (tidx - idx) % 4
        if diff == 0:
            return []
        if diff == 1:
            return self.turn_right()
        if diff == 2:
            return self.turn_right() * 2
        return self.turn_left()

    def reach_goal(self):
        """
High-level action: navigate to the mission-specified goal cell (any color filter) and then signal completion.
Implements (:action reach_goal :parameters ()).
"""
        words = re.sub('[^a-z ]', ' ', self.mission.lower()).split()
        COLORS = {'red', 'green', 'blue', 'purple', 'yellow', 'grey'}
        goal_color = None
        for (i, w) in enumerate(words):
            if w == 'goal' and i > 0 and (words[i - 1] in COLORS):
                goal_color = words[i - 1]
                break
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        while True:
            pos = self._agent_coords()
            cur_cell = str(self.full_grid[pos])
            parts = cur_cell.split()
            if parts and parts[0] == 'goal' and (goal_color is None or goal_color in parts):
                yield from self.done()
                return []
            goals: List[Tuple[int, int]] = []
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                toks = str(cell).split()
                if toks and toks[0] == 'goal' and (goal_color is None or goal_color in toks):
                    goals.append((r, c))
            if goals:
                targets = goals
            else:
                (R, C) = self.full_grid.shape
                frontiers: List[Tuple[int, int]] = []
                for ((r, c), cell) in np.ndenumerate(self.full_grid):
                    if str(cell) != 'unseen':
                        continue
                    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        (nr, nc) = (r + dr, c + dc)
                        if 0 <= nr < R and 0 <= nc < C and self._is_passable(self.full_grid[nr, nc]):
                            frontiers.append((nr, nc))
                            break
                if not frontiers:
                    for c in range(C):
                        if self._is_passable(self.full_grid[0, c]):
                            frontiers.append((0, c))
                        if self._is_passable(self.full_grid[R - 1, c]):
                            frontiers.append((R - 1, c))
                    for r in range(R):
                        if self._is_passable(self.full_grid[r, 0]):
                            frontiers.append((r, 0))
                        if self._is_passable(self.full_grid[r, C - 1]):
                            frontiers.append((r, C - 1))
                frontiers = [f for f in dict.fromkeys(frontiers) if f != pos]
                if not frontiers:
                    return []
                targets = frontiers
            path = self._shortest_path(pos, targets)
            if path is None or len(path) < 2:
                return []
            nxt = path[1]
            (dr, dc) = (nxt[0] - pos[0], nxt[1] - pos[1])
            desired_dir = DIR_MAP.get((dr, dc))
            if desired_dir is None:
                return []
            yield from self._turn_towards(desired_dir)
            yield from self.move_forward()

    def move(self, _from: str, _to: str):
        """
High-level action: go from region _from to region _to, replanning at each step and exploring unseen cells until the target is reached.
Implements (:action move :parameters (?from - region ?to - region))
"""
        self._agent_start_pos = self._agent_coords()

        def region_coords(name: str) -> List[Tuple[int, int]]:
            nm = self._parse_name(name)
            if nm == 'start':
                return [self._agent_start_pos]
            coords: List[Tuple[int, int]] = []
            for (pos, cell) in np.ndenumerate(self.full_grid):
                if str(cell) == nm:
                    coords.append(pos)
            return coords
        while True:
            pos = self._agent_coords()
            target_cells = region_coords(_to)
            if pos in target_cells:
                yield from self.done()
                return []
            goals = target_cells
            if not goals:
                (R, C) = self.full_grid.shape
                frontiers: List[Tuple[int, int]] = []
                for ((r, c), cell) in np.ndenumerate(self.full_grid):
                    if str(cell) != 'unseen':
                        continue
                    for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        (nr, nc) = (r + dr, c + dc)
                        if 0 <= nr < R and 0 <= nc < C and self._is_passable(self.full_grid[nr, nc]):
                            frontiers.append((nr, nc))
                            break
                if not frontiers:
                    for c in range(C):
                        if self._is_passable(self.full_grid[0, c]):
                            frontiers.append((0, c))
                        if self._is_passable(self.full_grid[R - 1, c]):
                            frontiers.append((R - 1, c))
                    for r in range(R):
                        if self._is_passable(self.full_grid[r, 0]):
                            frontiers.append((r, 0))
                        if self._is_passable(self.full_grid[r, C - 1]):
                            frontiers.append((r, C - 1))
                frontiers = [f for f in dict.fromkeys(frontiers) if f != pos]
                if not frontiers:
                    return []
                goals = frontiers
            path = self._shortest_path(pos, goals)
            if path is None or len(path) < 2:
                return []
            nxt = path[1]
            (dr, dc) = (nxt[0] - pos[0], nxt[1] - pos[1])
            DIR_MAP: Dict[Tuple[int, int], str] = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
            desired_dir = DIR_MAP.get((dr, dc))
            if desired_dir is None:
                return []
            yield from self._turn_towards(desired_dir)
            yield from self.move_forward()
        return []

    def finish(self, _a, _g):
        """
High-level action: signal completion at the target location _g.
Implements (:action finish :parameters (?a - agent ?g - location))
    """
        yield from self.done()
        return []

    def cross_river(self, _a, _c, _g):
        """
High-level action: traverse from crossing region _c to goal region _g.
Implements (:action cross_river :parameters (?a - agent ?c - location ?g - location))
    """
        yield from self.move(_a, _c, _g)
        return []

    def move_to_crossing(self, _a, _s, _c):
        """
High-level action: go from start region _s to crossing region _c.
Implements (:action move_to_crossing :parameters (?a - agent ?s - location ?c - location))
    """
        yield from self.move(_a, _s, _c)
        return []