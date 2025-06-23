import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Set

class Agent:

    def __init__(self):
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

    def am_next_to(self, obj: str) -> bool:
        (fr, fc) = self._front_coords()
        (R, C) = self.full_grid.shape
        return 0 <= fr < R and 0 <= fc < C and (self.full_grid[fr, fc] == obj)

    def lava_ahead(self) -> bool:
        (fr, fc) = self._front_coords()
        (R, C) = self.full_grid.shape
        return 0 <= fr < R and 0 <= fc < C and (self.full_grid[fr, fc] == 'lava')

    def safe_forward_infinite(self):
        while not self.lava_ahead():
            yield from self.move_forward()

    def pick_up_obj(self, obj: str) -> List[int]:
        if self.am_next_to(obj):
            return self.pick_up()
        return []

    def move_to_goal(self, agent: str, goal: str):
        """
    High-level action: navigate until the agent occupies the goal cell, then signal done().
    Performs frontier exploration (step-by-step) until the goal appears in the known map,
    then uses BFS-chasing one step at a time to walk to the goal, with fresh perception after each move.
    """
        import re
        from collections import deque
        m = re.match('^([a-zA-Z_]+)', goal)
        target = m.group(1) if m else goal
        (R, C) = self.full_grid.shape

        def is_free(cell: tuple[int, int]) -> bool:
            (r, c) = cell
            if not (0 <= r < R and 0 <= c < C):
                return False
            v = self.full_grid[r, c]
            return v not in ('wall', 'lava', 'door_locked')
        DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        DIR_ORDER = ['East', 'South', 'West', 'North']
        DIR_VECTORS = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        while True:
            (r0, c0) = self._agent_coords()
            cell = self.full_grid[r0, c0]
            if isinstance(cell, str) and cell.split()[0] == target:
                yield from self.done()
                return
            goals = [(r, c) for ((r, c), v) in np.ndenumerate(self.full_grid) if isinstance(v, str) and v.split()[0] == target]
            if goals:
                start = (r0, c0)
                goal_pos = goals[0]
                queue = deque([start])
                came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
                found = False
                while queue:
                    cur = queue.popleft()
                    if cur == goal_pos:
                        found = True
                        break
                    for (dr, dc) in DIRS:
                        nxt = (cur[0] + dr, cur[1] + dc)
                        if nxt not in came_from and is_free(nxt):
                            came_from[nxt] = cur
                            queue.append(nxt)
                if found:
                    path: list[tuple[int, int]] = []
                    node = goal_pos
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    nxt = path[1]
                    (dr, dc) = (nxt[0] - r0, nxt[1] - c0)
                    desired = next((d for (d, vec) in DIR_VECTORS.items() if vec == (dr, dc)))
                    ci = DIR_ORDER.index(self.current_dir)
                    di = DIR_ORDER.index(desired)
                    diff = (di - ci) % 4
                    if diff == 1:
                        yield from self.turn_right()
                    elif diff == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif diff == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            if 0 <= fr < R and 0 <= fc < C and is_free((fr, fc)):
                yield from self.move_forward()
            else:
                yield from self.turn_right()

    def move_to_goal_v2(self, agent: str, goal: str):
        """
Navigate until agent on goal, then done().
Sanitize goal name. Continuously locate goal if seen and chase via BFS one step at a time.
Otherwise, perform simple right-hand wall-following exploration to eventually see the goal.
"""
        import re
        from collections import deque
        target_match = re.match('^([a-zA-Z_]+)', goal.replace('-', '_'))
        target = target_match.group(1) if target_match else goal
        BLOCK = {'wall', 'lava', 'door_locked'}
        DIR_ORDER = ['East', 'South', 'West', 'North']
        DIR_VECT = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}

        def in_bounds(cell: tuple[int, int]) -> bool:
            (r, c) = cell
            (R, C) = self.full_grid.shape
            return 0 <= r < R and 0 <= c < C

        def passable(cell: tuple[int, int]) -> bool:
            (r, c) = cell
            return self.full_grid[r, c] not in BLOCK

        def bfs_step(start: tuple[int, int], goals: Set[tuple[int, int]]):
            """Return shortest path from start to any of goals, or None if unreachable."""
            queue = deque([start])
            came_from: Dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
            while queue:
                cur = queue.popleft()
                if cur in goals:
                    path: List[tuple[int, int]] = []
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for vec in DIR_VECT.values():
                    nxt = (cur[0] + vec[0], cur[1] + vec[1])
                    if nxt not in came_from and in_bounds(nxt) and passable(nxt):
                        if self.full_grid[nxt] == 'unseen':
                            continue
                        came_from[nxt] = cur
                        queue.append(nxt)
            return None
        while True:
            pos = self._agent_coords()
            under = self.full_grid[pos]
            if isinstance(under, str) and under.split()[0] == target:
                return self.done()
            goals = {cell for (cell, val) in np.ndenumerate(self.full_grid) if isinstance(val, str) and val.split()[0] == target}
            if goals:
                path = bfs_step(pos, goals)
                if path and len(path) > 1:
                    nxt = path[1]
                    (dr, dc) = (nxt[0] - pos[0], nxt[1] - pos[1])
                    desired = next((d for (d, v) in DIR_VECT.items() if v == (dr, dc)))
                    cur_i = DIR_ORDER.index(self.current_dir)
                    des_i = DIR_ORDER.index(desired)
                    diff = (des_i - cur_i) % 4
                    if diff == 1:
                        yield from self.turn_right()
                    elif diff == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif diff == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            cur_i = DIR_ORDER.index(self.current_dir)
            right_dir = DIR_ORDER[(cur_i + 1) % 4]
            right_vec = DIR_VECT[right_dir]
            right_cell = (pos[0] + right_vec[0], pos[1] + right_vec[1])
            if in_bounds(right_cell) and passable(right_cell):
                yield from self.turn_right()
                yield from self.move_forward()
                continue
            front = self._front_coords()
            if in_bounds(front) and passable(front):
                yield from self.move_forward()
                continue
            yield from self.turn_left()