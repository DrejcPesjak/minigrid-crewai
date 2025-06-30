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

    def move_forward_v2(self) -> list[int]:
        """
    Multi-step action: navigate until the agent stands on a goal cell.

    PDDL:
    (:action move_forward_v2
     :parameters ()
     :precondition ()
     :effect (and (agent ?to) (not (agent ?from)))
    )
    """

        def is_goal_cell(cell: str) -> bool:
            parts = cell.split()
            return parts[0] == 'goal'
        while True:
            if hasattr(self, 'prev_underlying') and is_goal_cell(self.prev_underlying):
                return []
            grid = self.full_grid
            goal_pos = list(zip(*np.where(np.char.startswith(grid.astype('<U20'), 'goal'))))
            if not goal_pos:
                yield from self.move_forward()
                continue
            start = self._agent_coords()
            (R, C) = grid.shape
            dq = deque([start])
            prev: dict[tuple[int, int], tuple[int, int]] = {start: None}
            target = None
            while dq:
                cur = dq.popleft()
                if cur in goal_pos:
                    target = cur
                    break
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    (nr, nc) = (cur[0] + dr, cur[1] + dc)
                    if not (0 <= nr < R and 0 <= nc < C):
                        continue
                    cell = grid[nr, nc]
                    if cell.startswith('wall') or cell == 'lava':
                        continue
                    if (nr, nc) not in prev:
                        prev[nr, nc] = cur
                        dq.append((nr, nc))
            if target is None:
                yield from self.move_forward()
                continue
            step = target
            while prev[step] != start:
                step = prev[step]
            (nr, nc) = step
            (ar, ac) = start
            (dr, dc) = (nr - ar, nc - ac)
            DIR_VECTORS = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
            desired_dir = DIR_VECTORS[dr, dc]
            dirs = ['North', 'East', 'South', 'West']
            cur_idx = dirs.index(self.current_dir)
            tgt_idx = dirs.index(desired_dir)
            diff = (tgt_idx - cur_idx) % 4
            if diff == 1:
                yield from self.turn_right()
            elif diff == 2:
                yield from self.turn_right()
                yield from self.turn_right()
            elif diff == 3:
                yield from self.turn_left()
            yield from self.move_forward()

    def is_goal(self, cell: str) -> bool:
        parts = cell.split()
        return parts[0] == 'goal'

    def move_forward_v3(self) -> List[int]:
        """
Multi-step action: navigate until the agent stands on the specified target in its mission.

PDDL:
(:action move_forward_v3
 :parameters ()
 :precondition ()
 :effect (and (at ?a ?goal))
)
"""
        mission = re.sub('-', '_', self.mission).lower()
        mission = re.sub('_(?:red|green|blue|purple|yellow|grey)$', '', mission)
        words = re.findall('\\w+', mission)
        COLORS = {'red', 'green', 'blue', 'purple', 'yellow', 'grey'}
        OBJECTS = {'unseen', 'empty', 'wall', 'floor', 'door', 'key', 'ball', 'box', 'goal', 'lava', 'agent'}
        color = next((w for w in words if w in COLORS), None)
        g_name = next((w for w in words if w in OBJECTS), None)
        if g_name is None:
            return []
        if hasattr(self, 'prev_underlying') and self.is_target_cell(self.prev_underlying, g_name, color):
            return []
        DIR = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        dirs = ['North', 'East', 'South', 'West']
        while True:
            grid = self.full_grid
            (R, C) = grid.shape
            target_cells = {(r, c) for r in range(R) for c in range(C) if self.is_target_cell(grid[r, c], g_name, color)}
            if not target_cells:
                (ar, ac) = self._agent_coords()
                cur_dir = self.current_dir
                cur_idx = dirs.index(cur_dir)
                left_dir = dirs[(cur_idx - 1) % 4]
                right_dir = dirs[(cur_idx + 1) % 4]
                (dr_front, dc_front) = DIR[cur_dir]
                (dr_left, dc_left) = DIR[left_dir]
                (dr_right, dc_right) = DIR[right_dir]

                def is_free(r: int, c: int) -> bool:
                    return 0 <= r < R and 0 <= c < C and (not (grid[r, c].startswith('wall') or grid[r, c] == 'lava'))
                if is_free(ar + dr_left, ac + dc_left):
                    yield from self.turn_left()
                    yield from self.move_forward()
                elif is_free(ar + dr_front, ac + dc_front):
                    yield from self.move_forward()
                elif is_free(ar + dr_right, ac + dc_right):
                    yield from self.turn_right()
                else:
                    yield from self.turn_right()
                    yield from self.turn_right()
                continue
            start = self._agent_coords()
            prev: Dict[Tuple[int, int], Tuple[int, int]] = {start: None}
            dq = deque([start])
            dest = None
            while dq:
                cur = dq.popleft()
                if cur in target_cells:
                    dest = cur
                    break
                for (dr, dc) in DIR.values():
                    (nr, nc) = (cur[0] + dr, cur[1] + dc)
                    if not (0 <= nr < R and 0 <= nc < C):
                        continue
                    cell = grid[nr, nc]
                    if cell.startswith('wall') or cell == 'lava':
                        continue
                    if (nr, nc) not in prev:
                        prev[nr, nc] = cur
                        dq.append((nr, nc))
            if dest is None:
                if self.am_next_to('wall') or self.lava_ahead():
                    yield from self.turn_right()
                else:
                    yield from self.move_forward()
                continue
            step = dest
            while prev[step] != start:
                step = prev[step]
            (nr, nc) = step
            (ar, ac) = start
            (dr, dc) = (nr - ar, nc - ac)
            DIR_VECTORS = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
            desired_dir = DIR_VECTORS[dr, dc]
            cur_idx = dirs.index(self.current_dir)
            tgt_idx = dirs.index(desired_dir)
            diff = (tgt_idx - cur_idx) % 4
            if diff == 1:
                yield from self.turn_right()
            elif diff == 2:
                yield from self.turn_right()
                yield from self.turn_right()
            elif diff == 3:
                yield from self.turn_left()
            yield from self.move_forward()

    def is_target_cell(self, cell: str, g_name: str, color: str) -> bool:
        parts = cell.split()
        if parts[0] != g_name:
            return False
        if color and (len(parts) < 2 or parts[1] != color):
            return False
        return True

    def move_forward_v4(self, _agent: str, _target: str) -> List[int]:
        """
Multi-step action: navigate until the agent stands on the specified target in its mission.

PDDL:
(:action move_forward_v4
 :parameters (?a - agent ?t - target)
 :precondition (and (agent ?a) (at ?a ?from) (target ?t) (not (at ?a ?t)))
 :effect (and (at ?a ?t) (not (at ?a ?from)))
)
"""
        mission = re.sub('-', '_', self.mission).lower()
        mission = re.sub('_(?:red|green|blue|purple|yellow|grey)$', '', mission)
        words = re.findall('\\w+', mission)
        COLORS = {'red', 'green', 'blue', 'purple', 'yellow', 'grey'}
        OBJECTS = {'unseen', 'empty', 'wall', 'floor', 'door', 'key', 'ball', 'box', 'goal', 'lava', 'agent'}
        color = next((w for w in words if w in COLORS), None)
        g_name = next((w for w in words if w in OBJECTS), None)
        if g_name is None:
            return []
        if hasattr(self, 'prev_underlying') and self.is_target_cell(self.prev_underlying, g_name, color):
            return []
        DIR = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        dirs = ['North', 'East', 'South', 'West']
        while True:
            grid = self.full_grid
            (R, C) = grid.shape
            target_cells = {(r, c) for r in range(R) for c in range(C) if self.is_target_cell(grid[r, c], g_name, color)}
            if not target_cells:
                (ar, ac) = self._agent_coords()
                cur_dir = self.current_dir
                cur_idx = dirs.index(cur_dir)
                left_dir = dirs[(cur_idx - 1) % 4]
                right_dir = dirs[(cur_idx + 1) % 4]
                (dr_front, dc_front) = DIR[cur_dir]
                (dr_left, dc_left) = DIR[left_dir]
                (dr_right, dc_right) = DIR[right_dir]

                def is_free(r: int, c: int) -> bool:
                    return 0 <= r < R and 0 <= c < C and (not (grid[r, c].startswith('wall') or grid[r, c] == 'lava'))
                if is_free(ar + dr_left, ac + dc_left):
                    yield from self.turn_left()
                    yield from self.move_forward()
                elif is_free(ar + dr_front, ac + dc_front):
                    yield from self.move_forward()
                elif is_free(ar + dr_right, ac + dc_right):
                    yield from self.turn_right()
                else:
                    yield from self.turn_right()
                    yield from self.turn_right()
                continue
            start = self._agent_coords()
            prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            dq = deque([start])
            dest = None
            while dq:
                cur = dq.popleft()
                if cur in target_cells:
                    dest = cur
                    break
                for (dr, dc) in DIR.values():
                    (nr, nc) = (cur[0] + dr, cur[1] + dc)
                    if not (0 <= nr < R and 0 <= nc < C):
                        continue
                    cell = grid[nr, nc]
                    if cell.startswith('wall') or cell == 'lava':
                        continue
                    if (nr, nc) not in prev:
                        prev[nr, nc] = cur
                        dq.append((nr, nc))
            if dest is None:
                if self.am_next_to('wall') or self.lava_ahead():
                    yield from self.turn_right()
                else:
                    yield from self.move_forward()
                continue
            step = dest
            while prev[step] != start:
                step = prev[step]
            (nr, nc) = step
            (ar, ac) = start
            (dr, dc) = (nr - ar, nc - ac)
            DIR_VECTORS = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
            desired_dir = DIR_VECTORS[dr, dc]
            cur_idx = dirs.index(self.current_dir)
            tgt_idx = dirs.index(desired_dir)
            diff = (tgt_idx - cur_idx) % 4
            if diff == 1:
                yield from self.turn_right()
            elif diff == 2:
                yield from self.turn_right()
                yield from self.turn_right()
            elif diff == 3:
                yield from self.turn_left()
            yield from self.move_forward()