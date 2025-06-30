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

    def is_goal(self, cell: str) -> bool:
        """Return True if the cell string corresponds to a goal object (any color)."""
        return cell.split()[0] == 'goal'

    def goal_navigation(self) -> List[int]:
        """
    Navigate the agent to the goal cell (any color) using BFS on the known map,
    then finish the episode. Returns a sequence of primitive action codes.
    Precondition: a goal must be present in the known global map.
    Effect: agent moves along a path to the goal and issues done().
    """
        target = None
        for ((r, c), cell) in np.ndenumerate(self.full_grid):
            if self.is_goal(cell):
                target = (r, c)
                break
        if target is None:
            return []
        start = self._agent_coords()
        (R, C) = self.full_grid.shape
        queue = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        found = False
        while queue:
            current = queue.popleft()
            if current == target:
                found = True
                break
            for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                (nr, nc) = (current[0] + dr, current[1] + dc)
                if (nr, nc) in came_from:
                    continue
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                cell = self.full_grid[nr, nc]
                obj = cell.split()[0]
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                came_from[nr, nc] = current
                queue.append((nr, nc))
        if not found:
            return []
        path: List[Tuple[int, int]] = []
        node = target
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        moves: List[int] = []
        cur_dir = self.current_dir
        cur_pos = start
        for next_pos in path[1:]:
            dr = next_pos[0] - cur_pos[0]
            dc = next_pos[1] - cur_pos[1]
            target_dir = VEC_TO_DIR[dr, dc]
            cur_idx = DIR_ORDER.index(cur_dir)
            tgt_idx = DIR_ORDER.index(target_dir)
            delta = (tgt_idx - cur_idx) % 4
            if delta == 1:
                moves += self.turn_right()
            elif delta == 2:
                moves += self.turn_right() + self.turn_right()
            elif delta == 3:
                moves += self.turn_left()
            moves += self.move_forward()
            cur_dir = target_dir
            cur_pos = next_pos
        moves += self.done()
        return moves

    def goal_navigation_v2(self) -> list[int]:
        """
    Navigate the agent to the goal cell (any color), exploring unknowns as needed.
    Re-plan on the growing full_grid between moves. Finish with done().
    Precondition: none (goal may be out of sight).
    Effect: agent reaches goal and issues done().
    """

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def is_goal_cell(cell: str) -> bool:
            return sanitize(cell).split()[0] == 'goal'

        def find_goal() -> tuple[int, int] | None:
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                if is_goal_cell(cell):
                    return (r, c)
            return None

        def neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            result: list[tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                obj = sanitize(self.full_grid[nr, nc]).split()[0]
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                result.append((nr, nc))
            return result

        def bfs(start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]] | None:
            queue = deque([start])
            came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
            while queue:
                current = queue.popleft()
                if current == target:
                    path: list[tuple[int, int]] = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(current):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = current
                    queue.append(nbr)
            return None
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        while True:
            target = find_goal()
            if target is not None:
                start = self._agent_coords()
                path = bfs(start, target)
                if path is not None:
                    if len(path) == 1:
                        yield from self.done()
                        return
                    next_pos = path[1]
                    dr = next_pos[0] - start[0]
                    dc = next_pos[1] - start[1]
                    target_dir = VEC_TO_DIR[dr, dc]
                    cur_idx = DIR_ORDER.index(self.current_dir)
                    tgt_idx = DIR_ORDER.index(target_dir)
                    delta = (tgt_idx - cur_idx) % 4
                    if delta == 1:
                        yield from self.turn_right()
                    elif delta == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif delta == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            (R, C) = self.full_grid.shape
            if 0 <= fr < R and 0 <= fc < C and (not self.lava_ahead()):
                front_obj = sanitize(self.full_grid[fr, fc]).split()[0]
                if front_obj not in ('wall',):
                    yield from self.move_forward()
                    continue
            yield from self.turn_right()

    def is_opening_ahead(self) -> bool:
        """
    Return True if an opening in the wall barrier is visible directly ahead (two cells ahead in the egocentric view).
    """
        if self.current_observation is None:
            return False

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])
        front = sanitize(self.current_observation[2, 3]).split()[0]
        beyond = sanitize(self.current_observation[1, 3]).split()[0]
        return front not in ('wall', 'lava', 'unseen') and beyond not in ('wall', 'lava', 'unseen')

    def follow_right_wall_until_opening(self) -> list[int]:
        """
    Follow the right-hand wall until an opening ahead is detected via is_opening_ahead().
    Uses the egocentric 7×7 observation for local wall-following.
    Returns a list of primitive action codes up to (but not including) the step through the opening.
    """
        moves: list[int] = []

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])
        while not self.is_opening_ahead():
            obs = self.current_observation
            right_obj = sanitize(obs[3, 4]).split()[0]
            if right_obj not in ('wall', 'lava', 'unseen'):
                moves += self.turn_right()
                moves += self.move_forward()
                continue
            front_obj = sanitize(obs[2, 3]).split()[0]
            if front_obj not in ('wall', 'lava', 'unseen'):
                moves += self.move_forward()
                continue
            moves += self.turn_left()
        return moves

    def goal_navigation_v3(self, _a: str, _t: str) -> List[int]:
        """
Navigate the agent to the target cell _t (any goal) by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Finish with done().
Precondition: none (goal may be out of sight).
Effect: agent reaches goal and issues done().
Parameters:
    _a: agent identifier (unused)
    _t: target identifier (unused)
    """

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def is_goal_cell(cell: str) -> bool:
            return sanitize(cell).split()[0] == 'goal'

        def find_goal() -> Optional[Tuple[int, int]]:
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                if is_goal_cell(cell):
                    return (r, c)
            return None

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            result: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                obj = sanitize(self.full_grid[nr, nc]).split()[0]
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                result.append((nr, nc))
            return result

        def bfs(start: Tuple[int, int], target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                current = queue.popleft()
                if current == target:
                    path: List[Tuple[int, int]] = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(current):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = current
                    queue.append(nbr)
            return None
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        while True:
            target = find_goal()
            if target is not None:
                start = self._agent_coords()
                path = bfs(start, target)
                if path is not None:
                    if len(path) == 1:
                        yield from self.done()
                        return
                    next_pos = path[1]
                    dr = next_pos[0] - start[0]
                    dc = next_pos[1] - start[1]
                    target_dir = VEC_TO_DIR[dr, dc]
                    cur_idx = DIR_ORDER.index(self.current_dir)
                    tgt_idx = DIR_ORDER.index(target_dir)
                    delta = (tgt_idx - cur_idx) % 4
                    if delta == 1:
                        yield from self.turn_right()
                    elif delta == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif delta == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            (R, C) = self.full_grid.shape
            if 0 <= fr < R and 0 <= fc < C and (not self.lava_ahead()):
                front_obj = sanitize(self.full_grid[fr, fc]).split()[0]
                if front_obj not in ('wall',):
                    yield from self.move_forward()
                    continue
            yield from self.turn_right()

    def goal_navigation_v4(self, _a: str, _t: str) -> List[int]:
        """
    Navigate the agent to the target cell _t (any goal) by exploring unknowns as needed,
    re-planning on the growing full_grid between moves. Avoid lava and walls, follow right-hand wall in corridors,
    and finish with done().
    Precondition: none (goal may be out of sight).
    Effect: agent reaches goal and issues done().
    Parameters:
        _a: agent identifier (unused)
        _t: target identifier (unused)
    """

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def is_goal_cell(cell: str) -> bool:
            return sanitize(cell).split()[0] == 'goal'

        def find_goal() -> Optional[Tuple[int, int]]:
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                if is_goal_cell(cell):
                    return (r, c)
            return None

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            result: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                obj = sanitize(self.full_grid[nr, nc]).split()[0]
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                result.append((nr, nc))
            return result

        def bfs(start: Tuple[int, int], target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                current = queue.popleft()
                if current == target:
                    path: List[Tuple[int, int]] = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(current):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = current
                    queue.append(nbr)
            return None
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        while True:
            target = find_goal()
            if target is not None:
                start = self._agent_coords()
                path = bfs(start, target)
                if path is not None:
                    if len(path) == 1:
                        yield from self.done()
                        return
                    next_pos = path[1]
                    dr = next_pos[0] - start[0]
                    dc = next_pos[1] - start[1]
                    target_dir = VEC_TO_DIR[dr, dc]
                    cur_idx = DIR_ORDER.index(self.current_dir)
                    tgt_idx = DIR_ORDER.index(target_dir)
                    delta = (tgt_idx - cur_idx) % 4
                    if delta == 1:
                        yield from self.turn_right()
                    elif delta == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif delta == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            (R, C) = self.full_grid.shape
            front_open = False
            if 0 <= fr < R and 0 <= fc < C and (not self.lava_ahead()):
                front_obj = sanitize(self.full_grid[fr, fc]).split()[0]
                if front_obj not in ('wall',):
                    front_open = True
            right_dir = DIR_ORDER[(DIR_ORDER.index(self.current_dir) + 1) % 4]
            (dr_r, dc_r) = DIR_TO_VEC[right_dir]
            (rr, rc) = (self._agent_coords()[0] + dr_r, self._agent_coords()[1] + dc_r)
            right_open = False
            if 0 <= rr < R and 0 <= rc < C:
                right_obj = sanitize(self.full_grid[rr, rc]).split()[0]
                if right_obj not in ('wall', 'lava', 'unseen'):
                    right_open = True
            if right_open:
                yield from self.turn_right()
                yield from self.move_forward()
            elif front_open:
                yield from self.move_forward()
            else:
                yield from self.turn_left()

    def goal_navigation_v5(self, _a: str, _t: str) -> List[int]:
        """
Navigate the agent to the target cell _t (any object) by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Avoid lava and walls,
fallback to right-hand wall-following in local corridors, and finish with done().
Precondition: none (target may be out of sight).
Effect: agent reaches target and issues done().
Parameters:
    _a: agent identifier (unused)
    _t: target identifier (unused)
"""

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def cell_parts(cell: str) -> List[str]:
            return cell.replace('-', '_').split()
        m = self.mission.lower().replace('-', '_')
        tokens = m.split()
        obj_name = tokens[-1]
        color = tokens[-2] if len(tokens) >= 3 and tokens[-2] in ('red', 'green', 'blue', 'purple', 'yellow', 'grey') else None

        def is_target_cell(cell: str) -> bool:
            parts = cell_parts(cell)
            if parts[0] != obj_name:
                return False
            if color:
                return len(parts) >= 2 and parts[1] == color
            return True

        def find_target() -> Optional[Tuple[int, int]]:
            for (idx, cell) in np.ndenumerate(self.full_grid):
                if is_target_cell(cell):
                    return (int(idx[0]), int(idx[1]))
            return None

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            result: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                if sanitize(self.full_grid[nr, nc]).split()[0] in ('wall', 'lava', 'unseen'):
                    continue
                result.append((nr, nc))
            return result

        def bfs(start: Tuple[int, int], target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                current = queue.popleft()
                if current == target:
                    path: List[Tuple[int, int]] = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(current):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = current
                    queue.append(nbr)
            return None
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        while True:
            target = find_target()
            if target is not None:
                start = self._agent_coords()
                path = bfs(start, target)
                if path is not None:
                    if len(path) == 1:
                        yield from self.done()
                        return
                    next_pos = path[1]
                    dr = next_pos[0] - start[0]
                    dc = next_pos[1] - start[1]
                    target_dir = VEC_TO_DIR.get((dr, dc))
                    cur_idx = DIR_ORDER.index(self.current_dir)
                    tgt_idx = DIR_ORDER.index(target_dir)
                    delta = (tgt_idx - cur_idx) % 4
                    if delta == 1:
                        yield from self.turn_right()
                    elif delta == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif delta == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            (R, C) = self.full_grid.shape
            front_open = False
            if 0 <= fr < R and 0 <= fc < C and (not self.lava_ahead()):
                if cell_parts(self.full_grid[fr, fc])[0] not in ('wall',):
                    front_open = True
            right_dir = DIR_ORDER[(DIR_ORDER.index(self.current_dir) + 1) % 4]
            (dr_r, dc_r) = DIR_TO_VEC[right_dir]
            (ar, ac) = self._agent_coords()
            (rr, rc) = (ar + dr_r, ac + dc_r)
            right_open = False
            if 0 <= rr < R and 0 <= rc < C:
                if cell_parts(self.full_grid[rr, rc])[0] not in ('wall', 'lava', 'unseen'):
                    right_open = True
            if right_open:
                yield from self.turn_right()
                yield from self.move_forward()
            elif front_open:
                yield from self.move_forward()
            else:
                yield from self.turn_left()

    def static_obstacle_navigation(self, _a: str, _t: str) -> List[int]:
        """
Navigate the agent to the target cell _t (any object) by planning on the static global map,
avoiding walls and lava, and finish with done().
Precondition: the target must already be visible in full_grid.
Effect: agent reaches the target and issues done().
Parameters:
    _a: agent identifier (unused)
    _t: target identifier (unused)
"""

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def cell_parts(cell: str) -> List[str]:
            return cell.replace('-', '_').split()
        m = self.mission.lower().replace('-', '_')
        tokens = m.split()
        obj_name = tokens[-1]
        COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
        color = tokens[-2] if len(tokens) >= 2 and tokens[-2] in COLORS else None

        def is_target_cell(cell: str) -> bool:
            parts = cell_parts(cell)
            if parts[0] != obj_name:
                return False
            if color:
                return len(parts) >= 2 and parts[1] == color
            return True
        target: Optional[Tuple[int, int]] = None
        for ((r, c), cell) in np.ndenumerate(self.full_grid):
            if is_target_cell(cell):
                target = (int(r), int(c))
                break
        if target is None:
            return []

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            nbrs: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                obj = sanitize(self.full_grid[nr, nc]).split()[0]
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                nbrs.append((nr, nc))
            return nbrs

        def bfs(start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                curr = queue.popleft()
                if curr == goal:
                    path: List[Tuple[int, int]] = []
                    node = curr
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(curr):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = curr
                    queue.append(nbr)
            return None
        start = self._agent_coords()
        path = bfs(start, target)
        if path is None:
            return []
        if len(path) == 1:
            return self.done()
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        moves: List[int] = []
        cur_dir = self.current_dir
        cur_pos = start
        for next_pos in path[1:]:
            dr = next_pos[0] - cur_pos[0]
            dc = next_pos[1] - cur_pos[1]
            target_dir = VEC_TO_DIR[dr, dc]
            cur_idx = DIR_ORDER.index(cur_dir)
            tgt_idx = DIR_ORDER.index(target_dir)
            delta = (tgt_idx - cur_idx) % 4
            if delta == 1:
                moves += self.turn_right()
            elif delta == 2:
                moves += self.turn_right() + self.turn_right()
            elif delta == 3:
                moves += self.turn_left()
            moves += self.move_forward()
            cur_dir = target_dir
            cur_pos = next_pos
        moves += self.done()
        return moves

    def static_obstacle_navigation_v2(self, _a: str, _t: str) -> List[int]:
        """
Navigate the agent to the target cell _t (any object) by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Avoid lava and walls,
fallback to right-hand wall-following in local corridors, and finish with done().
Precondition: none (target may be out of sight).
Effect: agent reaches target and issues done().
Parameters:
    _a: agent identifier (unused)
    _t: target identifier (unused)
"""

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def cell_parts(cell: str) -> List[str]:
            return cell.replace('-', '_').split()
        m = self.mission.lower().replace('-', '_')
        tokens = m.split()
        obj_name = tokens[-1]
        COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
        color = tokens[-2] if len(tokens) >= 2 and tokens[-2] in COLORS else None

        def is_target_cell(cell: str) -> bool:
            parts = cell_parts(cell)
            if not parts or parts[0] != obj_name:
                return False
            if color:
                return len(parts) >= 2 and parts[1] == color
            return True

        def find_target() -> Optional[Tuple[int, int]]:
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                if is_target_cell(cell):
                    return (int(r), int(c))
            return None

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            result: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                obj = sanitize(self.full_grid[nr, nc]).split()[0]
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                result.append((nr, nc))
            return result

        def bfs(start: Tuple[int, int], goals: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                curr = queue.popleft()
                if curr in goals:
                    path: List[Tuple[int, int]] = []
                    node = curr
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(curr):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = curr
                    queue.append(nbr)
            return None
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        while True:
            target = find_target()
            start = self._agent_coords()
            if target is not None:
                targ_parts = cell_parts(self.full_grid[target[0], target[1]])
                is_pickupable = targ_parts[0] in ('key', 'ball', 'box')
                if is_pickupable:
                    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    walk_targets: Set[Tuple[int, int]] = set()
                    (R, C) = self.full_grid.shape
                    for (dr, dc) in deltas:
                        (nr, nc) = (target[0] + dr, target[1] + dc)
                        if not (0 <= nr < R and 0 <= nc < C):
                            continue
                        if sanitize(self.full_grid[nr, nc]).split()[0] in ('wall', 'lava', 'unseen'):
                            continue
                        walk_targets.add((nr, nc))
                    if not walk_targets:
                        yield from self.follow_right_wall_until_opening()
                        yield from self.move_forward()
                        yield from self.pick_up()
                        yield from self.done()
                        return
                else:
                    walk_targets = {target}
                path = bfs(start, walk_targets)
                if path is not None:
                    if len(path) == 1:
                        if is_pickupable:
                            dr = target[0] - start[0]
                            dc = target[1] - start[1]
                            tgt_dir = VEC_TO_DIR[dr, dc]
                            cur_idx = DIR_ORDER.index(self.current_dir)
                            tgt_idx = DIR_ORDER.index(tgt_dir)
                            delta = (tgt_idx - cur_idx) % 4
                            if delta == 1:
                                yield from self.turn_right()
                            elif delta == 2:
                                yield from self.turn_right()
                                yield from self.turn_right()
                            elif delta == 3:
                                yield from self.turn_left()
                            yield from self.pick_up()
                        yield from self.done()
                        return
                    next_pos = path[1]
                    dr = next_pos[0] - start[0]
                    dc = next_pos[1] - start[1]
                    tgt_dir = VEC_TO_DIR[dr, dc]
                    cur_idx = DIR_ORDER.index(self.current_dir)
                    tgt_idx = DIR_ORDER.index(tgt_dir)
                    delta = (tgt_idx - cur_idx) % 4
                    if delta == 1:
                        yield from self.turn_right()
                    elif delta == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif delta == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            (R, C) = self.full_grid.shape
            front_open = False
            if 0 <= fr < R and 0 <= fc < C and (not self.lava_ahead()):
                if sanitize(self.full_grid[fr, fc]).split()[0] not in ('wall',):
                    front_open = True
            right_dir = DIR_ORDER[(DIR_ORDER.index(self.current_dir) + 1) % 4]
            (dr_r, dc_r) = DIR_TO_VEC[right_dir]
            (ar, ac) = self._agent_coords()
            (rr, rc) = (ar + dr_r, ac + dc_r)
            right_open = False
            if 0 <= rr < R and 0 <= rc < C:
                if sanitize(self.full_grid[rr, rc]).split()[0] not in ('wall', 'lava', 'unseen'):
                    right_open = True
            if right_open:
                yield from self.turn_right()
                yield from self.move_forward()
            elif front_open:
                yield from self.move_forward()
            else:
                yield from self.turn_left()

    def hazard_avoidance(self, _a: str, _t: str) -> List[int]:
        """
Navigate the agent to the target cell _t (any goal) by planning on the static global map,
avoiding walls and lava, and finish with done().
Precondition: the target must already be visible in full_grid.
Effect: agent reaches the target and issues done().
Parameters:
    _a: agent identifier (unused)
    _t: target identifier (unused)
"""

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def cell_parts(cell: str) -> List[str]:
            return cell.replace('-', '_').split()
        m = self.mission.lower().replace('-', '_')
        tokens = m.split()
        COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
        color = None
        if 'goal' in tokens:
            idx = tokens.index('goal')
            if idx >= 1 and tokens[idx - 1] in COLORS:
                color = tokens[idx - 1]

        def is_target_cell(cell: str) -> bool:
            parts = cell_parts(cell)
            if not parts or parts[0] != 'goal':
                return False
            if color:
                return len(parts) >= 2 and parts[1] == color
            return True
        target: Optional[Tuple[int, int]] = None
        for ((r, c), cell) in np.ndenumerate(self.full_grid):
            if is_target_cell(cell):
                target = (int(r), int(c))
                break
        if target is None:
            return []

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            nbrs: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                if sanitize(self.full_grid[nr, nc]).split()[0] in ('wall', 'lava', 'unseen'):
                    continue
                nbrs.append((nr, nc))
            return nbrs

        def bfs(start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                curr = queue.popleft()
                if curr == goal:
                    path: List[Tuple[int, int]] = []
                    node = curr
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(curr):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = curr
                    queue.append(nbr)
            return None
        start = self._agent_coords()
        path = bfs(start, target)
        if path is None:
            return []
        if len(path) == 1:
            return self.done()
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        moves: List[int] = []
        cur_dir = self.current_dir
        cur_pos = start
        for next_pos in path[1:]:
            dr = next_pos[0] - cur_pos[0]
            dc = next_pos[1] - cur_pos[1]
            target_dir = VEC_TO_DIR[dr, dc]
            cur_idx = DIR_ORDER.index(cur_dir)
            tgt_idx = DIR_ORDER.index(target_dir)
            delta = (tgt_idx - cur_idx) % 4
            if delta == 1:
                moves += self.turn_right()
            elif delta == 2:
                moves += self.turn_right() + self.turn_right()
            elif delta == 3:
                moves += self.turn_left()
            moves += self.move_forward()
            cur_dir = target_dir
            cur_pos = next_pos
        moves += self.done()
        return moves

    def hazard_avoidance_v2(self, _a: str, _t: str) -> List[int]:
        """
Navigate the agent to the target cell _t (any goal) by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Avoid walls, lava, and locked doors,
fallback to right-hand wall-following in local corridors, and finish with done().
Precondition: none (goal may be out of sight).
Effect: agent reaches goal and issues done().
Parameters:
    _a: agent identifier (unused)
    _t: target identifier (unused)
"""

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def is_goal_cell(cell: str) -> bool:
            return sanitize(cell).split()[0] == 'goal'

        def find_goal() -> Optional[Tuple[int, int]]:
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                if is_goal_cell(cell):
                    return (int(r), int(c))
            return None

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            result: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                cell = sanitize(self.full_grid[nr, nc])
                parts = cell.split()
                obj = parts[0]
                state = parts[1] if len(parts) > 1 else None
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                if obj == 'door' and state == 'locked':
                    continue
                result.append((nr, nc))
            return result

        def bfs(start: Tuple[int, int], target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                current = queue.popleft()
                if current == target:
                    path: List[Tuple[int, int]] = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(current):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = current
                    queue.append(nbr)
            return None
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        while True:
            target = find_goal()
            if target is not None:
                start = self._agent_coords()
                path = bfs(start, target)
                if path is not None:
                    if len(path) == 1:
                        yield from self.done()
                        return
                    next_pos = path[1]
                    dr = next_pos[0] - start[0]
                    dc = next_pos[1] - start[1]
                    target_dir = VEC_TO_DIR[dr, dc]
                    cur_idx = DIR_ORDER.index(self.current_dir)
                    tgt_idx = DIR_ORDER.index(target_dir)
                    delta = (tgt_idx - cur_idx) % 4
                    if delta == 1:
                        yield from self.turn_right()
                    elif delta == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif delta == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            (R, C) = self.full_grid.shape
            front_open = False
            if 0 <= fr < R and 0 <= fc < C and (not self.lava_ahead()):
                cell = sanitize(self.full_grid[fr, fc])
                parts = cell.split()
                obj = parts[0]
                state = parts[1] if len(parts) > 1 else None
                if obj != 'wall' and (not (obj == 'door' and state == 'locked')):
                    front_open = True
            right_dir = DIR_ORDER[(DIR_ORDER.index(self.current_dir) + 1) % 4]
            (dr_r, dc_r) = DIR_TO_VEC[right_dir]
            (ar, ac) = self._agent_coords()
            (rr, rc) = (ar + dr_r, ac + dc_r)
            right_open = False
            if 0 <= rr < R and 0 <= rc < C:
                cell = sanitize(self.full_grid[rr, rc])
                parts = cell.split()
                obj = parts[0]
                state = parts[1] if len(parts) > 1 else None
                if obj not in ('wall', 'lava', 'unseen') and (not (obj == 'door' and state == 'locked')):
                    right_open = True
            if right_open:
                yield from self.turn_right()
                yield from self.move_forward()
            elif front_open:
                yield from self.move_forward()
            else:
                yield from self.turn_left()

    def pickup_only(self, _a: str, _i: str) -> List[int]:
        """
Navigate the agent to the specified item _i by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Avoid lava and walls,
fallback to right-hand wall-following in local corridors, and finish with pick_up().
Precondition: none (item may be out of sight).
Effect: agent reaches item and issues pick_up().
Parameters:
    _a: agent identifier (unused)
    _i: item identifier (unused)
"""
        for code in self.static_obstacle_navigation_v2(_a, _i):
            yield code
            if code == 3:
                return

    def pickup_only_v2(self, _a: str, _i: str) -> List[int]:
        """
Navigate the agent to the specified item _i by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Avoid lava, walls, and locked doors,
fallback to frontier-based exploration of unseen areas, and finish with pick_up().
Precondition: none (item may be out of sight).
Effect: agent reaches item and issues pick_up().
Parameters:
    _a: agent identifier (unused)
    _i: item identifier (unused)
"""

        def sanitize(cell: str) -> str:
            parts = cell.replace('-', '_').split()
            return ' '.join(parts[0:1] + [s for s in parts[1:] if s in ('open', 'closed', 'locked')])

        def cell_parts(cell: str) -> List[str]:
            return cell.replace('-', '_').split()
        m = self.mission.lower().replace('-', '_')
        tokens = m.split()
        obj_name = tokens[-1]
        COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
        color = tokens[-2] if len(tokens) >= 2 and tokens[-2] in COLORS else None

        def is_target_cell(cell: str) -> bool:
            parts = cell_parts(cell)
            if not parts or parts[0] != obj_name:
                return False
            if color:
                return len(parts) >= 2 and parts[1] == color
            return True

        def find_target() -> Optional[Tuple[int, int]]:
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                if is_target_cell(cell):
                    return (int(r), int(c))
            return None

        def neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            (R, C) = self.full_grid.shape
            deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            result: List[Tuple[int, int]] = []
            for (dr, dc) in deltas:
                (nr, nc) = (pos[0] + dr, pos[1] + dc)
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                parts = sanitize(self.full_grid[nr, nc]).split()
                obj = parts[0]
                state = parts[1] if len(parts) > 1 else None
                if obj in ('wall', 'lava', 'unseen'):
                    continue
                if obj == 'door' and state == 'locked':
                    continue
                result.append((nr, nc))
            return result

        def bfs(start: Tuple[int, int], goals: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
            queue = deque([start])
            came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
            while queue:
                curr = queue.popleft()
                if curr in goals:
                    path: List[Tuple[int, int]] = []
                    node = curr
                    while node is not None:
                        path.append(node)
                        node = came_from[node]
                    path.reverse()
                    return path
                for nbr in neighbors(curr):
                    if nbr in came_from:
                        continue
                    came_from[nbr] = curr
                    queue.append(nbr)
            return None

        def find_frontier() -> Set[Tuple[int, int]]:
            frontier: Set[Tuple[int, int]] = set()
            (R, C) = self.full_grid.shape
            for ((r, c), cell) in np.ndenumerate(self.full_grid):
                if sanitize(cell).split()[0] in ('wall', 'lava', 'unseen'):
                    continue
                for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    (nr, nc) = (r + dr, c + dc)
                    if 0 <= nr < R and 0 <= nc < C and (self.full_grid[nr, nc] == 'unseen'):
                        frontier.add((r, c))
                        break
            return frontier
        DIR_TO_VEC = {'East': (0, 1), 'South': (1, 0), 'West': (0, -1), 'North': (-1, 0)}
        VEC_TO_DIR = {v: k for (k, v) in DIR_TO_VEC.items()}
        DIR_ORDER = ['North', 'East', 'South', 'West']
        while True:
            target = find_target()
            if target is not None:
                start = self._agent_coords()
                deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                walk_targets: Set[Tuple[int, int]] = set()
                (R, C) = self.full_grid.shape
                for (dr, dc) in deltas:
                    nbr = (target[0] + dr, target[1] + dc)
                    if 0 <= nbr[0] < R and 0 <= nbr[1] < C:
                        if sanitize(self.full_grid[nbr]).split()[0] not in ('wall', 'lava', 'unseen'):
                            walk_targets.add(nbr)
                if not walk_targets:
                    yield from self.follow_right_wall_until_opening()
                    yield from self.move_forward()
                else:
                    path = bfs(start, walk_targets)
                    if path:
                        for next_pos in path[1:]:
                            dr = next_pos[0] - start[0]
                            dc = next_pos[1] - start[1]
                            target_dir = VEC_TO_DIR[dr, dc]
                            cur_idx = DIR_ORDER.index(self.current_dir)
                            tgt_idx = DIR_ORDER.index(target_dir)
                            delta = (tgt_idx - cur_idx) % 4
                            if delta == 1:
                                yield from self.turn_right()
                            elif delta == 2:
                                yield from self.turn_right()
                                yield from self.turn_right()
                            elif delta == 3:
                                yield from self.turn_left()
                            yield from self.move_forward()
                            start = next_pos
                dr = target[0] - start[0]
                dc = target[1] - start[1]
                tgt_dir = VEC_TO_DIR[dr, dc]
                cur_idx = DIR_ORDER.index(self.current_dir)
                tgt_idx = DIR_ORDER.index(tgt_dir)
                delta = (tgt_idx - cur_idx) % 4
                if delta == 1:
                    yield from self.turn_right()
                elif delta == 2:
                    yield from self.turn_right()
                    yield from self.turn_right()
                elif delta == 3:
                    yield from self.turn_left()
                yield from self.pick_up()
                return
            frontier = find_frontier()
            if frontier:
                start = self._agent_coords()
                path = bfs(start, frontier)
                if path and len(path) >= 2:
                    next_pos = path[1]
                    dr = next_pos[0] - start[0]
                    dc = next_pos[1] - start[1]
                    tgt_dir = VEC_TO_DIR[dr, dc]
                    cur_idx = DIR_ORDER.index(self.current_dir)
                    tgt_idx = DIR_ORDER.index(tgt_dir)
                    delta = (tgt_idx - cur_idx) % 4
                    if delta == 1:
                        yield from self.turn_right()
                    elif delta == 2:
                        yield from self.turn_right()
                        yield from self.turn_right()
                    elif delta == 3:
                        yield from self.turn_left()
                    yield from self.move_forward()
                    continue
            (fr, fc) = self._front_coords()
            (R, C) = self.full_grid.shape
            if 0 <= fr < R and 0 <= fc < C and (not self.lava_ahead()):
                if sanitize(self.full_grid[fr, fc]).split()[0] not in ('wall', 'unseen'):
                    yield from self.move_forward()
                    continue
            yield from self.turn_right()

    def pickup_only_v3(self, _a: str, _i: str) -> List[int]:
        """
Navigate the agent to the specified item _i by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Avoid lava, walls, and locked doors,
fallback to static obstacle navigation (which handles corridors and pick-up stance),
and finish with pick_up().
Precondition: none (item may be out of sight).
Effect: agent reaches item and issues pick_up().
Parameters:
    _a: agent identifier (unused)
    _i: item identifier (unused)
"""
        while self.inventory is None:
            for code in self.pickup_only_v2(_a, _i):
                yield code
                if self.inventory is not None:
                    return

    def pickup_only_v4(self, _a: str, _i: str) -> List[int]:
        """
Navigate the agent to the specified item _i by exploring unknowns as needed,
re-planning on the growing full_grid between moves. Avoid lava, walls, and locked doors,
fallback to pickup_only_v3, and finish with pick_up().
Precondition: none (item may be out of sight).
Effect: agent reaches item and issues pick_up().
Parameters:
    _a: agent identifier (unused)
    _i: item identifier (unused)
"""
        m = self.mission.lower().replace('-', '_')
        tokens = m.split()
        obj_name = tokens[-1]
        COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
        color = tokens[-2] if len(tokens) >= 2 and tokens[-2] in COLORS else None
        desired = obj_name + (f' {color}' if color else '')
        while self.inventory is not None and self.inventory != desired:
            while not (self.am_next_to('empty') or self.am_next_to('floor')):
                yield from self.turn_right()
            yield from self.drop()
        if self.inventory == desired:
            return []
        yield from self.pickup_only_v3(_a, _i)
        return []