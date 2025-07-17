import re
from collections import deque
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Set
from typing import List, Optional, Tuple
from typing import List, Tuple, Dict

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

    def reach_goal(self, _a: str, _d: str, _r: str):
        """
High-level action: open the door _d (anywhere adjacent), step through it into region _r, and signal completion.
Implements (:action reach_goal :parameters (?a - agent ?d - door ?r - region))
    """
        door_tokens = self._parse_name(_d).split()
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        while True:
            full = self.full_grid
            door_cells = [(i, j) for ((i, j), cell) in np.ndenumerate(full) if all((tok in str(cell).split() for tok in door_tokens))]
            if door_cells:
                break
            pos = self._agent_coords()
            (R, C) = full.shape
            frontiers: list[tuple[int, int]] = []
            for ((i, j), cell) in np.ndenumerate(full):
                if str(cell) != 'unseen':
                    continue
                for (dr, dc) in DIRS:
                    (ni, nj) = (i + dr, j + dc)
                    if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                        frontiers.append((ni, nj))
                        break
            if not frontiers:
                for c in range(C):
                    if self._is_passable(full[0, c]):
                        frontiers.append((0, c))
                    if self._is_passable(full[R - 1, c]):
                        frontiers.append((R - 1, c))
                for r in range(R):
                    if self._is_passable(full[r, 0]):
                        frontiers.append((r, 0))
                    if self._is_passable(full[r, C - 1]):
                        frontiers.append((r, C - 1))
            goals = [g for g in dict.fromkeys(frontiers) if g != pos]
            if not goals:
                return []
            path = self._shortest_path(pos, goals)
            if not path or len(path) < 2:
                return []
            for (cur, nxt) in zip(path, path[1:]):
                (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
                desired = DIR_MAP.get((dr, dc))
                if desired is None:
                    return []
                yield from self._turn_towards(desired)
                yield from self.move_forward()
        yield from self._ensure_door_open(_d)
        yield from self.move_forward()
        yield from self.done()
        return []

    def move(self, _from: str, _to: str):
        """
High-level action: go from region _from to region _to.
Implements (:action move :parameters (?from - region ?to - region))
    """
        full = self.full_grid
        (R, C) = full.shape
        start = self._agent_coords()
        region_from: Set[Tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            (r, c) = queue.popleft()
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                (nr, nc) = (r + dr, c + dc)
                if 0 <= nr < R and 0 <= nc < C and ((nr, nc) not in region_from):
                    cell = full[nr, nc]
                    if self._is_passable(cell) and cell != 'unseen':
                        region_from.add((nr, nc))
                        queue.append((nr, nc))
        frontier: List[Tuple[int, int]] = []
        for ((i, j), cell) in np.ndenumerate(full):
            if (i, j) in region_from:
                continue
            if not self._is_passable(cell) or cell == 'unseen':
                continue
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                (ni, nj) = (i + dr, j + dc)
                if (ni, nj) in region_from:
                    frontier.append((i, j))
                    break
        if not frontier:
            return []
        region_to = set(frontier)
        queue = deque(frontier)
        while queue:
            (r, c) = queue.popleft()
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                (nr, nc) = (r + dr, c + dc)
                if 0 <= nr < R and 0 <= nc < C and ((nr, nc) not in region_to):
                    cell = full[nr, nc]
                    if self._is_passable(cell) and cell != 'unseen':
                        region_to.add((nr, nc))
                        queue.append((nr, nc))
        path = self._shortest_path(start, list(region_to))
        if not path or len(path) < 2:
            return []
        DIR_MAP: Dict[Tuple[int, int], str] = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        for (cur, nxt) in zip(path, path[1:]):
            (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
            desired = DIR_MAP.get((dr, dc))
            if desired is None:
                return []
            yield from self._turn_towards(desired)
            yield from self.move_forward()
        return []

    def finish(self, _a: str, _g: str):
        """
High-level action: move to region _g then signal completion.
Implements (:action finish :parameters (?a - agent ?g - region))
    """
        region_tokens = self._parse_name(_g).split()
        DIR_REL = {'North': {'front': (-1, 0), 'back': (1, 0), 'left': (0, -1), 'right': (0, 1)}, 'East': {'front': (0, 1), 'back': (0, -1), 'left': (-1, 0), 'right': (1, 0)}, 'South': {'front': (1, 0), 'back': (-1, 0), 'left': (0, 1), 'right': (0, -1)}, 'West': {'front': (0, -1), 'back': (0, 1), 'left': (1, 0), 'right': (-1, 0)}}
        rel_map = DIR_REL.get(self.current_dir, {})
        ori = next((tok for tok in region_tokens if tok in rel_map), None)
        if ori is None and 'goal' in region_tokens and ('front' in rel_map):
            ori = 'front'
        (dr_end, dc_end) = rel_map.get(ori, (0, 0))
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        DIR_MAP = {d: n for (d, n) in zip(DIRS, ['North', 'South', 'West', 'East'])}
        while True:
            full = self.full_grid
            goal_cells = [pos for (pos, cell) in np.ndenumerate(full) if self.is_goal(str(cell))]
            if goal_cells:
                break
            yield from self._explore_frontier()
        full = self.full_grid
        goal_cells = [pos for (pos, cell) in np.ndenumerate(full) if self.is_goal(str(cell))]
        (R, C) = full.shape
        entry_points = []
        for (gi, gj) in goal_cells:
            for (dr, dc) in DIRS:
                (ni, nj) = (gi + dr, gj + dc)
                if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                    entry_points.append((ni, nj))
        entry_points = list(dict.fromkeys(entry_points))
        start = self._agent_coords()
        if start not in entry_points:
            path = self._shortest_path(start, [p for p in entry_points if p != start])
            if not path or len(path) < 2:
                return []
            pos = start
            for nxt in path[1:]:
                (dr, dc) = (nxt[0] - pos[0], nxt[1] - pos[1])
                desired = DIR_MAP.get((dr, dc))
                if desired is None:
                    return []
                yield from self._turn_towards(desired)
                yield from self.move_forward()
                pos = nxt
        pos = self._agent_coords()
        target_goal = None
        for (gi, gj) in goal_cells:
            if abs(gi - pos[0]) + abs(gj - pos[1]) == 1:
                target_goal = (gi, gj)
                break
        if target_goal is None:
            return []
        (dr, dc) = (target_goal[0] - pos[0], target_goal[1] - pos[1])
        desired = DIR_MAP.get((dr, dc))
        if desired:
            yield from self._turn_towards(desired)
        yield from self.move_forward()
        yield from self.done()
        return []

    def cross_river(self, _a: str, _c: str, _g: str):
        """
High-level action: traverse from crossing region _c to goal region _g.
Implements (:action cross_river :parameters (?a - agent ?c - location ?g - location))
    """
        yield from self.move(_c, _g)
        return []

    def move_to_crossing(self, _a: str, _s: str, _c: str):
        """
High-level action: go from start region _s to crossing region _c.
Implements (:action move_to_crossing :parameters (?a - agent ?s - location ?c - location))
    """
        yield from self.move(_s, _c)
        return []

    def is_door(self, cell: str) -> bool:
        """Return True if the cell is a door (any color/state)."""
        tokens = cell.split()
        return bool(tokens) and tokens[0] == 'door'

    def is_locked(self, cell: str) -> bool:
        """Return True if the door cell is locked."""
        tokens = cell.split()
        return self.is_door(cell) and 'locked' in tokens

    def is_closed(self, cell: str) -> bool:
        """Return True if the door cell is closed (but not locked)."""
        tokens = cell.split()
        return self.is_door(cell) and 'closed' in tokens

    def toggle_door(self, _door: str) -> List[int]:
        """Primitive toggle action (attempt to open/close/unlock if holding key)."""
        return [5]

    def _ensure_door_open(self, _d: str):
        """Ensure door _d is open; unlock or toggle if needed."""
        door_name = self._parse_name(_d)
        tokens = door_name.split()
        COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
        color_token = next((tok for tok in tokens if tok in COLORS), None)
        full = self.full_grid
        door_positions = [pos for (pos, cell) in np.ndenumerate(full) if cell.startswith('door') and (color_token is None or color_token in cell.split())]
        if not door_positions:
            return []
        door_pos = door_positions[0]
        cell = full[door_pos]
        if 'open' in str(cell).split():
            return []
        start = self._agent_coords()
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        if abs(door_pos[0] - start[0]) + abs(door_pos[1] - start[1]) == 1:
            (dr, dc) = (door_pos[0] - start[0], door_pos[1] - start[1])
            desired = DIR_MAP.get((dr, dc))
            if desired:
                yield from self._turn_towards(desired)
            (fr, fc) = self._front_coords()
            door_cell = self.full_grid[fr, fc]
            if self.is_locked(door_cell) or self.is_closed(door_cell):
                yield from self.toggle_door(_d)
            return []
        (R, C) = full.shape
        adj = []
        for (dr, dc) in DIR_MAP:
            (ni, nj) = (door_pos[0] + dr, door_pos[1] + dc)
            if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                adj.append((ni, nj))
        if not adj:
            return []
        path = self._shortest_path(start, adj)
        if not path or len(path) < 2:
            return []
        pos = start
        for nxt in path[1:]:
            (dr, dc) = (nxt[0] - pos[0], nxt[1] - pos[1])
            desired = DIR_MAP.get((dr, dc))
            if desired is None:
                return []
            yield from self._turn_towards(desired)
            yield from self.move_forward()
            pos = nxt
        (dr, dc) = (door_pos[0] - pos[0], door_pos[1] - pos[1])
        desired = DIR_MAP.get((dr, dc))
        if desired:
            yield from self._turn_towards(desired)
        (fr, fc) = self._front_coords()
        door_cell = self.full_grid[fr, fc]
        if self.is_locked(door_cell) or self.is_closed(door_cell):
            yield from self.toggle_door(_d)
        return []

    def go_to_target(self, _a: str, _o: str, _r: str):
        """
High-level action: go to the target object matching the remembered (or provided) target
at the region _r, step to it, pick it up, and signal completion.
Implements (:action go_to_target :parameters (?a - agent ?o - obj ?r - region))
    """
        parsed_o = self._parse_name(_o).split()
        if not hasattr(self, 'target_tokens') or self.target_tokens != parsed_o:
            self.target_tokens = parsed_o
        region_tokens = self._parse_name(_r).split()
        DIR_REL = {'North': {'front': (-1, 0), 'back': (1, 0), 'left': (0, -1), 'right': (0, 1)}, 'East': {'front': (0, 1), 'back': (0, -1), 'left': (-1, 0), 'right': (1, 0)}, 'South': {'front': (1, 0), 'back': (-1, 0), 'left': (0, 1), 'right': (0, -1)}, 'West': {'front': (0, -1), 'back': (0, 1), 'left': (1, 0), 'right': (-1, 0)}}
        rel_map = DIR_REL.get(self.current_dir, {})
        ori = next((tok for tok in region_tokens if tok in rel_map), None)
        if ori is None:
            if 'end' in region_tokens and 'front' in rel_map:
                ori = 'front'
            elif 'start' in region_tokens and 'back' in rel_map:
                ori = 'back'
            else:
                return []
        (dr_end, dc_end) = rel_map[ori]
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        while True:
            full = self.full_grid
            target_cells = [pos for (pos, cell) in np.ndenumerate(full) if all((tok in str(cell).split() for tok in self.target_tokens))]
            if len(target_cells) > 1:
                break
            pos = self._agent_coords()
            (R, C) = full.shape
            frontiers: list[tuple[int, int]] = []
            for ((i, j), cell) in np.ndenumerate(full):
                if str(cell) != 'unseen':
                    continue
                for (dr, dc) in DIRS:
                    (ni, nj) = (i + dr, j + dc)
                    if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                        frontiers.append((ni, nj))
                        break
            if not frontiers:
                for c in range(C):
                    if self._is_passable(full[0, c]):
                        frontiers.append((0, c))
                    if self._is_passable(full[R - 1, c]):
                        frontiers.append((R - 1, c))
                for r in range(R):
                    if self._is_passable(full[r, 0]):
                        frontiers.append((r, 0))
                    if self._is_passable(full[r, C - 1]):
                        frontiers.append((r, C - 1))
            goals = [g for g in dict.fromkeys(frontiers) if g != pos]
            if not goals:
                return []
            path = self._shortest_path(pos, goals)
            if not path or len(path) < 2:
                return []
            for (cur, nxt) in zip(path, path[1:]):
                (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
                desired = DIR_MAP.get((dr, dc))
                if desired is None:
                    return []
                yield from self._turn_towards(desired)
                yield from self.move_forward()
        full = self.full_grid
        start = self._agent_coords()

        def _proj(cell: tuple[int, int]) -> int:
            return (cell[0] - start[0]) * dr_end + (cell[1] - start[1]) * dc_end
        (ti, tj) = max(target_cells, key=_proj)
        entry_points: list[tuple[int, int]] = []
        (R, C) = full.shape
        for (dr, dc) in DIRS:
            (ni, nj) = (ti + dr, tj + dc)
            if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                entry_points.append((ni, nj))
        if not entry_points:
            return []
        start = self._agent_coords()
        if start in entry_points:
            (dr, dc) = (ti - start[0], tj - start[1])
            desired = DIR_MAP.get((dr, dc))
            if desired:
                yield from self._turn_towards(desired)
            yield from self.move_forward()
            yield from self.pick_up()
            yield from self.done()
            return []
        path = self._shortest_path(start, entry_points)
        if not path or len(path) < 2:
            return []
        cur = start
        for nxt in path[1:]:
            (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
            desired = DIR_MAP.get((dr, dc))
            if desired is None:
                return []
            yield from self._turn_towards(desired)
            yield from self.move_forward()
            cur = nxt
        (dr, dc) = (ti - cur[0], tj - cur[1])
        desired = DIR_MAP.get((dr, dc))
        if desired:
            yield from self._turn_towards(desired)
        yield from self.move_forward()
        yield from self.pick_up()
        yield from self.done()
        return []

    def drop_item(self, _a: str, _o: str):
        """
High-level action: drop object _o if currently held.
Implements (:action drop_item :parameters (?a - agent ?o - target))
    """
        target_tokens = self._parse_name(_o).split()
        if self.inventory:
            inv_tokens = self._parse_name(self.inventory).split()
            if inv_tokens == target_tokens:
                yield from self.drop()
        return []

    def open_target_door(self, _a: str, _d: str):
        """
    High-level action: open the door _d and signal completion.
    Implements (:action open_target_door :parameters (?a - agent ?d - door))
    """
        yield from self.discover_door(_a, _d)
        yield from self.navigate_to_door(_a, _d)
        yield from self._ensure_door_open(_d)
        yield from self.done()
        return []

    def discover_door(self, _a: str, _d: str):
        """
    High-level action: discover the door _d.
    Implements (:action discover_door :parameters (?a - agent ?d - door))
    """
        door_tokens = self._parse_name(_d).split()
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        while True:
            full = self.full_grid
            if any((all((tok in str(cell).split() for tok in door_tokens)) for cell in full.flat)):
                return []
            pos = self._agent_coords()
            (R, C) = full.shape
            frontiers: list[tuple[int, int]] = []
            for ((i, j), cell) in np.ndenumerate(full):
                if str(cell) != 'unseen':
                    continue
                for (dr, dc) in DIRS:
                    (ni, nj) = (i + dr, j + dc)
                    if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                        frontiers.append((ni, nj))
                        break
            if not frontiers:
                for c in range(C):
                    if self._is_passable(full[0, c]):
                        frontiers.append((0, c))
                    if self._is_passable(full[R - 1, c]):
                        frontiers.append((R - 1, c))
                for r in range(R):
                    if self._is_passable(full[r, 0]):
                        frontiers.append((r, 0))
                    if self._is_passable(full[r, C - 1]):
                        frontiers.append((r, C - 1))
            goals = [g for g in dict.fromkeys(frontiers) if g != pos]
            if not goals:
                return []
            path = self._shortest_path(pos, goals)
            if not path or len(path) < 2:
                return []
            for (cur, nxt) in zip(path, path[1:]):
                (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
                desired = DIR_MAP.get((dr, dc))
                if desired is None:
                    return []
                yield from self._turn_towards(desired)
                yield from self.move_forward()

    def navigate_to_door(self, _a: str, _d: str):
        """
    High-level action: navigate to the door _d (adjacent cell).
    Implements (:action navigate_to_door :parameters (?a - agent ?d - door))
    """
        door_tokens = self._parse_name(_d).split()
        full = self.full_grid
        door_positions = [pos for (pos, cell) in np.ndenumerate(full) if all((tok in str(cell).split() for tok in door_tokens))]
        if not door_positions:
            return []
        door_pos = door_positions[0]
        start = self._agent_coords()
        (R, C) = full.shape
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        adj: list[tuple[int, int]] = []
        for (dr, dc) in DIR_MAP:
            (ni, nj) = (door_pos[0] + dr, door_pos[1] + dc)
            if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                adj.append((ni, nj))
        if start in adj:
            return []
        path = self._shortest_path(start, adj)
        if not path or len(path) < 2:
            return []
        pos = start
        for nxt in path[1:]:
            (dr, dc) = (nxt[0] - pos[0], nxt[1] - pos[1])
            desired = DIR_MAP.get((dr, dc))
            if desired is None:
                return []
            yield from self._turn_towards(desired)
            yield from self.move_forward()
            pos = nxt
        return []

    def open_door(self, _a: str, _d: str):
        """
High-level action: open the door _d.
Implements (:action open_door :parameters (?a - agent ?d - door))
    """
        yield from self.discover_door(_a, _d)
        yield from self.navigate_to_door(_a, _d)
        yield from self._ensure_door_open(_d)
        return []

    def ensure_door_open(self, _a: str, _d: str):
        """
High-level action: ensure the door _d is open (unlocking with key if necessary).
Implements (:action ensure_door_open :parameters (?a - agent ?d - door))
    """
        door_name = self._parse_name(_d)
        door_tokens = door_name.split()
        COLORS = ('red', 'green', 'blue', 'purple', 'yellow', 'grey')
        color_token = next((tok for tok in door_tokens if tok in COLORS), None)
        full = self.full_grid
        if color_token:
            door_cells = [(pos, cell) for (pos, cell) in np.ndenumerate(full) if all((tok in str(cell).split() for tok in door_tokens))]
            if door_cells and 'locked' in str(door_cells[0][1]).split():
                if not (self.inventory and color_token in self._parse_name(self.inventory).split()):
                    while True:
                        full = self.full_grid
                        key_cells = [(i, j) for ((i, j), cell) in np.ndenumerate(full) if cell.startswith('key') and color_token in str(cell).split()]
                        if key_cells:
                            break
                        yield from self._explore_frontier()
                    key_pos = key_cells[0]
                    DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
                    adj = []
                    (R, C) = full.shape
                    for (dr, dc) in DIR_MAP:
                        (ni, nj) = (key_pos[0] + dr, key_pos[1] + dc)
                        if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                            adj.append((ni, nj))
                    if not adj:
                        return []
                    start = self._agent_coords()
                    path = self._shortest_path(start, adj)
                    if not path or len(path) < 2:
                        return []
                    pos = start
                    for nxt in path[1:]:
                        (dr, dc) = (nxt[0] - pos[0], nxt[1] - pos[1])
                        desired = DIR_MAP.get((dr, dc))
                        if desired is None:
                            return []
                        yield from self._turn_towards(desired)
                        yield from self.move_forward()
                        pos = nxt
                    (dr, dc) = (key_pos[0] - pos[0], key_pos[1] - pos[1])
                    desired = DIR_MAP.get((dr, dc))
                    if desired:
                        yield from self._turn_towards(desired)
                    yield from self.pick_up()
        yield from self.discover_door(_a, _d)
        yield from self.navigate_to_door(_a, _d)
        yield from self._ensure_door_open(_d)
        return []

    def open_red_door(self, _a: str, _d: str):
        """
High-level action: open the red door _d and signal completion.
Implements (:action open_red_door :parameters (?a - agent ?d - door))
    """
        yield from self.ensure_door_open(_a, _d)
        yield from self.done()
        return []

    def open_blue_door(self, _a: str, _d: str):
        """
High-level action: open the blue door _d and signal completion.
Implements (:action open_blue_door :parameters (?a - agent ?d - door))
    """
        yield from self.open_target_door(_a, _d)
        return []

    def open_purple_door(self, _a: str, _d: str):
        """
High-level action: open the purple door _d and signal completion.
Implements (:action open_purple_door :parameters (?a - agent ?d - door))
    """
        yield from self.ensure_door_open(_a, _d)
        yield from self.done()
        return []

    def open_yellow_door(self, _a: str, _d: str):
        """
High-level action: open the yellow door _d.
Implements (:action open_yellow_door :parameters (?a - agent ?d - door))
    """
        yield from self.ensure_door_open(_a, _d)
        return []

    def open_grey_door(self, _a: str, _d: str):
        """
High-level action: open the grey door _d and signal completion.
Implements (:action open_grey_door :parameters (?a - agent ?d - door))
    """
        yield from self.ensure_door_open(_a, _d)
        yield from self.done()
        return []

    def remember_target(self, _a: str, _o: str) -> List[int]:
        """
High-level action: remember the target object _o.
Implements (:action remember_target :parameters (?a - agent ?o - obj))
    """
        self.target_tokens = self._parse_name(_o).split()
        return []

    def go_to_matching_end(self, _a: str, _e: str, _o: str):
        """
High-level action: move to region _e, go to the object matching the previously remembered target _o,
pick it up, and signal completion.
Implements (:action go_to_matching_end :parameters (?a - agent ?e - region ?o - obj))
    """
        desired = self._parse_name(_o).split()
        if not hasattr(self, 'target_tokens') or self.target_tokens != desired:
            self.target_tokens = desired
        region_tokens = self._parse_name(_e).split()
        DIR_REL = {'North': {'front': (-1, 0), 'back': (1, 0)}, 'East': {'front': (0, 1), 'back': (0, -1)}, 'South': {'front': (1, 0), 'back': (-1, 0)}, 'West': {'front': (0, -1), 'back': (0, 1)}}
        rel_map = DIR_REL.get(self.current_dir, {})
        ori = next((tok for tok in region_tokens if tok in rel_map), None)
        if ori is None:
            if 'end' in region_tokens and 'front' in rel_map:
                ori = 'front'
            elif 'start' in region_tokens and 'back' in rel_map:
                ori = 'back'
        if ori is None:
            return []
        (dr_end, dc_end) = rel_map[ori]
        full = self.full_grid
        start = self._agent_coords()
        (R, C) = full.shape
        seen = {start}
        queue = deque([start])
        while queue:
            (cr, cc) = queue.popleft()
            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                (nr, nc) = (cr + dr, cc + dc)
                if 0 <= nr < R and 0 <= nc < C and ((nr, nc) not in seen):
                    cell = full[nr, nc]
                    if str(cell) != 'unseen' and self._is_passable(str(cell)):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
        max_proj = max(((r - start[0]) * dr_end + (c - start[1]) * dc_end for (r, c) in seen))
        end_region = [pos for pos in seen if (pos[0] - start[0]) * dr_end + (pos[1] - start[1]) * dc_end == max_proj]
        if start not in end_region:
            path = self._shortest_path(start, end_region)
            if not path or len(path) < 2:
                return []
            DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
            cur = start
            for nxt in path[1:]:
                (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
                desired_dir = DIR_MAP.get((dr, dc))
                if desired_dir is None:
                    return []
                yield from self._turn_towards(desired_dir)
                yield from self.move_forward()
                cur = nxt
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        while True:
            full = self.full_grid
            pos = self._agent_coords()
            target_cells = [p for (p, cell) in np.ndenumerate(full) if all((tok in str(cell).split() for tok in self.target_tokens))]
            if target_cells:
                break
            frontiers = []
            for ((r, c), cell) in np.ndenumerate(full):
                if str(cell) != 'unseen':
                    continue
                for (dr, dc) in DIRS:
                    (nr, nc) = (r + dr, c + dc)
                    if 0 <= nr < R and 0 <= nc < C and self._is_passable(str(full[nr, nc])):
                        frontiers.append((nr, nc))
                        break
            if not frontiers:
                for cc in range(C):
                    if self._is_passable(str(full[0, cc])):
                        frontiers.append((0, cc))
                    if self._is_passable(str(full[R - 1, cc])):
                        frontiers.append((R - 1, cc))
                for rr in range(R):
                    if self._is_passable(str(full[rr, 0])):
                        frontiers.append((rr, 0))
                    if self._is_passable(str(full[rr, C - 1])):
                        frontiers.append((rr, C - 1))
            goals = [g for g in dict.fromkeys(frontiers) if g != pos]
            if not goals:
                return []
            path = self._shortest_path(pos, goals)
            if not path or len(path) < 2:
                return []
            for (cur, nxt) in zip(path, path[1:]):
                (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
                desired_dir = DIR_MAP.get((dr, dc))
                if desired_dir is None:
                    return []
                yield from self._turn_towards(desired_dir)
                yield from self.move_forward()
        pos = self._agent_coords()

        def proj(cell):
            return (cell[0] - pos[0]) * dr_end + (cell[1] - pos[1]) * dc_end
        (ti, tj) = max(target_cells, key=proj)
        entry_points = []
        for (dr, dc) in DIRS:
            (ni, nj) = (ti + dr, tj + dc)
            if 0 <= ni < R and 0 <= nj < C and self._is_passable(str(full[ni, nj])):
                entry_points.append((ni, nj))
        if not entry_points:
            return []
        start = self._agent_coords()
        if start not in entry_points:
            path = self._shortest_path(start, entry_points)
            if not path or len(path) < 2:
                return []
            cur = start
            for nxt in path[1:]:
                (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
                desired_dir = DIR_MAP.get((dr, dc))
                if desired_dir is None:
                    return []
                yield from self._turn_towards(desired_dir)
                yield from self.move_forward()
                cur = nxt
        start = self._agent_coords()
        (dr, dc) = (ti - start[0], tj - start[1])
        desired_dir = DIR_MAP.get((dr, dc))
        if desired_dir:
            yield from self._turn_towards(desired_dir)
        yield from self.move_forward()
        yield from self.pick_up()
        yield from self.done()
        return []

    def go_to_goal(self, _a: str, _d: str, _g: str):
        """
High-level action: after ensuring door _d is open, move through it into region _g and signal completion.
Implements (:action go_to_goal :parameters (?a - agent ?d - door ?g - region))
    """
        yield from self.ensure_door_open(_a, _d)
        door_tokens = self._parse_name(_d).split()
        region_tokens = self._parse_name(_g).split()
        full = self.full_grid
        door_positions = [pos for (pos, cell) in np.ndenumerate(full) if all((tok in str(cell).split() for tok in door_tokens))]
        if not door_positions:
            return []
        door_pos = door_positions[0]
        start = self._agent_coords()
        DIR_MAP = {(-1, 0): 'North', (1, 0): 'South', (0, -1): 'West', (0, 1): 'East'}
        (dr, dc) = (door_pos[0] - start[0], door_pos[1] - start[1])
        desired = DIR_MAP.get((dr, dc))
        if desired:
            yield from self._turn_towards(desired)
        yield from self.move_forward()
        yield from self.move_forward()
        yield from self.finish(_a, _g)
        return []

    def _explore_frontier(self):
        """
Yield a short sequence of moves towards an unseen frontier cell for exploration.
    """
        DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        full = self.full_grid
        pos = self._agent_coords()
        (R, C) = full.shape
        frontiers = []
        for ((i, j), cell) in np.ndenumerate(full):
            if str(cell) != 'unseen':
                continue
            for (dr, dc) in DIRS:
                (ni, nj) = (i + dr, j + dc)
                if 0 <= ni < R and 0 <= nj < C and self._is_passable(full[ni, nj]):
                    frontiers.append((ni, nj))
                    break
        if not frontiers:
            for c in range(C):
                if self._is_passable(full[0, c]):
                    frontiers.append((0, c))
                if self._is_passable(full[R - 1, c]):
                    frontiers.append((R - 1, c))
            for r in range(R):
                if self._is_passable(full[r, 0]):
                    frontiers.append((r, 0))
                if self._is_passable(full[r, C - 1]):
                    frontiers.append((r, C - 1))
        goals = [g for g in dict.fromkeys(frontiers) if g != pos]
        if not goals:
            return []
        path = self._shortest_path(pos, goals)
        if not path or len(path) < 2:
            return []
        DIR_MAP = {d: n for (d, n) in zip(DIRS, ['North', 'South', 'West', 'East'])}
        for (cur, nxt) in zip(path, path[1:]):
            (dr, dc) = (nxt[0] - cur[0], nxt[1] - cur[1])
            desired = DIR_MAP.get((dr, dc))
            if desired is None:
                return []
            yield from self._turn_towards(desired)
            yield from self.move_forward()