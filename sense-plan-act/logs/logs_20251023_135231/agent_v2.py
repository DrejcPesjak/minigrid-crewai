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
        self._stream_regions: Optional[List[Tuple[int, int]]] = None

    def turn_left(self) -> List[int]:
        return [0]

    def turn_right(self) -> List[int]:
        return [1]

    def move_forward(self) -> List[int]:
        return [2]

    def pick_up(self, _a: str, _b: str, _r: str):
        """PDDL action pick_up: pick up the specified box _b, stopping adjacent and facing it."""
        if self.inventory is not None:
            yield from self.drop()
        yield from self.reach_goal(_a, _b)
        yield from [3]

    def drop(self) -> List[int]:
        return [4]

    def toggle(self) -> List[int]:
        return [5]

    def done(self, _a: str, _d: str):
        """PDDL action done: step through the target door to succeed the mission."""
        yield from self.nav_to(_a, _d)
        yield from self.move_forward()

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

    def pick_up_obj(self, _a: str, _k: str):
        """PDDL action pick_up_obj: navigate to the specified key _k and pick it up."""
        name = self._parse_name(_k)
        target_tokens = set(name.split())
        if self.inventory is not None and set(self._parse_name(self.inventory).split()) == target_tokens:
            return []
        yield from self.reach_goal(_a, _k)
        yield from [3]

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
        """Navigate the agent to the specified target object g, exploring as needed.
Stops adjacent to the object and faces it, opening only unlocked doors when exploring."""
        target_tokens = set(self._parse_name(g).split())
        while True:
            goal_cells: List[Tuple[int, int]] = []
            (R, C) = self.full_grid.shape
            for r in range(R):
                for c in range(C):
                    parts = set(self.full_grid[r, c].split())
                    if target_tokens.issubset(parts):
                        goal_cells.append((r, c))
            if goal_cells:
                goal_pos = goal_cells[0]
                break
            door_opened = False
            for r in range(R):
                for c in range(C):
                    parts = self.full_grid[r, c].split()
                    if parts and parts[0] == 'door' and ('closed' in parts) and ('locked' not in parts):
                        for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                            nbr = (r + dr, c + dc)
                            if self._cell_passable(nbr[0], nbr[1]):
                                try:
                                    yield from self._navigate_to_cell(nbr)
                                    yield from self._face_direction(-dr, -dc)
                                    yield from self.toggle()
                                    yield from self.move_forward()
                                    door_opened = True
                                    break
                                except RuntimeError:
                                    continue
                        if door_opened:
                            break
                if door_opened:
                    break
            if door_opened:
                continue
            try:
                (cell, unseen) = self._find_frontier()
            except RuntimeError:
                raise RuntimeError(f'reach_goal: target {g} not found after full exploration')
            yield from self._navigate_to_cell(cell)
            dr = unseen[0] - cell[0]
            dc = unseen[1] - cell[1]
            yield from self._face_direction(dr, dc)
            (fr, fc) = self._front_coords()
            if 0 <= fr < R and 0 <= fc < C:
                parts = self.full_grid[fr, fc].split()
                if parts and parts[0] == 'door' and ('closed' in parts) and ('locked' not in parts):
                    yield from self.toggle()
            yield from self.move_forward()
        for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nbr = (goal_pos[0] + dr, goal_pos[1] + dc)
            if self._cell_passable(nbr[0], nbr[1]):
                try:
                    yield from self._navigate_to_cell(nbr)
                    yield from self._face_direction(goal_pos[0] - nbr[0], goal_pos[1] - nbr[1])
                    return
                except RuntimeError:
                    continue
        raise RuntimeError(f'reach_goal: no passable neighbor for target at {goal_pos}')

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
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while True:
            (R, C) = self.full_grid.shape
            crosses: list[tuple[int, int]] = [(r, c) for r in range(R) for c in range(C) if is_crossing(r, c)]
            if crosses:
                start = self._agent_coords()
                frontier = deque([start])
                came_from: Dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
                target: Optional[tuple[int, int]] = None
                while frontier:
                    curr = frontier.popleft()
                    if curr in crosses:
                        target = curr
                        break
                    for (dr, dc) in directions:
                        nbr = (curr[0] + dr, curr[1] + dc)
                        if nbr not in came_from and self._cell_passable(nbr[0], nbr[1]):
                            came_from[nbr] = curr
                            frontier.append(nbr)
                if target is not None:
                    yield from self._navigate_to_cell(target)
                    return
            (cell, unseen) = self._find_frontier()
            yield from self._navigate_to_cell(cell)
            dr = unseen[0] - cell[0]
            dc = unseen[1] - cell[1]
            yield from self._face_direction(dr, dc)
            yield from self.move_forward()

    def goto_goal(self, _a: str):
        """PDDL action goto_goal: navigate to the goal object, exploring as needed, then step onto it."""
        yield from self.reach_goal(_a, 'goal')
        yield from self.move_forward()

    def reach_target(self, _a: str, _d: str):
        """PDDL action reach_target: navigate to the specified door _d, stopping adjacent and facing it."""
        target_name = self._parse_name(_d)
        target_tokens = set(target_name.split())
        self._target_door_tokens = target_tokens
        (R, C) = self.full_grid.shape
        door_cells: List[Tuple[int, int]] = []
        for r in range(R):
            for c in range(C):
                parts = set(self.full_grid[r, c].split())
                if 'door' in parts and parts.issuperset(target_tokens):
                    door_cells.append((r, c))
        if not door_cells:
            while True:
                try:
                    (cell, unseen) = self._find_frontier()
                except RuntimeError:
                    raise RuntimeError(f'reach_target: door {_d} not found after full exploration')
                yield from self._navigate_to_cell(cell)
                (dr, dc) = (unseen[0] - cell[0], unseen[1] - cell[1])
                yield from self._face_direction(dr, dc)
                yield from self.move_forward()
                door_cells = []
                for rr in range(R):
                    for cc in range(C):
                        parts = set(self.full_grid[rr, cc].split())
                        if 'door' in parts and parts.issuperset(target_tokens):
                            door_cells.append((rr, cc))
                if door_cells:
                    break
        door_pos = door_cells[0]
        (fr, fc) = door_pos
        parts = set(self.full_grid[fr, fc].split())
        if ('closed' in parts or 'locked' in parts) and self.inventory in target_tokens:
            neighbors = []
            for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nbr = (fr + dr, fc + dc)
                if self._cell_passable(nbr[0], nbr[1]):
                    neighbors.append(nbr)
            if not neighbors:
                raise RuntimeError(f'reach_target: no passable neighbor to open door at {door_pos}')
            nbr = neighbors[0]
            yield from self._navigate_to_cell(nbr)
            yield from self._face_direction(fr - nbr[0], fc - nbr[1])
            yield from self.toggle()
            yield from self.move_forward()
            yield from self.turn_right()
            yield from self.turn_right()
            door_pos = door_pos
        neighbors: List[Tuple[int, int]] = []
        for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nbr = (door_pos[0] + dr, door_pos[1] + dc)
            if self._cell_passable(nbr[0], nbr[1]):
                neighbors.append(nbr)
        if not neighbors:
            raise RuntimeError(f'reach_target: no passable neighbor for door at {door_pos}')
        last_err: Optional[RuntimeError] = None
        for nbr in neighbors:
            try:
                yield from self._navigate_to_cell(nbr)
                yield from self._face_direction(door_pos[0] - nbr[0], door_pos[1] - nbr[1])
                return
            except RuntimeError as e:
                last_err = e
                continue
        raise RuntimeError(f'reach_target: unable to reach any neighbor of door at {door_pos}')

    def move(self, _a: str, _from: str, _to: str):
        """Navigate the agent from region _from to the specified adjacent region _to."""
        to = self._parse_name(_to)
        if to == 'gap':
            yield from self.goto_crossing(_a)
        elif to == 'goal':
            yield from self.goto_goal(_a)
        else:
            raise RuntimeError(f'move: unknown target region {_to}')

    def _stream_region_cells(self):
        """Return the ordered list of bank and stepping-stone region cells for crossing a stream.
Regions alternate between the agent's starting bank row, the barrier (stream) row,
and the opposite bank, based on visible passable cells in the barrier row.
Barrier row is determined between the agent and the goal if the goal is known,
otherwise the row with the most lava."""
        grid = self.full_grid
        (R, C) = grid.shape
        (agent_r, _) = self._agent_coords()
        goal_positions = []
        for r in range(R):
            for c in range(C):
                parts = set(grid[r, c].split())
                if 'goal' in parts:
                    goal_positions.append((r, c))
        if goal_positions:
            goal_r = goal_positions[0][0]
            (lo, hi) = sorted((agent_r, goal_r))
            candidates = []
            for r in range(lo + 1, hi):
                if any((grid[r, c].split()[0] == 'lava' for c in range(C))):
                    candidates.append(r)
            if candidates:
                lava_counts = [sum((1 for c in range(C) if grid[r, c].split()[0] == 'lava')) for r in candidates]
                barrier_row = candidates[int(np.argmax(lava_counts))]
            else:
                lava_counts = [sum((1 for cell in grid[r] if cell.split()[0] == 'lava')) for r in range(R)]
                barrier_row = int(np.argmax(lava_counts))
            bank0 = agent_r
            bank1 = goal_r
        else:
            lava_counts = [sum((1 for cell in grid[r] if cell.split()[0] == 'lava')) for r in range(R)]
            barrier_row = int(np.argmax(lava_counts))
            if agent_r < barrier_row:
                bank0 = agent_r
                bank1 = barrier_row + 1
            else:
                bank0 = agent_r
                bank1 = barrier_row - 1
            if not 0 <= bank1 < R:
                raise RuntimeError(f'_stream_region_cells: computed bank row {bank1} out of bounds')
        stepping_cols = [c for c in range(C) if self._cell_passable(barrier_row, c)]
        stepping_cols.sort()
        if not stepping_cols:
            raise RuntimeError('_stream_region_cells: no passable cells found on barrier row')
        regions: List[Tuple[int, int]] = []
        regions.append((bank0, stepping_cols[0]))
        for (idx, c) in enumerate(stepping_cols):
            regions.append((barrier_row, c))
            if idx % 2 == 0:
                regions.append((bank1, c))
            else:
                regions.append((bank0, c))
        return regions

    def cross_stream(self, _a: str, _from: str, _to: str):
        """Cross the stream one region step from region _from to adjacent region _to along stepping stones/banks."""
        m_from = re.search('\\d+', _from)
        m_to = re.search('\\d+', _to)
        if not m_from or not m_to:
            raise RuntimeError(f'cross_stream: could not parse region indices from {_from}, {_to}')
        from_idx = int(m_from.group())
        to_idx = int(m_to.group())
        if self._stream_regions is None:
            while True:
                try:
                    self._stream_regions = self._stream_region_cells()
                    break
                except RuntimeError:
                    (cell, unseen) = self._find_frontier()
                    yield from self._navigate_to_cell(cell)
                    dr = unseen[0] - cell[0]
                    dc = unseen[1] - cell[1]
                    yield from self._face_direction(dr, dc)
                    yield from self.move_forward()
            if not self._stream_regions:
                raise RuntimeError('cross_stream: no stream regions found')
        regions = self._stream_regions
        max_idx = len(regions) - 1
        if from_idx < 0 or to_idx < 0 or from_idx > max_idx or (to_idx > max_idx):
            raise RuntimeError(f'cross_stream: region index out of bounds from {from_idx} to {to_idx} with only {len(regions)} regions')
        start = regions[from_idx]
        goal = regions[to_idx]
        if self._agent_coords() != start:
            yield from self._navigate_to_cell(start)
        dr = goal[0] - start[0]
        dc = goal[1] - start[1]
        yield from self._face_direction(dr, dc)
        yield from self.move_forward()

    def nav_to(self, _a: str, _d: str):
        """PDDL action nav_to: navigate to the specified door _d, opening it if needed, stopping adjacent and facing it."""
        yield from self.reach_target(_a, _d)
        (fr, fc) = self._front_coords()
        (R, C) = self.full_grid.shape
        if not (0 <= fr < R and 0 <= fc < C):
            raise RuntimeError('nav_to: agent is facing out of bounds')
        parts = self.full_grid[fr, fc].split()
        if not parts or parts[0] != 'door':
            raise RuntimeError(f'nav_to: front cell {parts} is not a door')
        target_tokens = set(self._parse_name(_d).split())
        if not set(parts).issuperset(target_tokens):
            raise RuntimeError(f'nav_to: front door {parts} does not match target {_d}')
        if 'closed' in parts or 'locked' in parts:
            yield from self.toggle()

    def pick_up_item(self, _a: str, i: str):
        """PDDL action pick_up_item: navigate to the specified item i, stopping adjacent and facing it, then pick it up."""
        yield from self.reach_goal(_a, i)
        yield from self.pick_up()

    def toggle_door(self, _a: str, _d: str):
        """PDDL action toggle_door: toggle the specified door (open if closed, close if open), ignoring colour."""
        yield from self.reach_target(_a, _d)
        (fr, fc) = self._front_coords()
        (R, C) = self.full_grid.shape
        if not (0 <= fr < R and 0 <= fc < C):
            raise RuntimeError('toggle_door: agent is facing out of bounds')
        parts = self.full_grid[fr, fc].split()
        if not parts or parts[0] != 'door':
            raise RuntimeError(f'toggle_door: front cell {parts} is not a door')
        yield from self.toggle()

    def open_door(self, _a: str, _d: str):
        """PDDL action open_door: open the specified door _d."""
        yield from self.nav_to(_a, _d)

    def move_through(self, _a: str, _d: str, _r1: str, _r2: str):
        """PDDL action move_through: move through the specified door _d after positioning adjacent and facing it."""
        yield from self.nav_to(_a, _d)
        yield from self.move_forward()

    def open_blue_door(self, _a: str):
        """PDDL action open_blue_door: open the blue door."""
        yield from self.open_door(_a, 'door_blue', '_r1', '_r2')
        yield from self.move_through(_a, 'door_blue', '_r1', '_r2')

    def open_red_door(self, _a: str):
        """PDDL action open_red_door: open the red door."""
        yield from self.open_door(_a, 'door_red', '_r1', '_r2')
        yield from self.move_through(_a, 'door_red', '_r1', '_r2')

    def open_yellow_door(self, _a: str):
        """PDDL action open_yellow_door: open the yellow door and move through it."""
        yield from self.open_door(_a, 'door_yellow')
        yield from self.move_through(_a, 'door_yellow', '_r1', '_r2')

    def open_green_door(self, _a: str):
        """PDDL action open_green_door: open the green door and move through it."""
        yield from self.open_door(_a, 'door_green', '_r1', '_r2')
        yield from self.move_through(_a, 'door_green', '_r1', '_r2')

    def wall_ahead(self) -> bool:
        """Return True if the cell directly in front of the agent is a wall."""
        (fr, fc) = self._front_coords()
        (R, C) = self.full_grid.shape
        if not (0 <= fr < R and 0 <= fc < C):
            return False
        parts = self.full_grid[fr, fc].split()
        return parts and parts[0] == 'wall'

    def complete_memory_task(self, _ag: str):
        """PDDL action complete_memory_task: go to the end of the hallway, scan along the wall to find the opening, 
    observe the object on the north side through that opening, then go to the matching object on the south side and step onto it (matching by color)."""
        yield from self._face_direction(-1, 0)
        (R, C) = self.full_grid.shape
        while True:
            (fr, fc) = self._front_coords()
            if not (0 <= fr < R and 0 <= fc < C) or self.wall_ahead():
                break
            yield from self.move_forward()
        (wall_r, wall_c) = self._front_coords()
        if not (0 <= wall_r < R and 0 <= wall_c < C):
            raise RuntimeError('complete_memory_task: cannot locate wall at end of hallway; out of bounds')
        yield from self._face_direction(0, 1)
        lateral = 0
        skip = {'unseen', 'empty', 'floor', 'wall', 'lava', 'door', 'agent', 'goal'}
        while True:
            obs_r = wall_r - 1
            obs_c = wall_c + lateral
            if 0 <= obs_r < R and 0 <= obs_c < C:
                north_parts = self.full_grid[obs_r, obs_c].split()
                if len(north_parts) >= 2 and north_parts[0] not in skip:
                    break
            if self.wall_ahead():
                raise RuntimeError('complete_memory_task: no opening found in wall to observe north object')
            yield from self.move_forward()
            lateral += 1
        color = north_parts[1]
        yield from self._face_direction(1, 0)
        while True:
            (fr, fc) = self._front_coords()
            if not (0 <= fr < R and 0 <= fc < C):
                raise RuntimeError(f'complete_memory_task: out of bounds walking south looking for object of color {color}')
            parts = self.full_grid[fr, fc].split()
            if parts and color in parts:
                yield from self.move_forward()
                return
            if self.wall_ahead():
                raise RuntimeError(f'complete_memory_task: no matching object of color {color} found before end of hallway')
            yield from self.move_forward()

    def open_box(self, _a: str, _b: str, _k: str):
        """PDDL action open_box: open the specified box _b containing key _k."""
        key_tokens = set(self._parse_name(_k).split())
        (R, C) = self.full_grid.shape
        for r in range(R):
            for c in range(C):
                if key_tokens.issubset(self.full_grid[r, c].split()):
                    return []
        box_tokens = set(self._parse_name(_b).split())
        box_pos = None
        for r in range(R):
            for c in range(C):
                parts = set(self.full_grid[r, c].split())
                if 'box' in parts and box_tokens.issubset(parts):
                    box_pos = (r, c)
                    break
            if box_pos is not None:
                break
        if box_pos is None:
            return []
        for (dr, dc) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nbr = (box_pos[0] + dr, box_pos[1] + dc)
            if self._cell_passable(nbr[0], nbr[1]):
                yield from self._navigate_to_cell(nbr)
                yield from self._face_direction(-dr, -dc)
                yield from self.toggle()
                return
        raise RuntimeError(f'open_box: no passable neighbor to open box at {box_pos}')