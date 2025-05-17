import numpy as np
import heapq
from collections import deque
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
    
    # --------------------------------------------------
    def _global_pos(self) -> tuple[int, int]:
        loc = np.argwhere(self.full_grid == "agent")
        if loc.size == 0:
            raise RuntimeError("agent not found in full_grid")
        return tuple(loc[0])

    def _astar_path(self, targets: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        """Generic A*: shortest path from agent to any coord in *targets*."""
        if not targets:
            return None

        H, W  = self.full_grid.shape
        start = self._global_pos()
        h     = lambda p: min(abs(p[0]-t[0]) + abs(p[1]-t[1]) for t in targets)

        open_q = [(h(start), 0, start, [start])]       # (f, g, node, path)
        closed = set()

        while open_q:
            f, g, node, path = heapq.heappop(open_q)
            if node in closed:
                continue
            closed.add(node)
            if node in targets:
                return path
            nr, nc = node
            for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
                r, c = nr + dr, nc + dc
                if not (0 <= r < H and 0 <= c < W):
                    continue
                cell = self.full_grid[r, c]
                if cell in ("wall", "lava"):
                    continue                 # impassable
                step_cost = 1 if cell != "unseen" else 2   # bias against unknowns
                heapq.heappush(open_q,
                               (g + step_cost + h((r, c)),
                                g + step_cost,
                                (r, c),
                                path + [(r, c)]))
        return None                          # unreachable

    def _frontier_cells(self) -> set[tuple[int, int]]:
        """All safe squares that border at least one unseen square."""
        H, W = self.full_grid.shape
        frontier = set()
        for r in range(H):
            for c in range(W):
                if self.full_grid[r, c] in ("wall", "lava", "unseen"):
                    continue
                for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and self.full_grid[nr, nc] == "unseen":
                        frontier.add((r, c))
                        break
        return frontier
    
    def _is_goal(self, cell: str) -> bool:
            return isinstance(cell, str) and cell.startswith("goal")
    # --------------------------------------------------
    # main high-level routine
    # --------------------------------------------------
    def _farthest_reachable_safe(self) -> list[tuple[int, int]] | None:
        """Shortest path to the *most distant* reachable, already-observed safe cell."""
        H, W   = self.full_grid.shape
        start  = self._global_pos()
        q      = deque([start])
        pred   = {start: None}
        dist   = {start: 0}
        far    = start

        while q:
            r, c = q.popleft()
            for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cell = self.full_grid[nr, nc]
                if cell in ("wall", "lava", "unseen"):
                    continue          # impassable or unknown
                nxt = (nr, nc)
                if nxt not in pred:   # first time we reach it – shortest path
                    pred[nxt] = (r, c)
                    dist[nxt] = dist[(r, c)] + 1
                    q.append(nxt)
                    if dist[nxt] > dist[far]:
                        far = nxt

        if far == start:             # nowhere else to go
            return None

        # reconstruct path back-to-front
        path = []
        node = far
        while node is not None:
            path.append(node)
            node = pred[node]
        path.reverse()
        return path

    # --------------------------------------------------
    # main high-level routine
    # --------------------------------------------------
    def cross_lava(self) -> list[int]:
        """Plan a route: goal if visible, else go as far as possible in known space."""
        # 1) direct path to any goal-looking square
        is_goal  = lambda cell: isinstance(cell, str) and cell.startswith("goal")
        goal_cells = {
            (r, c)
            for r in range(self.full_grid.shape[0])
            for c in range(self.full_grid.shape[1])
            if is_goal(self.full_grid[r, c])
        }
        path = self._astar_path(goal_cells)

        # 2) no goal yet → head for the most distant safe cell we already know
        if path is None:
            path = self._farthest_reachable_safe()

        # 3) still nothing useful → fallback “turn until safe” probe
        if path is None or len(path) < 2:
            cmds = []
            for _ in range(4):
                if not self.lava_ahead():
                    cmds.append(2)        # move_forward
                    break
                cmds.append(1)            # turn_right
            return cmds

        # convert chosen path into primitive turns + moves
        dir_order = ["East", "South", "West", "North"]
        right     = {d: dir_order[(i + 1) % 4] for i, d in enumerate(dir_order)}
        vec2dir   = {(0, 1): "East", (1, 0): "South", (0, -1): "West", (-1, 0): "North"}

        cmds         = []
        sim_dir      = self.current_dir
        cur_r, cur_c = path[0]

        for nxt_r, nxt_c in path[1:]:
            dr, dc   = nxt_r - cur_r, nxt_c - cur_c
            tgt_dir  = vec2dir[(dr, dc)]
            while sim_dir != tgt_dir:
                cmds.append(1)            # turn_right
                sim_dir = right[sim_dir]
            cmds.append(2)                # move_forward
            cur_r, cur_c = nxt_r, nxt_c

        return cmds
        
