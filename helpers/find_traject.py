"""find trajectory utilities

author: Xuan Tung VU
"""

import numpy as np
from collections import deque


class RepresentativeTrajectoryBuilder:
    """build deterministic representative trajectories over precomputed masks

    the class operates on a flattened grid graph where each state has a list of
    successors (one per control). it provides:
    - "stay" trajectories inside one set,
    - transition trajectories between sets,
    - loop trajectories alternating between regions
    """

    def __init__(self, L_vals, P_vals, successors, mask_JD, mask_JD_inv,
                 mask_viabK2, mask_viabB,
                 mask_E2, mask_EB,
                 mask_capt_viabK2, mask_capt_viabB,
                 constraints=None,
                 start_points=None):
        """initialize all masks, graph data, and optional custom start points"""

        self.L_vals = L_vals
        self.P_vals = P_vals
        self.successors = successors

        self.grid = len(L_vals)
        self.n_states = self.grid * self.grid

        self.T_max = 40
        self.transition_tail_steps = 5
        self.interior_radius_goal = 25
        self.interior_radius_max = 25
        self.mask_JD = mask_JD.reshape(-1).astype(np.uint8)
        self.mask_JD_inv = mask_JD_inv.reshape(-1).astype(np.uint8)
        self.mask_viabK2 = mask_viabK2.reshape(-1).astype(np.uint8)
        self.mask_viabB = mask_viabB.reshape(-1).astype(np.uint8)
        self.mask_E2 = mask_E2.reshape(-1).astype(np.uint8)
        self.mask_EB = mask_EB.reshape(-1).astype(np.uint8)
        self.mask_E = np.logical_or(self.mask_E2, self.mask_EB).astype(np.uint8)


        self.mask_capt_viabK2 = mask_capt_viabK2.reshape(-1).astype(np.uint8)
        self.mask_capt_viabB  = mask_capt_viabB.reshape(-1).astype(np.uint8)

        if constraints is None:
            self.constraints = {}
        else:
            self.constraints = {
                name: [m.reshape(-1) for m in masks]
                for name, masks in constraints.items()
            }
        self._has_constraints = bool(self.constraints)

        self._visited = np.empty(self.n_states, dtype=bool)
        self._parent = np.empty(self.n_states, dtype=int)

        self.idx_E2_start = self._pick_central_point(self.mask_E2)
        self.idx_EB_start = self._pick_central_point(self.mask_EB)

        if start_points is None:
            self.start_points = {}
        else:
            mask_for_start = {
                "traject_O_D2_D1_stay": self.mask_JD,
                "traject_O_D2_D1": self.mask_JD,
                "traject_O_D1_D2_stay": self.mask_JD_inv,
                "traject_O_D1_D2": self.mask_JD_inv,
                "traject_R_D1_D2_stay": self.mask_E2,
                "traject_R_D1_D2": self.mask_E2,
                "traject_R_D2_D1_stay": self.mask_EB,
                "traject_R_D2_D1": self.mask_EB,
                "traject_capt_viab_D1": self.mask_capt_viabK2,
                "traject_capt_viab_D2":  self.mask_capt_viabB,
                "traject_loop_R_D1_D2": self.mask_E,
            }
            self.start_points = {}
            for name, lp in start_points.items():
                mask = mask_for_start.get(name, None)
                self.start_points[name] = self._coord_to_index(lp[0], lp[1], mask)

    def _coord_to_index(self, L, P, mask=None):
        """map physical coordinates to the nearest grid index

        if a mask is provided, the nearest admissible index in that mask is
        selected. if the mask is empty, fallback to the full grid nearest point
        """
        if mask is None:
            i = np.abs(self.L_vals - L).argmin()
            j = np.abs(self.P_vals - P).argmin()
            return int(i * self.grid + j)
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            i = np.abs(self.L_vals - L).argmin()
            j = np.abs(self.P_vals - P).argmin()
            return int(i * self.grid + j)
        coords = np.array([self._index_to_coord(idx) for idx in valid])
        dL = coords[:, 0] - L
        dP = coords[:, 1] - P
        dists = dL * dL + dP * dP
        idx_best = valid[np.argmin(dists)]
        return int(idx_best)

    def _index_to_coord(self, idx):
        """convert a flattened grid index back to (l, p) coordinates"""
        i = idx // self.grid
        j = idx % self.grid
        return self.L_vals[i], self.P_vals[j]

    def _pick_initial_point(self, mask, forbidden_mask):
        """pick a deterministic start index in `mask` excluding `forbidden_mask`"""
        valid = np.where((mask == 1) & (forbidden_mask == 0))[0]
        return int(valid[len(valid) // 2])

    def _pick_central_point(self, mask, fallback_mask=None):
        """pick the point closest to the geometric center of a mask"""
        valid = np.where(mask == 1)[0]
        if len(valid) == 0 and fallback_mask is not None:
            valid = np.where(fallback_mask == 1)[0]
        coords = np.array([self._index_to_coord(idx) for idx in valid])
        Lc = coords[:, 0].mean()
        Pc = coords[:, 1].mean()
        dists = (coords[:, 0] - Lc) ** 2 + (coords[:, 1] - Pc) ** 2
        idx_best = valid[np.argmin(dists)]
        return int(idx_best)

    def _successors_in_mask(self, idx, admissible_mask):
        """return valid successors of `idx` that stay inside `admissible_mask`"""
        row = self.successors[idx]
        return [int(s) for s in row if (s >= 0) and (admissible_mask[int(s)] == 1)]

    def _successors_all(self, idx):
        """return all valid successors of `idx` (excluding -1 placeholders)"""
        row = self.successors[idx]
        return [int(s) for s in row if s >= 0]

    def _apply_constraints(self, name, s):
        """check trajectory-specific constraints at state index `s`"""
        if not self._has_constraints:
            return True
        masks = self.constraints.get(name)
        if masks is None:
            return True
        for m in masks:
            if m[s] == 0:
                return False
        return True

    def _reconstruct_path(self, last_idx, parent):
        """reconstruct a path from a parent array ending at `last_idx`"""
        path = [int(last_idx)]
        cur = int(last_idx)
        while parent[cur] != -1:
            cur = int(parent[cur])
            path.append(cur)
        path.reverse()
        return path

    def _mask_margin(self, idx, mask, max_radius=None):
        """estimate how interior a point is inside `mask`

        the margin is the largest square ring radius around the point fully
        contained in the mask, clipped by `max_radius`
        """
        if mask[idx] == 0:
            return -1

        if max_radius is None:
            max_radius = self.interior_radius_max

        g = self.grid
        i = idx // g
        j = idx % g
        margin = 0

        for r in range(1, max_radius + 1):
            i0 = i - r
            i1 = i + r
            j0 = j - r
            j1 = j + r

            if i0 < 0 or i1 >= g or j0 < 0 or j1 >= g:
                break

            ok = True
            top = i0 * g
            bottom = i1 * g

            for col in range(j0, j1 + 1):
                if mask[top + col] == 0 or mask[bottom + col] == 0:
                    ok = False
                    break

            if not ok:
                break

            for row in range(i0 + 1, i1):
                base = row * g
                if mask[base + j0] == 0 or mask[base + j1] == 0:
                    ok = False
                    break

            if not ok:
                break

            margin = r

        return margin

    def _best_successor_in_mask(self, idx, admissible_mask):
        """choose the best next state inside a mask

        preference order:
        1) larger interior margin,
        2) non-self transition over self-loop
        """
        succs = self._successors_in_mask(idx, admissible_mask)
        if not succs:
            return None

        best_s = None
        best_score = (-1, -1)
        for s in succs:
            margin = self._mask_margin(s, admissible_mask)
            not_self = 1 if s != idx else 0
            score = (margin, not_self)
            if score > best_score:
                best_score = score
                best_s = int(s)

        return best_s

    def _append_tail_in_target(self, path_indices, target_mask, steps=None):
        """extend a path by a short tail that remains in the target mask"""
        if steps is None:
            steps = self.transition_tail_steps

        if not path_indices or steps <= 0:
            return path_indices

        idx = int(path_indices[-1])
        if target_mask[idx] != 1:
            return path_indices

        for _ in range(steps):
            nxt = self._best_successor_in_mask(idx, target_mask)
            if nxt is None:
                break
            path_indices.append(nxt)
            idx = nxt

        return path_indices

    def _compute_trajectory_stay(self, idx0, admissible_mask, max_steps=25):
        """generate a coordinate trajectory that stays inside one admissible mask"""
        if max_steps is None:
            max_steps = self.T_max
        traj = []
        idx = idx0
        index_to_coord = self._index_to_coord
        for _ in range(max_steps):
            traj.append(index_to_coord(idx))
            nxt = self._best_successor_in_mask(idx, admissible_mask)
            if nxt is None:
                break
            idx = nxt
        return traj
    
    def _compute_trajectory_stay_indices(self, idx0, admissible_mask, max_steps=25):
        """index-based variant of `_compute_trajectory_stay`"""
        if max_steps is None:
            max_steps = self.T_max
        traj = [int(idx0)]
        idx = int(idx0)
        for _ in range(max_steps):
            nxt = self._best_successor_in_mask(idx, admissible_mask)
            if nxt is None:
                break
            idx = nxt
            traj.append(int(idx))
        return traj


    def _shortest_path_bfs_global(self, idx_start, target_mask, max_steps, name):
        """find the shortest constrained path from a start index to a target mask"""
        visited = self._visited
        parent = self._parent
        visited.fill(False)
        parent.fill(-1)

        q = deque()
        q_append = q.append
        q_popleft = q.popleft
        succ_all = self._successors_all
        apply_constraints = self._apply_constraints

        q_append((idx_start, 0))
        visited[idx_start] = True

        if target_mask[idx_start] == 1:
            return [idx_start]

        while q:
            idx, dist = q_popleft()
            if dist >= max_steps:
                continue
            for s in succ_all(idx):
                if not visited[s]:
                    if not apply_constraints(name, s):
                        continue
                    visited[s] = True
                    parent[s] = idx
                    if target_mask[s] == 1:
                        path = [s]
                        cur = s
                        while parent[cur] != -1:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()
                        return path
                    q_append((s, dist + 1))
        return None

    def _path_to_target_prefer_interior(self, idx_start, target_mask, max_steps, name, min_margin=None):
        """find a short path to target while favoring interior arrival points"""
        if min_margin is None:
            min_margin = self.interior_radius_goal

        visited = self._visited
        parent = self._parent
        visited.fill(False)
        parent.fill(-1)

        q = deque([int(idx_start)])
        visited[idx_start] = True

        if target_mask[idx_start] == 1:
            return [int(idx_start)]

        succ_all = self._successors_all
        apply_constraints = self._apply_constraints

        best_idx = -1
        best_margin = -1
        best_depth = max_steps + 1

        depth = 0
        while q and depth < max_steps:
            level_size = len(q)
            level_best_idx = -1
            level_best_margin = -1

            for _ in range(level_size):
                idx = q.popleft()

                for s in succ_all(idx):
                    if visited[s]:
                        continue
                    if not apply_constraints(name, s):
                        continue

                    visited[s] = True
                    parent[s] = idx
                    q.append(s)

                    if target_mask[s] != 1:
                        continue

                    margin = self._mask_margin(s, target_mask)
                    step_depth = depth + 1

                    if margin > best_margin or (margin == best_margin and step_depth < best_depth):
                        best_idx = int(s)
                        best_margin = margin
                        best_depth = step_depth

                    if margin > level_best_margin:
                        level_best_margin = margin
                        level_best_idx = int(s)

            depth += 1

            # stop once a point is sufficiently interior at the current depth
            # to keep the path short
            if level_best_idx != -1 and level_best_margin >= min_margin:
                best_idx = level_best_idx
                break

        if best_idx == -1:
            return None

        return self._reconstruct_path(best_idx, parent)

    def _indices_to_traj(self, indices):
        """convert a list of indices to a list of (l, p) coordinates"""
        index_to_coord = self._index_to_coord
        return [index_to_coord(idx) for idx in indices]

    def trajectory_O_D2_D1_stay(self):
        """stay trajectory inside viability set of d2 starting from o_{d2,d1}"""
        idx0 = self.start_points.get(
            "traject_O_D2_D1_stay",
            self._pick_initial_point(self.mask_JD, self.mask_EB)
        )
        return self._compute_trajectory_stay(idx0, self.mask_viabB)

    def trajectory_O_D2_D1(self):
        """transition trajectory from o_{d2,d1} toward viability of d1"""
        idx0 = self.start_points.get(
            "traject_O_D2_D1",
            self._pick_initial_point(self.mask_JD, self.mask_EB)
        )
        path_indices = self._path_to_target_prefer_interior(
            idx_start=idx0,
            target_mask=self.mask_viabK2,
            max_steps=self.T_max,
            name="traject_O_D2_D1"
        )
        if path_indices is None:
            return self._compute_trajectory_stay(idx0, self.mask_viabB, max_steps=self.T_max)
        path_indices = self._append_tail_in_target(
            path_indices,
            self.mask_viabK2,
            steps=self.transition_tail_steps,
        )
        return self._indices_to_traj(path_indices)

    def trajectory_O_D1_D2_stay(self):
        """stay trajectory inside viability set of d1 starting from o_{d1,d2}"""
        idx0 = self.start_points.get(
            "traject_O_D1_D2_stay",
            self._pick_initial_point(self.mask_JD_inv, self.mask_E2)
        )
        return self._compute_trajectory_stay(idx0, self.mask_viabK2)

    def trajectory_O_D1_D2(self):
        """transition trajectory from o_{d1,d2} toward viability of d2"""
        idx0 = self.start_points.get(
            "traject_O_D1_D2",
            self._pick_initial_point(self.mask_JD_inv, self.mask_E2)
        )
        path_indices = self._path_to_target_prefer_interior(
            idx_start=idx0,
            target_mask=self.mask_viabB,
            max_steps=self.T_max,
            name="traject_O_D1_D2"
        )
        if path_indices is None:
            return self._compute_trajectory_stay(idx0, self.mask_viabK2, max_steps=self.T_max)
        path_indices = self._append_tail_in_target(
            path_indices,
            self.mask_viabB,
            steps=self.transition_tail_steps,
        )
        return self._indices_to_traj(path_indices)

    def trajectory_R_D1_D2_stay(self):
        """stay trajectory inside reversible set r_{d1,d2}"""
        idx0 = self.start_points.get(
            "traject_R_D1_D2_stay",
            self.idx_E2_start
        )
        return self._compute_trajectory_stay(idx0, self.mask_E2, max_steps=25)

    def trajectory_R_D2_D1_stay(self):
        """stay trajectory inside reversible set r_{d2,d1}"""
        idx0 = self.start_points.get(
            "traject_R_D2_D1_stay",
            self.idx_EB_start
        )
        return self._compute_trajectory_stay(idx0, self.mask_EB, max_steps=25)

    def trajectory_R_D1_D2(self):
        """transition trajectory from r_{d1,d2} to r_{d2,d1}"""
        idx_start = self.start_points.get(
            "traject_R_D1_D2",
            self.idx_E2_start
        )

        path_indices = self._path_to_target_prefer_interior(
            idx_start=idx_start,
            target_mask=self.mask_EB,
            max_steps=self.T_max,
            name="traject_R_D1_D2",
        )
        if path_indices is None:
            print("No constrained path found for traject_R_D1_D2")
            return [self._index_to_coord(idx_start)]

        path_indices = self._append_tail_in_target(
            path_indices,
            self.mask_EB,
            steps=self.transition_tail_steps,
        )
        return self._indices_to_traj(path_indices)

    def trajectory_R_D2_D1(self):
        """transition trajectory from r_{d2,d1} to r_{d1,d2}"""
        idx_start = self.start_points.get(
            "traject_R_D2_D1",
            self.idx_EB_start
        )

        path_indices = self._path_to_target_prefer_interior(
            idx_start=idx_start,
            target_mask=self.mask_E2,
            max_steps=self.T_max,
            name="traject_R_D2_D1",
        )
        if path_indices is None:
            print("No constrained path found for traject_R_D2_D1")
            return [self._index_to_coord(idx_start)]

        path_indices = self._append_tail_in_target(
            path_indices,
            self.mask_E2,
            steps=self.transition_tail_steps,
        )
        return self._indices_to_traj(path_indices)

    def trajectory_capt_viab_D1(self):
        """trajectory from capture-only region toward viability set of d1"""
        forbidden = (
            self.mask_JD |
            self.mask_JD_inv |
            self.mask_viabB |
            self.mask_E2 |
            self.mask_EB
        ).astype(np.uint8)

        idx0 = self.start_points.get(
            "traject_capt_viab_D1",
            self._pick_initial_point(self.mask_capt_viabK2, forbidden)
        )

        path_indices = self._path_to_target_prefer_interior(
            idx_start=idx0,
            target_mask=self.mask_viabK2,
            max_steps=self.T_max,
            name="traject_capt_viab_D1"
        )
        if path_indices is None:
            return [self._index_to_coord(idx0)]

        path_indices = self._append_tail_in_target(
            path_indices,
            self.mask_viabK2,
            steps=self.transition_tail_steps,
        )
        return self._indices_to_traj(path_indices)



    def trajectory_capt_viab_D2(self):
        """trajectory from capture-only region toward viability set of d2"""
        forbidden = (
            self.mask_JD |
            self.mask_JD_inv |
            self.mask_viabK2 |
            self.mask_E2 |
            self.mask_EB
        ).astype(np.uint8)

        idx0 = self.start_points.get(
            "traject_capt_viab_D2",
            self._pick_initial_point(self.mask_capt_viabB, forbidden)
        )

        path_indices = self._path_to_target_prefer_interior(
            idx_start=idx0,
            target_mask=self.mask_viabB,
            max_steps=self.T_max,
            name="traject_capt_viab_D2"
        )
        if path_indices is None:
            return [self._index_to_coord(idx0)]

        path_indices = self._append_tail_in_target(
            path_indices,
            self.mask_viabB,
            steps=self.transition_tail_steps,
        )
        return self._indices_to_traj(path_indices)
    

    def _path_bfs_farthest_in_mask(self, idx_start, target_mask, max_steps, name):
        """
        run bfs from idx_start and keep exploring up to max_steps
        instead of stopping at the first target point, return the farthest
        reachable point inside target_mask for a richer trajectory
        """
        visited = self._visited
        parent = self._parent
        visited.fill(False)
        parent.fill(-1)

        dist = np.full(self.n_states, -1, dtype=np.int16)

        q = deque()
        q_append = q.append
        q_popleft = q.popleft
        succ_all = self._successors_all
        apply_constraints = self._apply_constraints

        q_append(idx_start)
        visited[idx_start] = True
        dist[idx_start] = 0

        best = idx_start if target_mask[idx_start] == 1 else -1
        best_d = 0 if best != -1 else -1
        best_margin = self._mask_margin(idx_start, target_mask) if best != -1 else -1

        while q:
            idx = q_popleft()
            d = dist[idx]
            if d >= max_steps:
                continue

            for s in succ_all(idx):
                if visited[s]:
                    continue
                if not apply_constraints(name, s):
                    continue

                visited[s] = True
                parent[s] = idx
                dist[s] = d + 1

                if target_mask[s] == 1:
                    margin = self._mask_margin(s, target_mask)
                    score = (margin, dist[s])
                    best_score = (best_margin, best_d)
                    if score > best_score:
                        best = s
                        best_d = dist[s]
                        best_margin = margin

                q_append(s)

        if best == -1:
            return None

        path = [best]
        cur = best
        while parent[cur] != -1:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path
    
    
    def trajectory_loop_R_D1_D2(self, N_changes=6, max_steps_leg=None, rich=True):
        """build a loop trajectory alternating between r_{d1,d2} and r_{d2,d1}

        parameters:
        - n_changes: number of region switches to attempt
        - max_steps_leg: per-leg path search budget
        - rich: if true, use longer "richer" legs; otherwise shortest legs
        """
        if max_steps_leg is None:
            max_steps_leg = self.T_max

        idx = self.start_points.get("traject_loop_R_D1_D2", None)

        if idx is None:
            idx = self.idx_E2_start if np.any(self.mask_E2) else self.idx_EB_start

        if self.mask_E2[idx] == 1:
            current_region = "E2"
        elif self.mask_EB[idx] == 1:
            current_region = "EB"
        else:
            idx = self._coord_to_index(*self._index_to_coord(idx), mask=self.mask_E)
            current_region = "E2" if self.mask_E2[idx] == 1 else "EB"

        traj_indices = [idx]
        switch_positions = []

        for k in range(N_changes):
            target_region = "EB" if current_region == "E2" else "E2"
            target_mask = self.mask_EB if target_region == "EB" else self.mask_E2

            if rich:
                path = self._path_bfs_farthest_in_mask(
                    idx_start=idx,
                    target_mask=target_mask,
                    max_steps=max_steps_leg,
                    name=f"loop_{current_region}_to_{target_region}"
                )
            else:
                path = self._shortest_path_bfs_global(
                    idx_start=idx,
                    target_mask=target_mask,
                    max_steps=max_steps_leg,
                    name=f"loop_{current_region}_to_{target_region}"
                )

            if path is None or len(path) <= 1:
                break

            traj_indices.extend(path[1:])
            switch_positions.append(len(traj_indices) - 1)

            idx = path[-1]
            current_region = target_region

        final_mask = self.mask_E2 if current_region == "E2" else self.mask_EB
        if final_mask[idx] != 1:
            L, P = self._index_to_coord(idx)
            idx = self._coord_to_index(L, P, mask=final_mask)

        tail = self._compute_trajectory_stay_indices(idx, final_mask, max_steps=25)
        traj_indices.extend(tail[1:])

        return self._indices_to_traj(traj_indices), switch_positions
