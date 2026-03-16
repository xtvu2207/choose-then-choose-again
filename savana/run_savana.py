"""savanna experiment runner
author: Xuan Tung VU
"""

import sys
from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from viability import CaptureBasinCPU, ROptionSetsComputer, SavannaDynamics, ViabilityKernelCPU
from helpers.plot import ViabilityPlotter
from helpers.utils import ProgressBar, flatten_mask, rect_mask


# run configuration for output and scenario behavior
RUN_OUTPUT_PATH = "./savana/100.png"
RUN_SHOW = False
RUN_PLOT_TRAJ = True
RUN_MAX_ITER = 20

# state-space bounds k and rectangular domains d1, d2, d3
K = [(0.0, 116.0), (0.0, 10.0)]
D1 = [(0.0, 19.85), (2.45, 10.0)]
D2 = [(19.85, 116.0), (0.0, 2.45)]
D3 = [(19.85, 116.0), (2.45, 10.0)]

# numerical settings
GRID_POINTS = 1000
T_VIAB = 1000
T_CAP = 100
N_CORES = 126
VERBOSE = False

# trajectory post-processing settings
TARGET_TAIL_STEPS = 20
INTERSECTION_TAIL_STEPS = 20
TRAJ_INTERIOR_MARGIN_GOAL = 3
TRAJ_INTERIOR_MARGIN_MAX = 7
TRAJ_RESTRICT_TO_PLOT_UNION = True

def build_savanna_dynamics(params):
    """create the savannadynamics model from a parameter dictionary"""
    # keep model construction in one place to avoid duplicated parameter wiring
    return SavannaDynamics(params["W"], params["f"], params["dt"])


def pick_central_index_from_mask(L_vals, P_vals, mask_2d):
    """return the mask index closest to the mask centroid in (l, p)"""
    grid = len(L_vals)
    mask_flat = mask_2d.reshape(-1).astype(np.uint8)
    valid = np.where(mask_flat == 1)[0]
    if len(valid) == 0:
        raise ValueError("Empty mask")

    ii = valid // grid
    jj = valid % grid
    L_center = L_vals[ii].mean()
    P_center = P_vals[jj].mean()

    dist2 = (L_vals[ii] - L_center) ** 2 + (P_vals[jj] - P_center) ** 2
    return int(valid[np.argmin(dist2)])


def bfs_path_to_mask(successors, idx_start, target_mask_flat, max_steps=100):
    """find a shortest path from a start index to any point in target_mask_flat"""
    n_states = target_mask_flat.size
    visited = np.zeros(n_states, dtype=bool)
    parent = np.full(n_states, -1, dtype=np.int32)

    q = deque()
    q.append((int(idx_start), 0))
    visited[idx_start] = True

    if target_mask_flat[idx_start] == 1:
        return [int(idx_start)]

    while q:
        idx, dist = q.popleft()
        if dist >= max_steps:
            continue

        row = successors[idx]
        for s in row:
            s = int(s)
            if s < 0:
                continue
            if visited[s]:
                continue

            visited[s] = True
            parent[s] = idx

            if target_mask_flat[s] == 1:
                path = [s]
                cur = s
                while parent[cur] != -1:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return [int(x) for x in path]

            q.append((s, dist + 1))

    return None


def mask_margin(idx, mask_flat, grid, max_radius=TRAJ_INTERIOR_MARGIN_MAX):
    """compute how interior a state is in a binary mask

    the returned value is the largest square-ring radius fully inside the mask
    """
    if mask_flat[idx] == 0:
        return -1

    i = idx // grid
    j = idx % grid
    margin = 0

    for r in range(1, int(max_radius) + 1):
        i0 = i - r
        i1 = i + r
        j0 = j - r
        j1 = j + r

        if i0 < 0 or i1 >= grid or j0 < 0 or j1 >= grid:
            break

        ok = True
        top = i0 * grid
        bottom = i1 * grid

        for col in range(j0, j1 + 1):
            if mask_flat[top + col] == 0 or mask_flat[bottom + col] == 0:
                ok = False
                break
        if not ok:
            break

        for row in range(i0 + 1, i1):
            base = row * grid
            if mask_flat[base + j0] == 0 or mask_flat[base + j1] == 0:
                ok = False
                break
        if not ok:
            break

        margin = r

    return margin


def bfs_path_to_mask_prefer_interior(
    successors,
    idx_start,
    target_mask_flat,
    *,
    grid,
    max_steps=100,
    min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
    admissible_mask_flat=None,
):
    """bfs path to a target mask, preferring interior target states

    among reachable target nodes, the algorithm favors larger margin and then
    shorter depth. optional `admissible_mask_flat` restricts traversal
    """
    n_states = target_mask_flat.size
    visited = np.zeros(n_states, dtype=bool)
    parent = np.full(n_states, -1, dtype=np.int32)

    q = deque([int(idx_start)])
    visited[idx_start] = True

    if target_mask_flat[idx_start] == 1:
        return [int(idx_start)]

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
            row = successors[idx]
            for s in row:
                s = int(s)
                if s < 0 or visited[s]:
                    continue
                if admissible_mask_flat is not None and admissible_mask_flat[s] == 0:
                    continue

                visited[s] = True
                parent[s] = idx
                q.append(s)

                if target_mask_flat[s] != 1:
                    continue

                margin = mask_margin(s, target_mask_flat, grid)
                step_depth = depth + 1
                if margin > best_margin or (margin == best_margin and step_depth < best_depth):
                    best_idx = s
                    best_margin = margin
                    best_depth = step_depth

                if margin > level_best_margin:
                    level_best_margin = margin
                    level_best_idx = s

        depth += 1

        # stop as soon as a sufficiently interior point is found at this depth
        if level_best_idx != -1 and level_best_margin >= min_margin:
            best_idx = level_best_idx
            break

    if best_idx == -1:
        return None

    path = [int(best_idx)]
    cur = int(best_idx)
    while parent[cur] != -1:
        cur = int(parent[cur])
        path.append(cur)
    path.reverse()
    return path


def best_successor_in_mask(
    successors,
    idx,
    admissible_mask_flat,
    *,
    grid,
    avoid_visited=None,
):
    """select the best successor in-mask using deterministic ranking

    preference order:
    1) not yet visited,
    2) larger interior margin,
    3) non-self transition
    """
    row = successors[int(idx)]
    best_s = None
    best_score = (-1, -1, -1)
    for s in row:
        s = int(s)
        if s < 0 or admissible_mask_flat[s] == 0:
            continue

        margin = mask_margin(s, admissible_mask_flat, grid)
        not_self = 1 if s != idx else 0
        not_visited = 1 if (avoid_visited is None or s not in avoid_visited) else 0
        score = (not_visited, margin, not_self)
        if score > best_score:
            best_score = score
            best_s = s

    return best_s


def follow_in_mask_indices(successors, idx_start, admissible_mask_flat, n_steps, *, grid):
    """greedily continue inside a mask for a fixed number of steps"""
    if n_steps <= 0:
        return []

    path = []
    idx = int(idx_start)
    visited = {idx}
    for _ in range(int(n_steps)):
        next_idx = best_successor_in_mask(
            successors,
            idx,
            admissible_mask_flat,
            grid=grid,
            avoid_visited=visited,
        )
        if next_idx is None:
            break
        path.append(next_idx)
        visited.add(next_idx)
        idx = next_idx
    return path


def indices_to_traj(L_vals, P_vals, indices):
    """convert flattened state indices to coordinate trajectory points"""
    grid = len(L_vals)
    traj = []
    for idx in indices:
        i = int(idx) // grid
        j = int(idx) % grid
        traj.append((float(L_vals[i]), float(P_vals[j])))
    return traj


def compute_viability_mask(kernel, successors, base_mask, *, n_cores):
    """run viability iteration for a rectangular initialization mask"""
    # update the initial mask then run fixed-point viability iteration
    kernel.set_initial_viable(
        flatten_mask(base_mask),
        bounds=K,
        grid_points=GRID_POINTS,
    )
    kernel.iterate_viability(successors, parallel=True, n_cores=n_cores, verbose=False)
    return np.array(kernel.is_viable, copy=True).reshape(GRID_POINTS, GRID_POINTS)


def compute_capture_mask(
    successors,
    target_mask,
    *,
    exclude_mask=None,
    exclude_target=False,
    t_capture=T_CAP,
    model_params,
    n_cores,
    progress_bar=None,
):
    """compute a capture basin mask for a given target"""
    # capture kernel reuses the same successor graph built from viability kernel
    capture = CaptureBasinCPU(
        K_bounds=K,
        U_values=model_params["U_values"],
        grid_points=GRID_POINTS,
        T=t_capture,
        exclude_target=exclude_target,
        verbose=False,
    )
    capture.set_dynamics(build_savanna_dynamics(model_params))

    capture.set_successors(successors)
    capture.set_target(
        flatten_mask(target_mask),
        target_bounds=K,
        target_grid_points=GRID_POINTS,
    )

    # optional exclusion mask removes regions from final capture output
    if exclude_mask is not None:
        capture.set_exclude_mask(
            flatten_mask(exclude_mask),
            bounds=K,
            grid_points=GRID_POINTS,
        )

    capture.iterate_capture(parallel=True, n_cores=n_cores, verbose=False)

    if progress_bar is not None:
        progress_bar.advance(1)

    return np.array(capture.is_capture, copy=True).reshape(GRID_POINTS, GRID_POINTS)


def compute_reversible_sets(
    L_vals,
    P_vals,
    params,
    *,
    max_iter=RUN_MAX_ITER,
    t_capture=T_CAP,
    n_cores=N_CORES,
    successors=None,
    viab_masks=None,
    viability_bar=None,
    capture_bar=None,
):
    """compute viability and reversible sets for domain pairs

    if `successors` and `viab_masks` are provided, they are reused to avoid
    recomputation when running the same setup with different capture horizons
    """
    if successors is None or viab_masks is None:
        # build global kernel when successors or viability masks are not precomputed
        kernel = ViabilityKernelCPU(
            K_bounds=K,
            U_values=params["U_values"],
            grid_points=GRID_POINTS,
            T=T_VIAB,
            verbose=False,
        )
        kernel.set_dynamics(build_savanna_dynamics(params))

        if successors is None:
            successors = kernel.compute_successors(parallel=True, n_cores=n_cores)
            if viability_bar is not None:
                viability_bar.advance(1)

        if viab_masks is None:
            # compute baseline viability domains on shared grid
            viab_D1 = compute_viability_mask(kernel, successors, rect_mask(L_vals, P_vals, D1), n_cores=n_cores)
            if viability_bar is not None:
                viability_bar.advance(1)

            viab_D2 = compute_viability_mask(kernel, successors, rect_mask(L_vals, P_vals, D2), n_cores=n_cores)
            if viability_bar is not None:
                viability_bar.advance(1)

            viab_D3 = compute_viability_mask(kernel, successors, rect_mask(L_vals, P_vals, D3), n_cores=n_cores)
            if viability_bar is not None:
                viability_bar.advance(1)
                viability_bar.done()
        else:
            viab_D1 = np.asarray(viab_masks["viab_D1"])
            viab_D2 = np.asarray(viab_masks["viab_D2"])
            viab_D3 = np.asarray(viab_masks["viab_D3"])
    else:
        viab_D1 = np.asarray(viab_masks["viab_D1"])
        viab_D2 = np.asarray(viab_masks["viab_D2"])
        viab_D3 = np.asarray(viab_masks["viab_D3"])

    ropt = ROptionSetsComputer(
        K_bounds=K,
        U_values=params["U_values"],
        grid_points=GRID_POINTS,
        T_viability=T_VIAB,
        T_capture=t_capture,
        verbose=False,
    )
    ropt.set_dynamics(build_savanna_dynamics(params))

    # first reversible pair d2 <-> d3
    R_D2_D3_flat, R_D3_D2_flat = ropt.run(
        D1_bounds=D2,
        D2_bounds=D3,
        max_iter=max_iter,
        successors=successors,
        parallel=True,
        n_cores=n_cores,
        verbose=False,
    )
    if capture_bar is not None:
        capture_bar.advance(1)

    if VERBOSE:
        print(f"[ROptionSetsComputer D2<->D3] converged={ropt.converged}, n_iter={ropt.n_iter}")

    # second reversible pair d1 <-> d3
    R_D1_D3_flat, R_D3_D1_flat = ropt.run(
        D1_bounds=D1,
        D2_bounds=D3,
        max_iter=max_iter,
        successors=successors,
        parallel=True,
        n_cores=n_cores,
        verbose=False,
    )
    if capture_bar is not None:
        capture_bar.advance(1)

    if VERBOSE:
        print(f"[ROptionSetsComputer D1<->D3] converged={ropt.converged}, n_iter={ropt.n_iter}")

    R_D2_D3 = np.array(R_D2_D3_flat, copy=True).reshape(GRID_POINTS, GRID_POINTS)
    R_D3_D2 = np.array(R_D3_D2_flat, copy=True).reshape(GRID_POINTS, GRID_POINTS)
    R_D1_D3 = np.array(R_D1_D3_flat, copy=True).reshape(GRID_POINTS, GRID_POINTS)
    R_D3_D1 = np.array(R_D3_D1_flat, copy=True).reshape(GRID_POINTS, GRID_POINTS)

    return {
        "viab_D1": viab_D1,
        "viab_D2": viab_D2,
        "viab_D3": viab_D3,
        "R_D2_D3": R_D2_D3,
        "R_D3_D2": R_D3_D2,
        "R_D1_D3": R_D1_D3,
        "R_D3_D1": R_D3_D1,
        "successors": successors,
    }


def build_intersection_trajectories(L_vals, P_vals, successors, R_D2_D3, R_D3_D2, R_D1_D3, R_D3_D1):
    """build representative trajectories from/to the r-set intersection"""
    # define intersection and union masks used for trajectory routing
    inter_mask = np.logical_and(R_D3_D2, R_D3_D1).astype(np.uint8)
    union_plot_mask = np.logical_or.reduce([R_D2_D3, R_D3_D2, R_D1_D3, R_D3_D1]).astype(np.uint8)

    trajectories = {
        "intersection": inter_mask,
        "inter_to_R_D2_D3": None,
        "inter_to_R_D1_D3": None,
        "R_D2_D3_to_inter": None,
        "R_D1_D3_to_inter": None,
    }

    if inter_mask.sum() == 0:
        return trajectories

    grid = len(L_vals)
    idx_inter = pick_central_index_from_mask(L_vals, P_vals, inter_mask)

    R_D2_D3_flat = R_D2_D3.reshape(-1).astype(np.uint8)
    R_D1_D3_flat = R_D1_D3.reshape(-1).astype(np.uint8)
    inter_flat = inter_mask.reshape(-1).astype(np.uint8)
    union_plot_flat = union_plot_mask.reshape(-1).astype(np.uint8)

    admissible = union_plot_flat if TRAJ_RESTRICT_TO_PLOT_UNION else None

    # 1) from the intersection to d2 then continue inside d2
    path_inter_to_D2 = bfs_path_to_mask_prefer_interior(
        successors,
        idx_inter,
        R_D2_D3_flat,
        grid=grid,
        max_steps=T_CAP,
        min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
        admissible_mask_flat=admissible,
    )
    if path_inter_to_D2 is None:
        path_inter_to_D2 = bfs_path_to_mask_prefer_interior(
            successors,
            idx_inter,
            R_D2_D3_flat,
            grid=grid,
            max_steps=T_CAP,
            min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
            admissible_mask_flat=None,
        )
    if path_inter_to_D2 is None:
        path_inter_to_D2 = bfs_path_to_mask(successors, idx_inter, R_D2_D3_flat, max_steps=T_CAP)
    if path_inter_to_D2:
        tail_D2 = follow_in_mask_indices(
            successors,
            path_inter_to_D2[-1],
            R_D2_D3_flat,
            TARGET_TAIL_STEPS,
            grid=grid,
        )
        full_inter_to_D2 = path_inter_to_D2 + tail_D2
        trajectories["inter_to_R_D2_D3"] = indices_to_traj(L_vals, P_vals, full_inter_to_D2)

        # 2) from the last d2 point back to the intersection then continue there
        path_D2_to_inter = bfs_path_to_mask_prefer_interior(
            successors,
            full_inter_to_D2[-1],
            inter_flat,
            grid=grid,
            max_steps=T_CAP,
            min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
            admissible_mask_flat=admissible,
        )
        if path_D2_to_inter is None:
            path_D2_to_inter = bfs_path_to_mask_prefer_interior(
                successors,
                full_inter_to_D2[-1],
                inter_flat,
                grid=grid,
                max_steps=T_CAP,
                min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
                admissible_mask_flat=None,
            )
        if path_D2_to_inter is None:
            path_D2_to_inter = bfs_path_to_mask(
                successors,
                full_inter_to_D2[-1],
                inter_flat,
                max_steps=T_CAP,
            )
        if path_D2_to_inter:
            tail_inter_from_D2 = follow_in_mask_indices(
                successors,
                path_D2_to_inter[-1],
                inter_flat,
                INTERSECTION_TAIL_STEPS,
                grid=grid,
            )
            full_D2_to_inter = path_D2_to_inter + tail_inter_from_D2
            trajectories["R_D2_D3_to_inter"] = indices_to_traj(L_vals, P_vals, full_D2_to_inter)

    # 1) from the intersection to d1 then continue inside d1
    path_inter_to_D1 = bfs_path_to_mask_prefer_interior(
        successors,
        idx_inter,
        R_D1_D3_flat,
        grid=grid,
        max_steps=T_CAP,
        min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
        admissible_mask_flat=admissible,
    )
    if path_inter_to_D1 is None:
        path_inter_to_D1 = bfs_path_to_mask_prefer_interior(
            successors,
            idx_inter,
            R_D1_D3_flat,
            grid=grid,
            max_steps=T_CAP,
            min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
            admissible_mask_flat=None,
        )
    if path_inter_to_D1 is None:
        path_inter_to_D1 = bfs_path_to_mask(successors, idx_inter, R_D1_D3_flat, max_steps=T_CAP)
    if path_inter_to_D1:
        tail_D1 = follow_in_mask_indices(
            successors,
            path_inter_to_D1[-1],
            R_D1_D3_flat,
            TARGET_TAIL_STEPS,
            grid=grid,
        )
        full_inter_to_D1 = path_inter_to_D1 + tail_D1
        trajectories["inter_to_R_D1_D3"] = indices_to_traj(L_vals, P_vals, full_inter_to_D1)

        # 2) from the last d1 point back to the intersection then continue there
        path_D1_to_inter = bfs_path_to_mask_prefer_interior(
            successors,
            full_inter_to_D1[-1],
            inter_flat,
            grid=grid,
            max_steps=T_CAP,
            min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
            admissible_mask_flat=admissible,
        )
        if path_D1_to_inter is None:
            path_D1_to_inter = bfs_path_to_mask_prefer_interior(
                successors,
                full_inter_to_D1[-1],
                inter_flat,
                grid=grid,
                max_steps=T_CAP,
                min_margin=TRAJ_INTERIOR_MARGIN_GOAL,
                admissible_mask_flat=None,
            )
        if path_D1_to_inter is None:
            path_D1_to_inter = bfs_path_to_mask(
                successors,
                full_inter_to_D1[-1],
                inter_flat,
                max_steps=T_CAP,
            )
        if path_D1_to_inter:
            tail_inter_from_D1 = follow_in_mask_indices(
                successors,
                path_D1_to_inter[-1],
                inter_flat,
                INTERSECTION_TAIL_STEPS,
                grid=grid,
            )
            full_D1_to_inter = path_D1_to_inter + tail_inter_from_D1
            trajectories["R_D1_D3_to_inter"] = indices_to_traj(L_vals, P_vals, full_D1_to_inter)

    return trajectories


def build_traj_list(*, traj_data: Dict[str, object]):
    """convert trajectory dictionary to plotter-compatible list format"""
    traj_list = []

    if traj_data.get("inter_to_R_D2_D3") is not None:
        traj_list.append((traj_data["inter_to_R_D2_D3"],))
    if traj_data.get("inter_to_R_D1_D3") is not None:
        traj_list.append((traj_data["inter_to_R_D1_D3"],))
    if traj_data.get("R_D2_D3_to_inter") is not None:
        traj_list.append((traj_data["R_D2_D3_to_inter"],))
    if traj_data.get("R_D1_D3_to_inter") is not None:
        traj_list.append((traj_data["R_D1_D3_to_inter"],))

    return traj_list


def compute_option_masks(
    *,
    viab_D1,
    viab_D2,
    viab_D3,
    capt_viab_D1,
    capt_viab_D2,
    capt_viab_D3,
):
    """compute o-sets as viability and capture intersections"""
    return {
        "O_D3_D2": np.logical_and(viab_D3, capt_viab_D2).astype(np.uint8),
        "O_D2_D3": np.logical_and(viab_D2, capt_viab_D3).astype(np.uint8),
        "O_D3_D1": np.logical_and(viab_D3, capt_viab_D1).astype(np.uint8),
        "O_D1_D3": np.logical_and(viab_D1, capt_viab_D3).astype(np.uint8),
    }


def compute_capture_masks_for_domains(
    *,
    successors,
    viab_D1,
    viab_D2,
    viab_D3,
    t_capture,
    model_params,
    n_cores,
):
    """compute capture masks for each domain viability set"""
    targets = {
        "capt_viab_D1": viab_D1,
        "capt_viab_D2": viab_D2,
        "capt_viab_D3": viab_D3,
    }
    out = {}
    for name, target in targets.items():
        out[name] = compute_capture_mask(
            successors,
            target_mask=target,
            exclude_target=True,
            t_capture=t_capture,
            model_params=model_params,
            n_cores=n_cores,
        )
    return out


def plot_savana_ro_sets(
    *,
    L_vals,
    P_vals,
    masks,
    option_masks,
    save_path: Path,
    show: bool,
    legend_groups=None,
    legend_group_locs=None,
    legend_group_anchors=None,
    legend_group_anchor_transforms=None,
    legend_loc: str = "upper center",
):
    """plot r/o sets for a specific capture horizon variant"""
    # each index maps to one rendered layer in the same order as v_list
    color_list = [
        "green",    # 0 -> r_{d2,d3}
        "cyan",     # 1 -> r_{d3,d2}
        "orange",   # 2 -> o_{d3,d2}
        "magenta",  # 3 -> o_{d2,d3}
        "olive",    # 4 -> r_{d1,d3}
        "blue",     # 5 -> r_{d3,d1}
        "brown",    # 6 -> o_{d3,d1}
        "purple",   # 7 -> o_{d1,d3}
    ]

    label_list = [
        r"$\mathcal{R}(D_2,D_3)$",
        r"$\mathcal{R}(D_3,D_2)$",
        r"$\mathcal{O}(D_3,D_2)$",
        r"$\mathcal{O}(D_2,D_3)$",
        r"$\mathcal{R}(D_1,D_3)$",
        r"$\mathcal{R}(D_3,D_1)$",
        r"$\mathcal{O}(D_3,D_1)$",
        r"$\mathcal{O}(D_1,D_3)$",
    ]

    # if one r set equals one o set keep a single legend entry
    equal_pairs = [
        ("R_D2_D3", "O_D2_D3", 0, 3, 2, 3),
        ("R_D3_D2", "O_D3_D2", 1, 2, 3, 2),
        ("R_D1_D3", "O_D1_D3", 4, 7, 1, 3),
        ("R_D3_D1", "O_D3_D1", 5, 6, 3, 1),
    ]
    for r_key, o_key, r_idx, o_idx, i_dom, j_dom in equal_pairs:
        if np.array_equal(masks[r_key], option_masks[o_key]):
            label_list[r_idx] = (
                rf"$\mathcal{{R}}(D_{i_dom},D_{j_dom})"
                rf" = \mathcal{{O}}(D_{i_dom},D_{j_dom})$"
            )
            label_list[o_idx] = None

    plotter = ViabilityPlotter(
        L_vals_list=[L_vals] * 8,
        P_vals_list=[P_vals] * 8,
        V_list=[
            masks["R_D2_D3"],
            masks["R_D3_D2"],
            option_masks["O_D3_D2"],
            option_masks["O_D2_D3"],
            masks["R_D1_D3"],
            masks["R_D3_D1"],
            option_masks["O_D3_D1"],
            option_masks["O_D1_D3"],
        ],
        alpha_list=[0.7, 0.7, 0.45, 0.45, 0.7, 0.7, 0.45, 0.45],
        color_list=color_list,
        label_list=label_list,
        show_x_labels=[True] * 8,
        x_constraint_values=None,
        x_lines=[D2[0][0]],
        y_lines=[D3[1][0]],
        traj_list=None,
        k=10,
        layer_types=["r", "r", "o", "o", "r", "r", "o", "o"],
        legend_loc=legend_loc,
        legend_groups=legend_groups,
        legend_group_locs=legend_group_locs,
        legend_group_anchors=legend_group_anchors,
        legend_group_anchor_transforms=legend_group_anchor_transforms,
        show_base_x_line=False,
        x_min_plot=0.0,
        y_min_plot=0.0,
        x_label="T (Tree density)",
        y_label="G (Grass density)",
        domain_label_positions=[(9, 4), (60, 1), (90, 4)],
        intersection_masks_list=None,
    )
    plotter.plot(title="", save_path=str(save_path), show=show)


def main(
    *,
    plot_traj: bool = RUN_PLOT_TRAJ,
    output_path: str = RUN_OUTPUT_PATH,
    show: bool = RUN_SHOW,
):
    """run the full savanna workflow and save all configured figures"""
    # base model parameters
    W = 920.0
    f = 0.98
    dt = 0.1

    # build control grid for two-dimensional control vector [dt, dg]
    delta_G_values = np.linspace(0.0, 0.1, 50)
    delta_T_values = np.linspace(0.015, 0.2, 50)
    U_values = [[dT, dG] for dG in delta_G_values for dT in delta_T_values]

    params = {
        "W": W,
        "f": f,
        "dt": dt,
        "U_values": U_values,
    }

    # build state grid used across all kernels and masks
    L_vals = np.linspace(K[0][0], K[0][1], GRID_POINTS)
    P_vals = np.linspace(K[1][0], K[1][1], GRID_POINTS)

    viability_bar = ProgressBar("VIABILITY", 4)
    capture_bar = ProgressBar("ROPT", 2, start_immediately=False)

    # compute base reversible masks for default capture horizon
    masks = compute_reversible_sets(
        L_vals,
        P_vals,
        params,
        max_iter=RUN_MAX_ITER,
        t_capture=T_CAP,
        n_cores=N_CORES,
        viability_bar=viability_bar,
        capture_bar=capture_bar,
    )
    capture_bar.done()

    traj_list = []
    if plot_traj:
        # compute optional representative trajectories only when requested
        traj_data = build_intersection_trajectories(
            L_vals,
            P_vals,
            masks["successors"],
            masks["R_D2_D3"],
            masks["R_D3_D2"],
            masks["R_D1_D3"],
            masks["R_D3_D1"],
        )
        traj_list = build_traj_list(traj_data=traj_data)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plotter = ViabilityPlotter(
        L_vals_list=[L_vals] * 4,
        P_vals_list=[P_vals] * 4,
        V_list=[
            masks["R_D2_D3"],
            masks["R_D3_D2"],
            masks["R_D1_D3"],
            masks["R_D3_D1"],
        ],
        alpha_list=[0.7, 0.7, 0.7, 0.7],
        color_list=["green", "cyan", "olive", "blue"],
        label_list=[
            r"$\mathcal{R}(D_2,D_3)$",
            r"$\mathcal{R}(D_3,D_2)$",
            r"$\mathcal{R}(D_1,D_3)$",
            r"$\mathcal{R}(D_3,D_1)$",
        ],
        show_x_labels=[True, True, True, True],
        x_constraint_values=None,
        x_lines=[D2[0][0]],
        y_lines=[D3[1][0]],
        traj_list=traj_list,
        k=12,
        layer_types=["r", "r", "r", "r"],
        legend_fontsize=15,
        legend_groups=[[1], [0], [2, 3]],
        legend_group_locs=["upper right", "center", "center"],
        legend_group_anchors=[None, (90, 1), (20, 8)],
        legend_group_anchor_transforms=[None, "data", "data"],
        arrow_scale=0.02,
        arrow_linewidth=1.0,
        arrow_mutation_scale=10.0,
        show_base_x_line=False,
        x_min_plot=0.0,
        y_min_plot=0.0,
        x_label="T (Tree density)",
        y_label="G (Grass density)",
        domain_label_positions=[(9, 4), (60, 1.5), (90, 4)],
        intersection_masks_list=[masks["R_D3_D2"], masks["R_D3_D1"]],
        intersection_label=r"$\mathcal{R}(D_3,D_2) \cap \mathcal{R}(D_3,D_1)$",
        handoff_start_marker="X",
        handoff_start_marker_size=120,
        handoff_start_marker_edgecolor="black",
        n_skip=7
    )
    plotter.plot(title="", save_path=str(output_file), show=show)

    print(f"[PLOT] saved: {output_file}")

    # reuse viability masks and successors for alternative capture horizons
    viab_for_reuse = {
        "viab_D1": masks["viab_D1"],
        "viab_D2": masks["viab_D2"],
        "viab_D3": masks["viab_D3"],
    }

    for t_cap_variant, file_name in ((10, "10.png"), (20, "20.png")):
        # recompute reversible sets for each capture horizon variant
        masks_tcap = compute_reversible_sets(
            L_vals,
            P_vals,
            params,
            max_iter=RUN_MAX_ITER,
            t_capture=t_cap_variant,
            n_cores=N_CORES,
            successors=masks["successors"],
            viab_masks=viab_for_reuse,
        )

        # recompute capture and option masks for the same horizon
        capture_masks_tcap = compute_capture_masks_for_domains(
            successors=masks["successors"],
            viab_D1=masks["viab_D1"],
            viab_D2=masks["viab_D2"],
            viab_D3=masks["viab_D3"],
            t_capture=t_cap_variant,
            model_params=params,
            n_cores=N_CORES,
        )

        option_masks_tcap = compute_option_masks(
            viab_D1=masks["viab_D1"],
            viab_D2=masks["viab_D2"],
            viab_D3=masks["viab_D3"],
            capt_viab_D1=capture_masks_tcap["capt_viab_D1"],
            capt_viab_D2=capture_masks_tcap["capt_viab_D2"],
            capt_viab_D3=capture_masks_tcap["capt_viab_D3"],
        )

        # export comparison figure for current horizon
        tcap_output_file = output_file.parent / file_name
        legend_groups = None
        legend_group_locs = None
        legend_group_anchors = None
        legend_group_anchor_transforms = None
        legend_loc = "upper center"

        if t_cap_variant == 10:
            # split legend in 3 blocks for the 10.png figure
            legend_groups = [
                [4, 5],  # R(D1,D3)=O(D1,D3), R(D3,D1)=O(D3,D1)
                [2, 1],  # O(D3,D2), R(D3,D2)
                [0, 3],  # remaining entries
            ]
            legend_group_locs = ["center", "center", "center"]
            legend_group_anchors = [(53, 8), (100, 6), (90, 1)]
            legend_group_anchor_transforms = ["data", "data", "data"]
        elif t_cap_variant == 20:
            # split legend in 3 blocks for the 20.png figure
            legend_groups = [
                [4, 5],  # R(D1,D3)=O(D1,D3), R(D3,D1)=O(D3,D1)
                [2, 1],  # O(D3,D2), R(D3,D2)
                [0, 3],  # remaining entries
            ]
            legend_group_locs = ["center", "center", "center"]
            legend_group_anchors = [(55, 9.5), (72, 6), (90, 1)]
            legend_group_anchor_transforms = ["data", "data", "data"]

        plot_savana_ro_sets(
            L_vals=L_vals,
            P_vals=P_vals,
            masks=masks_tcap,
            option_masks=option_masks_tcap,
            save_path=tcap_output_file,
            show=show,
            legend_groups=legend_groups,
            legend_group_locs=legend_group_locs,
            legend_group_anchors=legend_group_anchors,
            legend_group_anchor_transforms=legend_group_anchor_transforms,
            legend_loc=legend_loc,
        )
        print(f"[PLOT] saved: {tcap_output_file}")


if __name__ == "__main__":
    main()
