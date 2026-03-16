"""predator-prey experiment runner
author: Xuan Tung VU
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from viability import CaptureBasinCPU, PredPreyModel, ROptionSetsComputer, ViabilityKernelCPU
from helpers.plot import ViabilityPlotter
from helpers.find_traject import RepresentativeTrajectoryBuilder
from helpers.utils import ProgressBar, flatten_mask, rect_mask


RUN_OUTPUT_PATH = "./prey_predateur/prey_predateur.png"
RUN_SHOW = False
RUN_PLOT_TRAJ = False

# state-space bounds and domain rectangles
K = [(0.0, 50.0), (0.0, 26.0)]
Y_S_VALUES = [10.0, 11.5, 15.0]
REAPP_Y_S = 15.0
REAPP_U_MAX = 0.763
REAPP_FILENAME = "reapp.png"

# numerical settings
GRID_POINTS = 3000
T_VIAB = 10000
T_CAP = 40
N_CORES = 126
VERBOSE = False

TRAJECTORY_ORDER = [
    "traject_O_D2_D1_stay",
    "traject_O_D2_D1",
    "traject_O_D1_D2_stay",
    "traject_O_D1_D2",
    "traject_R_D1_D2_stay",
    "traject_R_D1_D2",
    "traject_R_D2_D1_stay",
    "traject_R_D2_D1",
    "traject_capt_viab_D1",
    "traject_capt_viab_D2",
    "traject_loop_R_D1_D2",
]

START_POINTS = {
    "traject_R_D2_D1": (13.0, 15.9),
    "traject_R_D2_D1_stay": (13.0, 15.9),
    "traject_R_D1_D2_stay": (13.0, 12.0),
    "traject_R_D1_D2": (13.0, 12.0),
    "traject_capt_viab_D1": (8.0, 22.0),
    "traject_capt_viab_D2": (16.0, 4.5),
    "traject_loop_R_D1_D2": (14.64, 8.91),
}


def build_domains(y_s: float):
    """build D1 and D2 bounds from one split value y_s"""
    y_split = float(y_s)
    d1 = [(0.0, 50.0), (0.0, y_split)]
    d2 = [(0.0, 50.0), (y_split, 26.0)]
    return d1, d2


def compute_viability_mask(kernel, successors, base_mask, *, n_cores):
    """compute viability mask from a given initialization mask"""
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
    model_params,
    n_cores,
    progress_bar=None,
):
    """compute capture basin mask for the predator-prey model"""
    capture = CaptureBasinCPU(
        K_bounds=K,
        U_values=model_params["U_values"],
        grid_points=GRID_POINTS,
        T=T_CAP,
        exclude_target=exclude_target,
        verbose=False,
    )
    capture.set_dynamics(
        PredPreyModel(
            model_params["b1"],
            model_params["k1"],
            model_params["k2"],
            model_params["d2"],
            model_params["dt"],
        )
    )

    capture.set_successors(successors)
    capture.set_target(
        flatten_mask(target_mask),
        target_bounds=K,
        target_grid_points=GRID_POINTS,
    )
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


def compute_R_sets(
    L_vals,
    P_vals,
    params,
    D1_bounds,
    D2_bounds,
    *,
    max_iter=20,
    n_cores=N_CORES,
    viability_bar=None,
    capture_bar=None,
):
    """compute viability, reversible sets, and successor graph"""
    kernel = ViabilityKernelCPU(
        K_bounds=K,
        U_values=params["U_values"],
        grid_points=GRID_POINTS,
        T=T_VIAB,
        verbose=False,
    )
    kernel.set_dynamics(
        PredPreyModel(params["b1"], params["k1"], params["k2"], params["d2"], params["dt"])
    )

    successors = kernel.compute_successors(parallel=True, n_cores=n_cores)
    if viability_bar is not None:
        viability_bar.advance(1)

    # build rectangular masks for d1 and d2
    mask_D1 = rect_mask(L_vals, P_vals, D1_bounds)
    mask_D2 = rect_mask(L_vals, P_vals, D2_bounds)

    viab_D1 = compute_viability_mask(kernel, successors, mask_D1, n_cores=n_cores)
    if viability_bar is not None:
        viability_bar.advance(1)

    viab_D2 = compute_viability_mask(kernel, successors, mask_D2, n_cores=n_cores)
    if viability_bar is not None:
        viability_bar.advance(1)
        viability_bar.done()

    if capture_bar is not None:
        capture_bar.set_total_steps(capture_bar.current + 1)

    # run the fixed-point loop for reversible sets
    ropt = ROptionSetsComputer(
        K_bounds=K,
        U_values=params["U_values"],
        grid_points=GRID_POINTS,
        T_viability=T_VIAB,
        T_capture=T_CAP,
        verbose=False,
    )
    ropt.set_dynamics(
        PredPreyModel(params["b1"], params["k1"], params["k2"], params["d2"], params["dt"])
    )

    R_D1_D2_flat, R_D2_D1_flat = ropt.run(
        D1_bounds=D1_bounds,
        D2_bounds=D2_bounds,
        max_iter=max_iter,
        successors=successors,
        parallel=True,
        n_cores=n_cores,
        verbose=False,
    )

    if capture_bar is not None:
        capture_bar.advance(1)

    if VERBOSE:
        print(f"[ROptionSetsComputer] converged={ropt.converged}, n_iter={ropt.n_iter}")

    R_D1_D2 = np.array(R_D1_D2_flat, copy=True).reshape(GRID_POINTS, GRID_POINTS)
    R_D2_D1 = np.array(R_D2_D1_flat, copy=True).reshape(GRID_POINTS, GRID_POINTS)
    return viab_D1, viab_D2, R_D1_D2, R_D2_D1, successors


def build_traj_list(trajs: Optional[Dict[str, object]]):
    """convert trajectory mapping to plotter list format"""
    if not trajs:
        return []

    traj_list = []
    for key in TRAJECTORY_ORDER:
        if key not in trajs or trajs[key] is None:
            continue
        value = trajs[key]
        if isinstance(value, tuple) and len(value) == 2:
            traj_list.append((value[0], value[1]))
        else:
            traj_list.append((value,))
    return traj_list


def plot_viab_and_sets(
    L_vals,
    P_vals,
    *,
    D1_bounds,
    D2_bounds,
    model_params,
    viab_D1_mask,
    viab_D2_mask,
    O_D2_D1_mask,
    O_D1_D2_mask,
    R_D1_D2_mask,
    R_D2_D1_mask,
    viab_D1_label,
    viab_D2_label,
    trajs,
    output_path: Path,
    show: bool,
    split_legend_by_domain: bool = False,
    legend_loc: str = "upper right",
    d2_legend_center=None,
    d1_legend_center=None,
    d1_legend_loc: str = "center",
    only_viab_d1_legend: bool = False,
    phase_vectors: bool = False,
    phase_model: Optional[str] = None,
    phase_params: Optional[Dict[str, float]] = None,
    phase_control=(0.0, 0.0),
):
    """plot viability, option, and reversible sets with trajectories"""
    traj_list = build_traj_list(trajs)
    d2_group_loc = "center" if d2_legend_center is not None else "upper right"
    d2_group_transform = "data" if d2_legend_center is not None else None
    label_list = (
        [viab_D1_label, None, None, None, None, None]
        if only_viab_d1_legend
        else [
            viab_D1_label,
            viab_D2_label,
            r"$\mathcal{O}(D_1,D_2)$",
            r"$\mathcal{O}(D_2,D_1)$",
            r"$\mathcal{R}(D_1,D_2)$",
            r"$\mathcal{R}(D_2,D_1)$",
        ]
    )

    plotter = ViabilityPlotter(
        L_vals_list=[L_vals] * 6,
        P_vals_list=[P_vals] * 6,
        V_list=[
            viab_D1_mask,
            viab_D2_mask,
            O_D1_D2_mask,
            O_D2_D1_mask,
            R_D1_D2_mask,
            R_D2_D1_mask,
        ],
        alpha_list=[0.15, 0.15, 0.4, 0.4, 0.6, 0.6],
        color_list=[
            "blue",
            "red",
            "cyan",
            "darkorange",
            "green",
            "purple",
        ],
        label_list=label_list,
        show_x_labels=[True] * 6,
        x_constraint_values=[D2_bounds[1][1], D1_bounds[1][1]],
        traj_list=traj_list,
        k=2,
        layer_types=["viab", "viab", "o", "o", "r", "r"],
        legend_groups=(
            [[1, 3, 5], [0, 2, 4]]
            if split_legend_by_domain
            else None
        ),
        legend_group_locs=(
            [d2_group_loc, d1_legend_loc]
            if split_legend_by_domain
            else None
        ),
        legend_group_anchors=(
            [
                d2_legend_center,
                (
                    d1_legend_center
                    if d1_legend_center is not None
                    else (8.0, 5.0)
                )
                if d1_legend_loc == "center"
                else None,
            ]
            if split_legend_by_domain
            else None
        ),
        legend_group_anchor_transforms=(
            [d2_group_transform, "data" if d1_legend_loc == "center" else None]
            if split_legend_by_domain
            else None
        ),
        legend_loc=legend_loc,
        x_label="X (Prey population size)",
        y_label="Y (Predator population size)",
        domain_label_positions=[
            (-0.6, 0.5 * (D1_bounds[1][0] + D1_bounds[1][1])),
            (-0.6, 0.5 * (D2_bounds[1][0] + D2_bounds[1][1])),
        ],
        equilibrium_prey_predateur=True,
        equilibrium_params=model_params,
        phase_vectors=phase_vectors,
        phase_model=phase_model,
        phase_params=phase_params,
        phase_control=phase_control,
        phase_color="black",
        phase_linestyle="--",
    )
    plotter.plot(title="", save_path=str(output_path), show=show)


def main(
    *,
    plot_traj: bool = RUN_PLOT_TRAJ,
    output_path: str = RUN_OUTPUT_PATH,
    show: bool = RUN_SHOW,
    y_s_values=Y_S_VALUES,
):
    """run full predator-prey workflow and export one plot per Y_s value"""
    b1 = 0.65
    k1 = 0.06
    k2 = 0.08
    d2 = 0.32
    dt = 0.1

    u_grid = np.linspace(-0.5, 0.5, 50)
    U_values = [[uX, uY] for uX in u_grid for uY in u_grid]
    params = dict(b1=b1, k1=k1, k2=k2, d2=d2, dt=dt, U_values=U_values)
    max_iter = 20

    L_vals = np.linspace(K[0][0], K[0][1], GRID_POINTS)
    P_vals = np.linspace(K[1][0], K[1][1], GRID_POINTS)

    output_path_obj = Path(output_path)
    output_root = output_path_obj.parent if output_path_obj.suffix else output_path_obj
    output_root.mkdir(parents=True, exist_ok=True)

    for y_s in y_s_values:
        y_s = float(y_s)
        D1_bounds, D2_bounds = build_domains(y_s)

        print(f"\n[RUN] Y_s = {y_s:.2f}")
        viability_bar = ProgressBar("VIABILITY", 3)
        capture_bar = ProgressBar("CAPTURE", 2 * max_iter + 2, start_immediately=False)

        viab_D1, viab_D2, R_D1_D2, R_D2_D1, successors = compute_R_sets(
            L_vals,
            P_vals,
            params,
            D1_bounds,
            D2_bounds,
            max_iter=max_iter,
            n_cores=N_CORES,
            viability_bar=viability_bar,
            capture_bar=capture_bar,
        )

        capture_bar.set_total_steps(capture_bar.current + 2)
        capt_viab_D1 = compute_capture_mask(
            successors,
            target_mask=viab_D1,
            exclude_mask=None,
            exclude_target=True,
            model_params=params,
            n_cores=N_CORES,
            progress_bar=capture_bar,
        )
        capt_viab_D2 = compute_capture_mask(
            successors,
            target_mask=viab_D2,
            exclude_mask=None,
            exclude_target=True,
            model_params=params,
            n_cores=N_CORES,
            progress_bar=capture_bar,
        )
        capture_bar.done()

        # combine viability and capture masks to form option sets
        O_D1_D2 = np.logical_and(viab_D1, capt_viab_D2).astype(np.uint8)
        O_D2_D1 = np.logical_and(viab_D2, capt_viab_D1).astype(np.uint8)

        trajs = None
        if plot_traj:
            # compute one representative loop trajectory on r_{d1,d2}
            traj = RepresentativeTrajectoryBuilder(
                L_vals,
                P_vals,
                successors,
                O_D2_D1,
                O_D1_D2,
                viab_D1,
                viab_D2,
                R_D1_D2,
                R_D2_D1,
                capt_viab_D1,
                capt_viab_D2,
                constraints=None,
                start_points=START_POINTS,
            )
            loop_traj, switch_positions = traj.trajectory_loop_R_D1_D2(
                N_changes=3,
                max_steps_leg=100,
                rich=False,
            )
            trajs = {"traject_loop_R_D1_D2": (loop_traj, switch_positions)}

        output_file = output_root / f"{int(y_s)}.png"
        is_15_png = output_file.stem == "15"
        viab_d1_label = r"$\operatorname{Viab}(D_1)$"
        viab_d2_label = None if is_15_png else r"$\operatorname{Viab}(D_2)$"
        d2_legend_center = (8.0, 22.0) if abs(y_s - 11.5) < 1e-12 else None
        d1_legend_center = (7.0, 3.0) if is_15_png else None

        plot_viab_and_sets(
            L_vals,
            P_vals,
            D1_bounds=D1_bounds,
            D2_bounds=D2_bounds,
            model_params=params,
            viab_D1_mask=viab_D1,
            viab_D2_mask=viab_D2,
            O_D2_D1_mask=O_D2_D1,
            O_D1_D2_mask=O_D1_D2,
            R_D1_D2_mask=R_D1_D2,
            R_D2_D1_mask=R_D2_D1,
            viab_D1_label=viab_d1_label,
            viab_D2_label=viab_d2_label,
            trajs=trajs,
            output_path=output_file,
            show=show,
            split_legend_by_domain=True if is_15_png else not is_15_png,
            d2_legend_center=d2_legend_center,
            d1_legend_center=d1_legend_center,
            only_viab_d1_legend=is_15_png,
            phase_vectors=abs(y_s - 10.0) < 1e-12,
            phase_model="predator_prey",
            phase_params=params,
        )
        print(f"[PLOT] saved: {output_file}")

    # additional reapp figure:
    # same sets as the base workflow, but with Y_s=15 and a different control upper bound
    u_grid_reapp = np.linspace(-0.5, REAPP_U_MAX, 50)
    U_values_reapp = [[uX, uY] for uX in u_grid_reapp for uY in u_grid_reapp]

    params_reapp = dict(
        b1=b1,
        k1=k1,
        k2=k2,
        d2=d2,
        dt=dt,
        U_values=U_values_reapp,
    )
    y_s_reapp = float(REAPP_Y_S)
    D1_bounds_reapp, D2_bounds_reapp = build_domains(y_s_reapp)

    print(f"\n[RUN] REAPP (Y_s = {y_s_reapp:.2f}, u_max = {REAPP_U_MAX:.2f})")
    viability_bar = ProgressBar("VIABILITY", 3)
    capture_bar = ProgressBar("CAPTURE", 2 * max_iter + 2, start_immediately=False)

    viab_D1, viab_D2, R_D1_D2, R_D2_D1, successors = compute_R_sets(
        L_vals,
        P_vals,
        params_reapp,
        D1_bounds_reapp,
        D2_bounds_reapp,
        max_iter=max_iter,
        n_cores=N_CORES,
        viability_bar=viability_bar,
        capture_bar=capture_bar,
    )

    capture_bar.set_total_steps(capture_bar.current + 2)
    capt_viab_D1 = compute_capture_mask(
        successors,
        target_mask=viab_D1,
        exclude_mask=None,
        exclude_target=True,
        model_params=params_reapp,
        n_cores=N_CORES,
        progress_bar=capture_bar,
    )
    capt_viab_D2 = compute_capture_mask(
        successors,
        target_mask=viab_D2,
        exclude_mask=None,
        exclude_target=True,
        model_params=params_reapp,
        n_cores=N_CORES,
        progress_bar=capture_bar,
    )
    capture_bar.done()

    O_D1_D2 = np.logical_and(viab_D1, capt_viab_D2).astype(np.uint8)
    O_D2_D1 = np.logical_and(viab_D2, capt_viab_D1).astype(np.uint8)

    trajs = None
    if plot_traj:
        traj = RepresentativeTrajectoryBuilder(
            L_vals,
            P_vals,
            successors,
            O_D2_D1,
            O_D1_D2,
            viab_D1,
            viab_D2,
            R_D1_D2,
            R_D2_D1,
            capt_viab_D1,
            capt_viab_D2,
            constraints=None,
            start_points=START_POINTS,
        )
        loop_traj, switch_positions = traj.trajectory_loop_R_D1_D2(
            N_changes=3,
            max_steps_leg=100,
            rich=False,
        )
        trajs = {"traject_loop_R_D1_D2": (loop_traj, switch_positions)}

    reapp_output_file = output_root / REAPP_FILENAME
    plot_viab_and_sets(
        L_vals,
        P_vals,
        D1_bounds=D1_bounds_reapp,
        D2_bounds=D2_bounds_reapp,
        model_params=params_reapp,
        viab_D1_mask=viab_D1,
        viab_D2_mask=viab_D2,
        O_D2_D1_mask=O_D2_D1,
        O_D1_D2_mask=O_D1_D2,
        R_D1_D2_mask=R_D1_D2,
        R_D2_D1_mask=R_D2_D1,
        viab_D1_label=r"$\operatorname{Viab}(D_1)$",
        viab_D2_label=r"$\operatorname{Viab}(D_2)$",
        trajs=trajs,
        output_path=reapp_output_file,
        show=show,
        split_legend_by_domain=True,
        d2_legend_center=(6.0, 24.0),
        d1_legend_loc="lower right",
        legend_loc="lower right",
        phase_vectors=False,
        phase_model="predator_prey",
        phase_params=params_reapp,
        phase_control=(0.0, 0.0),
    )
    print(f"[PLOT] saved: {reapp_output_file}")


if __name__ == "__main__":
    main(plot_traj=False)
