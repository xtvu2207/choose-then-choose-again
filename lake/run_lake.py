"""lake experiment runner and plotting scenarios

author: Xuan Tung VU
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from viability import (
    CaptureBasinCPU,
    PhosphoreDynamics,
    ROptionSetsComputer,
    ViabilityKernelCPU,
)
from helpers.plot import ViabilityPlotter
from helpers.find_traject import RepresentativeTrajectoryBuilder
from helpers.utils import ProgressBar, flatten_mask, rect_mask


# run configuration for plot selection and output behavior
RUN_PLOTS = ("d1", "d2", "o", "s")
RUN_OUTPUT_DIR = "./lake"
RUN_SHOW = False
RUN_PLOT_TRAJ = True


# state-space bounds and rectangular domains
K = [(6.9, 100.0), (0.0, 25.0)]
D1 = [(6.9, 100.0), (0.0, 15.0)]
D2 = [(6.9, 100.0), (15.0, 25.0)]

# numerical settings
GRID_POINTS = 3000
T_VIAB = 10000
T_CAP = 40
N_CORES = 126
VERBOSE = False

# optional fixed start points to get reproducible trajectories
START_POINTS = {
    "traject_R_D2_D1": (13.0, 15.9),
    "traject_R_D2_D1_stay": (13.0, 15.9),
    "traject_R_D1_D2_stay": (13.0, 12.0),
    "traject_R_D1_D2": (13.0, 12.0),
    "traject_capt_viab_D1": (9.0, 20.0),
    "traject_capt_viab_D2": (15.0, 4.5),
    "traject_loop_R_D1_D2": (14.98, 7.86),
}


@dataclass(frozen=True)
class LayerStyle:
    """visual style of one plotted layer"""

    color: str
    alpha: float
    label: Optional[str]


@dataclass(frozen=True)
class PlotScenario:
    """plot configuration: visible layers, trajectories, and labels"""

    key: str
    layer_keys: Tuple[str, ...]
    trajectory_keys: Tuple[str, ...]
    label_overrides: Dict[str, Optional[str]]
    arrow_step: int = 2


# visual style by semantic layer key
LAYER_STYLES: Dict[str, LayerStyle] = {
    "viab_D1": LayerStyle("blue", 0.15, r"$\operatorname{Viab}(D_1)$"),
    "capt_viab_D1": LayerStyle(
        "darkgoldenrod",
        0.15,
        r"$\operatorname{Capt}^{40}_{K}(\operatorname{Viab}(D_1))$",
    ),
    "viab_D2": LayerStyle("red", 0.15, r"$\operatorname{Viab}(D_2)$"),
    "capt_viab_D2": LayerStyle(
        "dimgray",
        0.15,
        r"$\operatorname{Capt}^{40}_{K}(\operatorname{Viab}(D_2))$",
    ),
    "O_D1_D2": LayerStyle("cyan", 0.4, r"$\mathcal{O}(D_1,D_2)$"),
    "O_D2_D1": LayerStyle("darkorange", 0.4, r"$\mathcal{O}(D_2,D_1)$"),
    "R_D1_D2": LayerStyle("green", 0.6, r"$\mathcal{R}(D_1,D_2)$"),
    "R_D2_D1": LayerStyle("purple", 0.6, r"$\mathcal{R}(D_2,D_1)$"),
}

# plotting order for trajectories and their color slots
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

# per-figure scenario definitions used by main
SCENARIOS: Dict[str, PlotScenario] = {
    "d1": PlotScenario(
        key="d1",
        layer_keys=("viab_D1", "capt_viab_D1"),
        trajectory_keys=("traject_capt_viab_D1",),
        label_overrides={},
        arrow_step=2,
    ),
    "d2": PlotScenario(
        key="d2",
        layer_keys=("viab_D2", "capt_viab_D2"),
        trajectory_keys=("traject_capt_viab_D2",),
        label_overrides={},
        arrow_step=4,
    ),
    "o": PlotScenario(
        key="o",
        layer_keys=("viab_D1", "capt_viab_D1", "viab_D2", "capt_viab_D2", "O_D1_D2", "O_D2_D1"),
        trajectory_keys=(
            "traject_O_D2_D1_stay",
            "traject_O_D2_D1",
            "traject_O_D1_D2_stay",
            "traject_O_D1_D2",
        ),
        label_overrides={
            "viab_D1": None,
            "capt_viab_D1": None,
            "viab_D2": None,
            "capt_viab_D2": None,
        },
        arrow_step=4,
    ),
    "s": PlotScenario(
        key="s",
        layer_keys=(
            "viab_D1",
            "capt_viab_D1",
            "viab_D2",
            "capt_viab_D2",
            "O_D1_D2",
            "O_D2_D1",
            "R_D1_D2",
            "R_D2_D1",
        ),
        trajectory_keys=("traject_loop_R_D1_D2",),
        label_overrides={
            "viab_D1": None,
            "capt_viab_D1": None,
            "viab_D2": None,
            "capt_viab_D2": None,
            "O_D1_D2": None,
            "O_D2_D1": None,
        },
        arrow_step=6,
    ),
}




def compute_viability_mask(kernel, successors, base_mask, *, n_cores):
    """compute viability mask from an initial domain mask"""
    # this call updates the internal kernel mask before iteration
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
    """compute capture basin mask for a target set"""
    # capture kernel is configured with precomputed successors from viability kernel
    capture = CaptureBasinCPU(
        K_bounds=K,
        U_values=model_params["U_values"],
        grid_points=GRID_POINTS,
        T=T_CAP,
        exclude_target=exclude_target,
        verbose=False,
    )
    capture.set_successors(successors)
    capture.set_target(
        flatten_mask(target_mask),
        target_bounds=K,
        target_grid_points=GRID_POINTS,
    )
    # optional exclusion mask removes regions from final capture set
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


def compute_E_sets(
    L_vals,
    P_vals,
    params,
    *,
    max_iter=20,
    n_cores=N_CORES,
    viability_bar=None,
    capture_bar=None,
):
    """compute viability masks and reversible sets e for the lake model"""
    # build one global kernel and one successor graph reused across all sets
    kernel_global = ViabilityKernelCPU(
        K_bounds=K,
        U_values=params["U_values"],
        grid_points=GRID_POINTS,
        T=T_VIAB,
        verbose=False,
    )
    kernel_global.set_dynamics(
        PhosphoreDynamics(params["b"], params["r"], params["m"], params["q"], params["dt"])
    )

    successors_global = kernel_global.compute_successors(parallel=True, n_cores=n_cores)
    if viability_bar is not None:
        viability_bar.advance(1)

    # define initial rectangular domains on the same grid
    mask_D1 = rect_mask(L_vals, P_vals, D1)
    mask_D2 = rect_mask(L_vals, P_vals, D2)

    viab_D1_mask = compute_viability_mask(kernel_global, successors_global, mask_D1, n_cores=n_cores)
    if viability_bar is not None:
        viability_bar.advance(1)

    viab_D2_mask = compute_viability_mask(kernel_global, successors_global, mask_D2, n_cores=n_cores)
    if viability_bar is not None:
        viability_bar.advance(1)
        viability_bar.done()

    if capture_bar is not None:
        capture_bar.set_total_steps(capture_bar.current + 1)

    # run reversible option fixed-point solver on d1 <-> d2
    ropt = ROptionSetsComputer(
        K_bounds=K,
        U_values=params["U_values"],
        grid_points=GRID_POINTS,
        T_viability=T_VIAB,
        T_capture=T_CAP,
        verbose=False,
    )
    ropt.set_dynamics(
        PhosphoreDynamics(params["b"], params["r"], params["m"], params["q"], params["dt"])
    )

    R_D1_D2_flat, R_D2_D1_flat = ropt.run(
        D1_bounds=D1,
        D2_bounds=D2,
        max_iter=max_iter,
        successors=successors_global,
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
    E_mask = np.logical_or(R_D1_D2, R_D2_D1).astype(np.uint8)

    return viab_D1_mask, viab_D2_mask, R_D1_D2, R_D2_D1, E_mask, successors_global


def build_trajectory_getter(
    L_vals,
    P_vals,
    params,
    successors_global,
    masks: Dict[str, np.ndarray],
) -> Callable[[str], object]:
    """return a cached trajectory resolver by trajectory name"""
    # instantiate trajectory helper once with all required masks
    traj = RepresentativeTrajectoryBuilder(
        L_vals,
        P_vals,
        successors_global,
        masks["O_D2_D1"],
        masks["O_D1_D2"],
        masks["viab_D1"],
        masks["viab_D2"],
        masks["R_D1_D2"],
        masks["R_D2_D1"],
        masks["capt_viab_D1"],
        masks["capt_viab_D2"],
        constraints=None,
        start_points=START_POINTS,
    )

    trajectories = {
        "traject_O_D2_D1_stay": traj.trajectory_O_D2_D1_stay,
        "traject_O_D2_D1": traj.trajectory_O_D2_D1,
        "traject_O_D1_D2_stay": traj.trajectory_O_D1_D2_stay,
        "traject_O_D1_D2": traj.trajectory_O_D1_D2,
        "traject_R_D1_D2_stay": traj.trajectory_R_D1_D2_stay,
        "traject_R_D1_D2": traj.trajectory_R_D1_D2,
        "traject_R_D2_D1_stay": traj.trajectory_R_D2_D1_stay,
        "traject_R_D2_D1": traj.trajectory_R_D2_D1,
        "traject_capt_viab_D1": traj.trajectory_capt_viab_D1,
        "traject_capt_viab_D2": traj.trajectory_capt_viab_D2,
        "traject_loop_R_D1_D2": lambda: traj.trajectory_loop_R_D1_D2(
            N_changes=3,
            max_steps_leg=100,
            rich=False,
        ),
    }

    cache: Dict[str, object] = {}

    def get_trajectory(name: str):
        """compute and cache a trajectory on first access"""
        # lazy cache avoids recomputation when multiple scenarios reuse the same path
        if name not in trajectories:
            raise KeyError(f"Unknown trajectory: {name}")
        if name not in cache:
            cache[name] = trajectories[name]()
        return cache[name]

    return get_trajectory


def build_traj_list(trajs: Optional[Dict[str, object]]):
    """convert trajectory mapping to plotter trajectory list format"""
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


def layer_type_from_key(key: str) -> str:
    """infer rendering layer type from mask key naming convention"""
    if key.startswith("viab_"):
        return "viab"
    if key.startswith("capt_"):
        return "capt"
    if key.startswith("O_"):
        return "o"
    if key.startswith("R_"):
        return "r"
    return "other"


def plot_scenario(
    scenario: PlotScenario,
    L_vals: np.ndarray,
    P_vals: np.ndarray,
    masks: Dict[str, np.ndarray],
    trajs: Optional[Dict[str, object]],
    output_path: Path,
    show: bool,
):
    """render one scenario and write the figure to disk"""
    # select only layers defined by this scenario
    V_list = [masks[key] for key in scenario.layer_keys]
    layer_types = [layer_type_from_key(key) for key in scenario.layer_keys]
    color_list = [LAYER_STYLES[key].color for key in scenario.layer_keys]
    alpha_list = [LAYER_STYLES[key].alpha for key in scenario.layer_keys]

    label_list = []
    for key in scenario.layer_keys:
        label = LAYER_STYLES[key].label
        if key in scenario.label_overrides:
            label = scenario.label_overrides[key]
        label_list.append(label)

    # transform trajectory dict into plotter sequence format
    traj_list = build_traj_list(trajs)

    plotter = ViabilityPlotter(
        L_vals_list=[L_vals] * len(V_list),
        P_vals_list=[P_vals] * len(V_list),
        V_list=V_list,
        alpha_list=alpha_list,
        color_list=color_list,
        label_list=label_list,
        show_x_labels=[True] * len(V_list),
        x_constraint_values=[D2[1][1], D1[1][1]],
        traj_list=traj_list,
        k=scenario.arrow_step,
        layer_types=layer_types,
        x_label="External phosphorus load",
        y_label="Phosphorus mass in the lake",
    )
    plotter.plot(title=None, save_path=str(output_path), show=show)


def main(
    *,
    plots: Sequence[str] = RUN_PLOTS,
    output_dir: str = RUN_OUTPUT_DIR,
    show: bool = RUN_SHOW,
    plot_traj: bool = RUN_PLOT_TRAJ,
):
    """run selected lake scenarios and save all output images"""
    # validate scenario keys before running heavy computations
    unknown_plots = [key for key in plots if key not in SCENARIOS]
    if unknown_plots:
        raise ValueError(
            f"Unknown plot scenario(s): {unknown_plots}. "
            f"Available: {sorted(SCENARIOS.keys())}"
        )

    b = 2.2676
    r = 101.96
    m = 26.9
    q = 2.222
    dt = 0.1

    U_values = [[u, 0.0] for u in np.linspace(-0.5, 0.5, 50)]
    params = dict(b=b, r=r, m=m, q=q, dt=dt, U_values=U_values)
    max_iter = 20

    # build state grid used by all kernels and masks
    L_vals = np.linspace(K[0][0], K[0][1], GRID_POINTS)
    P_vals = np.linspace(K[1][0], K[1][1], GRID_POINTS)

    viability_bar = ProgressBar("VIABILITY", 3)
    capture_bar = ProgressBar("CAPTURE", 2 * max_iter + 2, start_immediately=False)

    # compute base viability and reversible masks first
    viab_D1_mask, viab_D2_mask, R_D1_D2_mask, R_D2_D1_mask, E_mask, successors_global = compute_E_sets(
        L_vals,
        P_vals,
        params,
        max_iter=max_iter,
        n_cores=N_CORES,
        viability_bar=viability_bar,
        capture_bar=capture_bar,
    )
    capture_bar.set_total_steps(capture_bar.current + 2)

    # compute capture masks of each viability set
    capt_viab_D1 = compute_capture_mask(
        successors_global,
        target_mask=viab_D1_mask,
        exclude_mask=None,
        exclude_target=True,
        model_params=params,
        n_cores=N_CORES,
        progress_bar=capture_bar,
    )

    capt_viab_D2 = compute_capture_mask(
        successors_global,
        target_mask=viab_D2_mask,
        exclude_mask=None,
        exclude_target=True,
        model_params=params,
        n_cores=N_CORES,
        progress_bar=capture_bar,
    )
    capture_bar.done()

    # option sets are viability ∩ capture in opposite domain
    O_D1_D2_mask = np.logical_and(viab_D1_mask, capt_viab_D2).astype(np.uint8)
    O_D2_D1_mask = np.logical_and(viab_D2_mask, capt_viab_D1).astype(np.uint8)

    masks = {
        "viab_D1": viab_D1_mask,
        "capt_viab_D1": capt_viab_D1,
        "viab_D2": viab_D2_mask,
        "capt_viab_D2": capt_viab_D2,
        "O_D1_D2": O_D1_D2_mask,
        "O_D2_D1": O_D2_D1_mask,
        "R_D1_D2": R_D1_D2_mask,
        "R_D2_D1": R_D2_D1_mask,
        "E": E_mask,
    }

    get_trajectory = None
    if plot_traj:
        # enable lazy trajectory generation only when plotting is requested
        get_trajectory = build_trajectory_getter(L_vals, P_vals, params, successors_global, masks)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for scenario_key in plots:
        scenario = SCENARIOS[scenario_key]
        scenario_trajs = None
        if plot_traj and get_trajectory is not None:
            # compute only the trajectories required by this scenario
            scenario_trajs = {name: get_trajectory(name) for name in scenario.trajectory_keys}

        output_path = output_root / f"{scenario.key}.png"
        plot_scenario(
            scenario,
            L_vals=L_vals,
            P_vals=P_vals,
            masks=masks,
            trajs=scenario_trajs,
            output_path=output_path,
            show=show,
        )
        print(f"[PLOT] saved: {output_path}")


if __name__ == "__main__":
    main()
