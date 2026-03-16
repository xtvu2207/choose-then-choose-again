"""parameter sweep for predator-prey set ratios
author: Xuan Tung VU
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from viability import CaptureBasinCPU, PredPreyModel, ViabilityKernelCPU
from helpers.utils import flatten_mask

# state-space bounds and domain rectangles
K = [(0.0, 50.0), (0.0, 28.0)]
D1 = [(0.0, 50.0), (0.0, 15.0)]
D2 = [(0.0, 50.0), (15.0, 28.0)]

# numerical settings
GRID_POINTS = 3000
T_VIAB = 10000
T_CAP = 40
N_CORES = 126


path_save = "./prey_predateur/u.png"

def rect_mask(L_vals, P_vals, bounds):
    """build a rectangular binary mask on the (l, p) grid"""
    lo_l, hi_l = bounds[0]
    lo_p, hi_p = bounds[1]
    mask_l = (L_vals >= lo_l) & (L_vals <= hi_l)
    mask_p = (P_vals >= lo_p) & (P_vals <= hi_p)
    return np.logical_and.outer(mask_l, mask_p).astype(np.uint8)


def compute_viability_mask(kernel, successors, base_mask, *, n_cores):
    """compute viability mask from an initial domain mask"""
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
    return np.array(capture.is_capture, copy=True).reshape(GRID_POINTS, GRID_POINTS)


def compute_R_sets(L_vals, P_vals, params, *, max_iter=20):
    """compute viability and reversible sets by fixed-point iteration"""
    kernel_global = ViabilityKernelCPU(
        K_bounds=K,
        U_values=params["U_values"],
        grid_points=GRID_POINTS,
        T=T_VIAB,
        verbose=False,
    )
    kernel_global.set_dynamics(
        PredPreyModel(
            params["b1"],
            params["k1"],
            params["k2"],
            params["d2"],
            params["dt"],
        )
    )

    successors = kernel_global.compute_successors(
        parallel=True, n_cores=N_CORES
    )

    # build rectangular masks for d1 and d2
    mask_D1 = rect_mask(L_vals, P_vals, D1)
    mask_D2 = rect_mask(L_vals, P_vals, D2)

    viab_D1 = compute_viability_mask(
        kernel_global, successors, mask_D1,
        n_cores=N_CORES
    )

    viab_D2 = compute_viability_mask(
        kernel_global, successors, mask_D2,
        n_cores=N_CORES
    )

    # initialize reversible sets from the viability sets
    R_D1_D2 = viab_D1.copy()
    R_D2_D1 = viab_D2.copy()

    # update r-sets until convergence or max_iter
    for _ in range(max_iter):
        capt_from_R_D1_D2 = compute_capture_mask(
            successors,
            target_mask=R_D1_D2,
            exclude_mask=None,
            exclude_target=False,
            model_params=params,
            n_cores=N_CORES,
        )
        R_D2_D1_new = np.logical_and(viab_D2, capt_from_R_D1_D2).astype(np.uint8)

        capt_from_R_D2_D1 = compute_capture_mask(
            successors,
            target_mask=R_D2_D1,
            exclude_mask=None,
            exclude_target=False,
            model_params=params,
            n_cores=N_CORES,
        )
        R_D1_D2_new = np.logical_and(viab_D1, capt_from_R_D2_D1).astype(np.uint8)

        if np.array_equal(R_D2_D1_new, R_D2_D1) and np.array_equal(R_D1_D2_new, R_D1_D2):
            R_D2_D1, R_D1_D2 = R_D2_D1_new, R_D1_D2_new
            break

        R_D2_D1, R_D1_D2 = R_D2_D1_new, R_D1_D2_new

    return viab_D1, viab_D2, R_D1_D2, R_D2_D1, successors

 


def experiment():
    """run a control-bound sweep and plot set-size ratios"""
    b1 = 0.65
    k1 = 0.06
    k2 = 0.08
    d2 = 0.32
    dt = 0.1

    L_vals = np.linspace(K[0][0], K[0][1], GRID_POINTS)
    P_vals = np.linspace(K[1][0], K[1][1], GRID_POINTS)

    upper_values = np.linspace(0.5, 1.5, 20)

    ratios_D2_O_D2_D1 = []
    ratios_D2_R_D2_D1 = []

    ratios_D1_O_D1_D2 = []
    ratios_D1_R_D1_D2 = []

    # sweep the upper control bound and store ratio metrics
    for up in upper_values:
        print(f"\nComputing for upper control bound: {up:.2f}")
        u_grid = np.linspace(-0.5, up, 50)
        U_values = [[uX, uY] for uX in u_grid for uY in u_grid]
        params = dict(
            b1=b1,
            k1=k1,
            k2=k2,
            d2=d2,
            dt=dt,
            U_values=U_values,
        )

        viab_D1, viab_D2, R_D1_D2, R_D2_D1, successors = compute_R_sets(
            L_vals, P_vals, params,
            max_iter=20
        )

        capt_viab_D1 = compute_capture_mask(
            successors,
            target_mask=viab_D1,
            exclude_mask=None,
            exclude_target=True,
            model_params=params,
            n_cores=N_CORES
        )

        capt_viab_D2 = compute_capture_mask(
            successors,
            target_mask=viab_D2,
            exclude_mask=None,
            exclude_target=True,
            model_params=params,
            n_cores=N_CORES
        )

        # combine viability and capture masks to form option sets
        O_D1_D2 = np.logical_and(viab_D1, capt_viab_D2).astype(np.uint8)
        O_D2_D1 = np.logical_and(viab_D2, capt_viab_D1).astype(np.uint8)

        size_viab_D2 = np.sum(viab_D2)
        size_viab_D1 = np.sum(viab_D1)

        size_R_D2_D1 = np.sum(R_D2_D1)
        size_R_D1_D2 = np.sum(R_D1_D2)
        size_O_D2_D1 = np.sum(O_D2_D1)
        size_O_D1_D2 = np.sum(O_D1_D2)

        ratios_D2_O_D2_D1.append(size_O_D2_D1 / size_viab_D2 if size_viab_D2 > 0 else 0)
        ratios_D2_R_D2_D1.append(size_R_D2_D1 / size_viab_D2 if size_viab_D2 > 0 else 0)

        ratios_D1_O_D1_D2.append(size_O_D1_D2 / size_viab_D1 if size_viab_D1 > 0 else 0)
        ratios_D1_R_D1_D2.append(size_R_D1_D2 / size_viab_D1 if size_viab_D1 > 0 else 0)

    # plot all ratios against the sampled control bounds
    plt.figure(figsize=(7, 6))
    plt.plot(
        upper_values,
        ratios_D2_O_D2_D1,
        marker='o',
        color="darkorange",
        label=r"$\left|\mathcal{O}(D_2,D_1)\right| \, / \, \left|\operatorname{Viab}(D_2)\right|$"
    )

    plt.plot(
        upper_values,
        ratios_D2_R_D2_D1,
        marker='o',
        color="purple",
        label=r"$\left|\mathcal{R}(D_2,D_1)\right| \, / \, \left|\operatorname{Viab}(D_2)\right|$"
    )

    plt.plot(
        upper_values,
        ratios_D1_O_D1_D2,
        marker='o',
        color="cyan",
        label=r"$\left|\mathcal{O}(D_1,D_2)\right| \, / \, \left|\operatorname{Viab}(D_1)\right|$"
    )

    plt.plot(
        upper_values,
        ratios_D1_R_D1_D2,
        marker='o',
        color="green",
        label=r"$\left|\mathcal{R}(D_1,D_2)\right| \, / \, \left|\operatorname{Viab}(D_1)\right|$"
    )

    plt.xlabel(r'$\mathbf{Control\ upper\ bound\ } \mathbf{u_{\max}}$', fontsize=13)
    plt.ylabel("Ratio (%)",fontweight="bold",fontsize=13)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path_save, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    experiment()
