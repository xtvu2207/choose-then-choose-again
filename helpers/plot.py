"""plotting utilities for viability and trajectory visualizations

author: Xuan Tung VU
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class ViabilityPlotter:
    """plot viability/capture/option/reversible masks and optional trajectories"""

    LAYER_RANK = {
        "viab": 0,
        "capt": 1,
        "o": 2,
        "r": 3,
        "other": 4,
    }

    def __init__(
        self,
        L_vals_list,
        P_vals_list,
        V_list,
        color_list=None,
        label_list=None,
        show_x_labels=None,
        x_constraint_values=None,
        x_lines=None,
        y_lines=None,
        traj_list=None,
        alpha_list=None,
        k=2,
        arrow_scale=0.1,
        x_label=None,
        y_label=None,
        domain_label_positions=None,
        intersection_masks_list=None,
        intersection_label="Intersection",
        arrow_linewidth=2.0,
        arrow_mutation_scale=None,
        layer_types=None,
        legend_loc="upper right",
        legend_groups=None,
        legend_group_locs=None,
        legend_group_anchors=None,
        legend_group_anchor_transforms=None,
        legend_framealpha=0.95,
        legend_fontsize=17,
        show_base_x_line=True,
        x_min_plot=None,
        y_min_plot=None,
        hide_handoff_start_markers=False,
        handoff_start_marker=None,
        handoff_start_marker_size=110,
        handoff_start_marker_edgecolor="black",
        n_skip = 3,
        equilibrium_prey_predateur=False,
        equilibrium_params=None,
        equilibrium_marker_size=80,
        equilibrium_color="black",
        equilibrium_marker="o",
        equilibrium_label=None,
        phase_vectors=False,
        phase_model=None,
        phase_params=None,
        phase_control=(0.0, 0.0),
        phase_grid_size=13,
        phase_color="black",
        phase_linestyle="--",
        phase_linewidth=0.7,
        phase_segment_scale=0.55,
        phase_head_fraction=0.28,
    ):
        """store plot configuration

        inputs are lists so multiple masks can be overlaid on the same figure
        """
        self.L_vals_list = L_vals_list
        self.P_vals_list = P_vals_list
        self.V_list = V_list
        self.x_lines = x_lines
        self.y_lines = y_lines
        self.color_list = color_list if color_list is not None else ["green", "blue", "orange", "purple"]
        self.label_list = label_list if label_list is not None else [f"V{i+1}" for i in range(len(V_list))]
        self.show_x_labels = show_x_labels if show_x_labels is not None else [True if i < 2 else False for i in range(len(V_list))]
        self.x_constraint_values = x_constraint_values
        self.traj_list = traj_list
        self.k = k
        self.arrow_scale = float(arrow_scale)
        self.arrow_linewidth = float(arrow_linewidth)
        self.arrow_mutation_scale = arrow_mutation_scale
        self.layer_types = layer_types
        self.legend_loc = legend_loc
        self.legend_groups = legend_groups
        self.legend_group_locs = legend_group_locs
        self.legend_group_anchors = legend_group_anchors
        self.legend_group_anchor_transforms = legend_group_anchor_transforms
        self.legend_fontsize = float(legend_fontsize)
        if legend_framealpha is None:
            self.legend_framealpha = None
        else:
            self.legend_framealpha = min(1.0, max(0.0, float(legend_framealpha)))
        self.show_base_x_line = bool(show_base_x_line)
        self.x_min_plot = x_min_plot
        self.y_min_plot = y_min_plot
        self.hide_handoff_start_markers = bool(hide_handoff_start_markers)
        self.handoff_start_marker = handoff_start_marker
        self.handoff_start_marker_size = float(handoff_start_marker_size)
        self.handoff_start_marker_color = "gold"
        self.handoff_start_marker_edgecolor = handoff_start_marker_edgecolor
        self.end_marker_color = "gold"
        self.equilibrium_prey_predateur = bool(equilibrium_prey_predateur)
        self.equilibrium_params = equilibrium_params
        self.equilibrium_marker_size = float(equilibrium_marker_size)
        self.equilibrium_color = equilibrium_color
        self.equilibrium_marker = equilibrium_marker
        self.equilibrium_label = equilibrium_label
        self.phase_vectors = bool(phase_vectors)
        self.phase_model = phase_model
        self.phase_params = phase_params
        self.phase_control = phase_control
        self.phase_grid_size = max(3, int(phase_grid_size))
        self.phase_color = phase_color
        self.phase_linestyle = phase_linestyle
        self.phase_linewidth = float(phase_linewidth)
        self.phase_segment_scale = float(phase_segment_scale)
        self.phase_head_fraction = float(phase_head_fraction)
        self.x_label = "L" if x_label is None else x_label
        self.y_label = "P" if y_label is None else y_label
        self.domain_label_positions = (
            [(6.5, 7.5), (6.5, 20.0)]
            if domain_label_positions is None
            else domain_label_positions
        )
        self.intersection_masks_list = intersection_masks_list
        self.intersection_label = intersection_label

        if alpha_list is None:
            self.alpha_list = [0.35] * len(V_list)
        else:
            self.alpha_list = alpha_list
        self.n_skip = n_skip

    def _prey_predator_equilibrium(self):
        """return prey-predator equilibrium point (x, y) if configured"""
        if not self.equilibrium_prey_predateur:
            return None
        if not isinstance(self.equilibrium_params, dict):
            return None

        try:
            b1 = float(self.equilibrium_params["b1"])
            k1 = float(self.equilibrium_params["k1"])
            k2 = float(self.equilibrium_params["k2"])
            d2 = float(self.equilibrium_params["d2"])
        except (KeyError, TypeError, ValueError):
            return None

        if k1 == 0.0 or k2 == 0.0 or d2 == 0.0:
            return None

        try:
            u1 = float(self.phase_control[0])
            u2 = float(self.phase_control[1])
        except (TypeError, ValueError, IndexError):
            u1 = 0.0
            u2 = 0.0

        # Solve:
        #   b1*x - k1*x*y + u1 = 0
        #   k2*x*y - d2*y + u2 = 0
        # by eliminating x, yielding A*y^2 + B*y + C = 0.
        A = k1 * d2
        B = -(k1 * u2 + b1 * d2 + u1 * k2)
        C = b1 * u2

        disc = B * B - 4.0 * A * C
        if disc < 0.0:
            return None

        sqrt_disc = np.sqrt(disc)
        roots_y = [(-B - sqrt_disc) / (2.0 * A), (-B + sqrt_disc) / (2.0 * A)]
        candidates = []
        for y in roots_y:
            if y <= 0.0:
                continue
            x = (d2 * y - u2) / (k2 * y)
            if x > 0.0 and np.isfinite(x) and np.isfinite(y):
                candidates.append((x, y))

        if not candidates:
            return None

        # Prefer the branch closest to the uncontrolled interior equilibrium.
        x0 = d2 / k2
        y0 = b1 / k1
        return min(candidates, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)

    def _normalize_layer_type(self, value):
        """normalize user aliases to one of: viab, capt, o, r, other"""
        if value is None:
            return None
        t = str(value).strip().lower()
        aliases = {
            "viab": "viab",
            "viability": "viab",
            "capture": "capt",
            "capt": "capt",
            "o": "o",
            "option": "o",
            "r": "r",
            "roption": "r",
            "roptions": "r",
            "other": "other",
        }
        return aliases.get(t, t)

    def _infer_layer_type(self, idx):
        """infer layer type for ordering if explicit types are not provided"""
        if self.layer_types is not None and idx < len(self.layer_types):
            normalized = self._normalize_layer_type(self.layer_types[idx])
            if normalized is not None:
                return normalized

        lab = None
        if self.label_list is not None and idx < len(self.label_list):
            lab = self.label_list[idx]
        if lab in [None, ""]:
            return "other"

        s = str(lab).lower()
        if "viab" in s:
            return "viab"
        if "capt" in s:
            return "capt"
        if r"\mathcal{o" in s:
            return "o"
        if r"\mathcal{r" in s:
            return "r"
        return "other"

    def _phase_rhs_predator_prey(self, X, Y):
        """predator-prey vector field used for phase arrows"""
        if not isinstance(self.phase_params, dict):
            return None, None

        try:
            b1 = float(self.phase_params["b1"])
            k1 = float(self.phase_params["k1"])
            k2 = float(self.phase_params["k2"])
            d2 = float(self.phase_params["d2"])
        except (KeyError, TypeError, ValueError):
            return None, None

        try:
            u1 = float(self.phase_control[0])
            u2 = float(self.phase_control[1])
        except (TypeError, ValueError, IndexError):
            u1 = 0.0
            u2 = 0.0

        dX = b1 * X - k1 * X * Y + u1
        dY = k2 * X * Y - d2 * Y + u2
        return dX, dY

    def _plot_phase_vectors(self, Xmin, Xmax, Ymin, Ymax):
        """overlay dashed phase vectors for configured dynamics model"""
        if not self.phase_vectors:
            return

        model = str(self.phase_model).strip().lower() if self.phase_model is not None else ""
        if model not in {"predator_prey", "prey_predator", "predprey"}:
            return

        x0 = max(0.0, float(Xmin))
        y0 = max(0.0, float(Ymin))
        x1 = float(Xmax)
        y1 = float(Ymax)
        if x1 <= x0 or y1 <= y0:
            return

        n = self.phase_grid_size
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        Xg, Yg = np.meshgrid(xs, ys, indexing="xy")

        dX, dY = self._phase_rhs_predator_prey(Xg, Yg)
        if dX is None or dY is None:
            return

        speed = np.hypot(dX, dY)
        nonzero = speed > 1e-12
        if not np.any(nonzero):
            return

        U = np.zeros_like(dX)
        V = np.zeros_like(dY)
        U[nonzero] = dX[nonzero] / speed[nonzero]
        V[nonzero] = dY[nonzero] / speed[nonzero]

        dx_cell = (x1 - x0) / max(1, n - 1)
        dy_cell = (y1 - y0) / max(1, n - 1)
        seg_len = min(dx_cell, dy_cell) * max(0.05, self.phase_segment_scale)
        if seg_len <= 0.0:
            return

        half_seg = 0.5 * seg_len
        head_len = seg_len * max(0.05, self.phase_head_fraction)

        for i in range(n):
            for j in range(n):
                if not nonzero[i, j]:
                    continue

                ux = float(U[i, j])
                uy = float(V[i, j])
                x = float(Xg[i, j])
                y = float(Yg[i, j])

                x_start = x - half_seg * ux
                y_start = y - half_seg * uy
                x_end = x + half_seg * ux
                y_end = y + half_seg * uy

                plt.plot(
                    [x_start, x_end],
                    [y_start, y_end],
                    color=self.phase_color,
                    linestyle=self.phase_linestyle,
                    linewidth=self.phase_linewidth,
                    zorder=7,
                )

                plt.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(x_end - head_len * ux, y_end - head_len * uy),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=self.phase_color,
                        lw=self.phase_linewidth,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=8,
                )

    def _layer_rank(self, idx):
        """return numeric z-order rank for one plotted layer"""
        t = self._infer_layer_type(idx)
        return self.LAYER_RANK.get(t, self.LAYER_RANK["other"])

    def _plot_order(self, n):
        """compute deterministic overlay order from semantic layer ranks"""
        return sorted(range(n), key=lambda i: (self._layer_rank(i), i))

    def plot(self, title="", save_path="s.png", show=True):
        """render and save one figure

        - filled contours represent binary masks
        - optional trajectory overlays show start/end markers and arrows
        """
        plt.figure(figsize=(7, 6))
        n = len(self.V_list)
        order = self._plot_order(n)
        inter_handle = None
        is_savanna_style = (self.x_label == "Trees" and self.y_label == "Grass")

        for i_plot in order:
            L_vals = self.L_vals_list[i_plot]
            P_vals = self.P_vals_list[i_plot]
            V = self.V_list[i_plot]
            z_layer = 1 + self._layer_rank(i_plot)

            Lg, Pg = np.meshgrid(L_vals, P_vals, indexing="ij")
            mask = np.ma.masked_where(V == 0, V)
            
            # render each active set as a filled binary contour
            plt.contourf(
                Lg, Pg, mask,
                levels=[0.5, 1.1],
                colors=[self.color_list[i_plot % len(self.color_list)]],
                alpha=self.alpha_list[i_plot],
                zorder=z_layer,
            )

            """
            plt.contour(
                lg, pg, v,
                levels=[0.6],
                colors=[self.color_list[i_plot % len(self.color_list)]],
                linewidths=0.8,
                linestyles="--"
            )
            """
            lab = None
            if self.label_list is not None and i_plot < len(self.label_list):
                lab = self.label_list[i_plot]
            if lab not in [None, ""]:
                plt.plot([], [], color=self.color_list[i_plot % len(self.color_list)],
                        linewidth=2, linestyle="--", label=lab)

        if self.intersection_masks_list:
            # optionally highlight an intersection region across masks
            L0 = self.L_vals_list[0]
            P0 = self.P_vals_list[0]
            LL0, PP0 = np.meshgrid(L0, P0, indexing="ij")
            inter = np.ones_like(self.intersection_masks_list[0], dtype=bool)
            for mask in self.intersection_masks_list:
                inter &= np.asarray(mask, dtype=bool)
            if np.any(inter):
                inter_mask = np.ma.masked_where(~inter, inter.astype(float))
                plt.contourf(
                    LL0,
                    PP0,
                    inter_mask,
                    levels=[0.5, 1.1],
                    colors=["red"],
                    alpha=0.8,
                    zorder=5,
                )
                inter_handle = Patch(facecolor="red", edgecolor="red", linewidth=0.5, alpha=0.8)

        start_handoff_flags = []
        end_handoff_flags = []
        detect_handoff = (
            self.hide_handoff_start_markers
            or self.handoff_start_marker is not None
        )
        if self.traj_list is not None and detect_handoff:
            key_scale = 10**9
            start_keys = []
            end_keys = []
            for entry in self.traj_list:
                traj = entry[0] if len(entry) > 0 else None
                if not traj:
                    start_keys.append(None)
                    end_keys.append(None)
                    continue
                x0, y0 = traj[0]
                x1, y1 = traj[-1]
                start_keys.append((int(round(x0 * key_scale)), int(round(y0 * key_scale))))
                end_keys.append((int(round(x1 * key_scale)), int(round(y1 * key_scale))))

            n_traj = len(self.traj_list)
            start_handoff_flags = [False] * n_traj
            end_handoff_flags = [False] * n_traj
            for i in range(n_traj):
                sk = start_keys[i]
                ek = end_keys[i]
                if sk is None:
                    continue
                for j in range(n_traj):
                    if i == j:
                        continue
                    if end_keys[j] == sk:
                        start_handoff_flags[i] = True
                    if start_keys[j] == ek:
                        end_handoff_flags[i] = True
        elif self.traj_list is not None:
            start_handoff_flags = [False] * len(self.traj_list)
            end_handoff_flags = [False] * len(self.traj_list)

        draw_any_start_marker = False
        draw_any_end_marker = False
        draw_any_handoff_marker = False
        if self.traj_list is not None:
            for traj_idx, traj_entry in enumerate(self.traj_list):
                if len(traj_entry) >= 3:
                    traj = traj_entry[0]
                    switches = traj_entry[2]
                elif len(traj_entry) == 2:
                    traj = traj_entry[0]
                    second = traj_entry[1]
                    switches = second if isinstance(second, (list, tuple, np.ndarray)) else None
                elif len(traj_entry) == 1:
                    traj = traj_entry[0]
                    switches = None
                else:
                    continue
                if not traj:
                    continue
                Ls = [p[0] for p in traj]
                Ps = [p[1] for p in traj]
                col = "black"

                # start/end markers plus main polyline
                is_handoff_start = (
                    bool(start_handoff_flags)
                    and start_handoff_flags[traj_idx]
                )
                is_handoff_end = (
                    bool(end_handoff_flags)
                    and end_handoff_flags[traj_idx]
                )

                draw_start_marker = True
                if is_handoff_start and self.handoff_start_marker is not None:
                    plt.scatter(
                        Ls[0],
                        Ps[0],
                        marker=self.handoff_start_marker,
                        s=self.handoff_start_marker_size,
                        color=self.handoff_start_marker_color,
                        zorder=12,
                        edgecolors=self.handoff_start_marker_edgecolor,
                    )
                    draw_any_handoff_marker = True
                    draw_start_marker = False
                elif (
                    is_handoff_start
                    and self.hide_handoff_start_markers
                ):
                    draw_start_marker = False

                if draw_start_marker:
                    plt.scatter(Ls[0], Ps[0], marker='*', s=250, color='gold', zorder=12, edgecolors='black')
                    draw_any_start_marker = True
                draw_end_marker = not (is_handoff_end and self.handoff_start_marker is not None)
                if draw_end_marker:
                    plt.scatter(
                        Ls[-1],
                        Ps[-1],
                        marker='D',
                        s=60,
                        color=self.end_marker_color,
                        zorder=12,
                        edgecolors='black',
                    )
                    draw_any_end_marker = True
                plt.plot(Ls, Ps, color=col, linewidth=1, zorder=8)
                if switches is not None:
                    for pos in switches:
                        if 0 <= pos < len(Ls):
                            plt.scatter(Ls[pos], Ps[pos], s=50, color="darkred", zorder=10)

                k = self.k
                n_skip = self.n_skip
                for i in range(n_skip, len(Ls) - 1 - n_skip, k):
                    x0, y0 = Ls[i], Ps[i]
                    x1, y1 = Ls[i + 1], Ps[i + 1]
                    dx = x1 - x0
                    dy = y1 - y0
                    norm = np.hypot(dx, dy)
                    if norm == 0:
                        continue

                    xf = x0 + 0.8 * dx
                    yf = y0 + 0.8 * dy
                    dx_small = dx * self.arrow_scale
                    dy_small = dy * self.arrow_scale

                    arrowprops = dict(
                        arrowstyle='-|>',
                        color=col,
                        lw=self.arrow_linewidth,
                        shrinkA=0,
                        shrinkB=0,
                    )
                    if self.arrow_mutation_scale is not None:
                        arrowprops["mutation_scale"] = self.arrow_mutation_scale

                    # add directional arrows along the trajectory
                    plt.annotate(
                        '',
                        xy=(xf, yf),
                        xytext=(xf - dx_small, yf - dy_small),
                        arrowprops=arrowprops,
                        zorder=9,
                    )

        # auto-range axes from non-empty masks
        X_used = []
        Y_used = []
        for L_vals, P_vals, V in zip(self.L_vals_list, self.P_vals_list, self.V_list):
            idx = np.where(V > 0)
            if len(idx[0]) > 0:
                X_used.append(L_vals[idx[0]].min())
                X_used.append(L_vals[idx[0]].max())
                Y_used.append(P_vals[idx[1]].min())
                Y_used.append(P_vals[idx[1]].max())

        if len(X_used) == 0:
            X_used = [0, 1]
            Y_used = [0, 1]

        Xmin = max(0, min(X_used))
        Xmax = max(X_used)
        Ymin = max(0, min(Y_used))
        Ymax = max(Y_used)

        show_constraints = True
        if self.show_x_labels is not None:
            show_constraints = any(bool(v) for v in self.show_x_labels[:n])

        if show_constraints and self.x_constraint_values is not None and len(self.x_constraint_values) > 0:
            Ymax = max(Ymax, max(self.x_constraint_values) * 1.05)

        if self.x_lines is not None and len(self.x_lines) > 0:
            Xmin = min(Xmin, min(self.x_lines))
            Xmax = max(Xmax, max(self.x_lines))
        if self.y_lines is not None and len(self.y_lines) > 0:
            Ymin = min(Ymin, min(self.y_lines))
            Ymax = max(Ymax, max(self.y_lines))

        Xspan = Xmax - Xmin
        Yspan = Ymax - Ymin
        if Xspan == 0:
            Xspan = 1.0
        if Yspan == 0:
            Yspan = 1.0
        Xmin -= 0.05 * Xspan
        Xmax += 0.05 * Xspan
        Ymin -= 0.05 * Yspan
        Ymax += 0.05 * Yspan

        eq_point = self._prey_predator_equilibrium()
        if eq_point is not None:
            eq_x, eq_y = eq_point
            Xmin = min(Xmin, eq_x)
            Xmax = max(Xmax, eq_x)
            Ymin = min(Ymin, eq_y)
            Ymax = max(Ymax, eq_y)


        if self.show_base_x_line:
            plt.axvline(x=self.L_vals_list[0][0], color="black", linestyle="--", linewidth=0.8)

        if show_constraints and self.x_constraint_values is not None:
            y_constraints = sorted(
                y_c for y_c in self.x_constraint_values
                if Ymin <= y_c <= Ymax
            )
  
            for y_c in y_constraints:
                plt.axhline(y=y_c, color="black", linestyle="--", linewidth=0.8)
        if self.x_lines is not None:
            x_constraints = sorted(x for x in self.x_lines if Xmin <= x <= Xmax)
            for x_c in x_constraints:
                plt.axvline(x=x_c, color="black", linestyle="--", linewidth=0.8)
            if is_savanna_style and len(x_constraints) > 0:
                ax = plt.gca()
                x_map = {x_c: i for i, x_c in enumerate(x_constraints, start=1)}
                xticks = list(ax.get_xticks())
                xticks = [t for t in xticks if not np.isclose(t, 20)]
                for x_c in x_constraints:
                    if not any(np.isclose(t, x_c) for t in xticks):
                        xticks.append(x_c)
                ax.set_xticks(sorted(xticks))

                def x_formatter(x, pos):
                    for x_c, idx in x_map.items():
                        if np.isclose(x, x_c):
                            return rf"$\mathbf{{T_{{{idx}}}}}$"
                    return f"{x:g}"

                ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_formatter))
        if self.y_lines is not None:
            y_constraints = sorted(y for y in self.y_lines if Ymin <= y <= Ymax)
            for y_c in y_constraints:
                plt.axhline(y=y_c, color="black", linestyle="--", linewidth=0.8)
            if is_savanna_style and len(y_constraints) > 0:
                ax = plt.gca()
                y_map = {y_c: i for i, y_c in enumerate(y_constraints, start=1)}
                yticks = list(ax.get_yticks())
                for y_c in y_constraints:
                    if not any(np.isclose(t, y_c) for t in yticks):
                        yticks.append(y_c)
                ax.set_yticks(sorted(yticks))

                def y_formatter(y, pos):
                    for y_c, idx in y_map.items():
                        if np.isclose(y, y_c):
                            return rf"$\mathbf{{G_{{{idx}}}}}$"
                    return f"{y:g}"

                ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_formatter))
        if self.x_min_plot is not None:
            Xmin = float(self.x_min_plot)
        if self.y_min_plot is not None:
            Ymin = float(self.y_min_plot)

        self._plot_phase_vectors(Xmin, Xmax, Ymin, Ymax)

        x_left = self.L_vals_list[0][0] - 1.4 if self.x_min_plot is None else Xmin
        plt.xlim(x_left, Xmax)
        plt.ylim(Ymin, Ymax)

        if eq_point is not None:
            plt.scatter(
                eq_point[0],
                eq_point[1],
                s=self.equilibrium_marker_size,
                color=self.equilibrium_color,
                marker=self.equilibrium_marker,
                zorder=15,
            )

        plt.xlabel(self.x_label, fontsize=16, fontweight="bold")
        plt.ylabel(self.y_label, fontsize=16, fontweight="bold")
        plt.title(title)
        # build legend entries from visible layers and trajectory markers
        def append_layer_legend_entry(handles, labels, layer_idx):
            if layer_idx < 0 or layer_idx >= n:
                return
            lab = None
            if self.label_list is not None and layer_idx < len(self.label_list):
                lab = self.label_list[layer_idx]
            if lab in [None, ""]:
                return

            col = self.color_list[layer_idx % len(self.color_list)]
            a = self.alpha_list[layer_idx] if self.alpha_list is not None else 0.35
            handles.append(Patch(facecolor=col, edgecolor=col, linewidth=0.5, alpha=a))
            labels.append(lab)

        def append_extra_legend_entries(handles, labels):
            if inter_handle is not None:
                handles.append(inter_handle)
                labels.append(self.intersection_label)

            if self.traj_list and draw_any_start_marker:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker='*',
                        linestyle='None',
                        markerfacecolor='gold',
                        markeredgecolor='black',
                        markersize=14,
                    )
                )
                labels.append("Start")

            if self.traj_list and draw_any_handoff_marker:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=self.handoff_start_marker,
                        linestyle='None',
                        markerfacecolor=self.handoff_start_marker_color,
                        markeredgecolor=self.handoff_start_marker_edgecolor,
                        markersize=9,
                    )
                )
                labels.append("Handoff point")

            if self.traj_list and draw_any_end_marker:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker='D',
                        linestyle='None',
                        markerfacecolor=self.end_marker_color,
                        markeredgecolor='black',
                        markersize=8,
                    )
                )
                labels.append("End")

            if eq_point is not None and self.equilibrium_label not in [None, ""]:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=self.equilibrium_marker,
                        linestyle='None',
                        markerfacecolor=self.equilibrium_color,
                        markeredgecolor=self.equilibrium_color,
                        markersize=8,
                    )
                )
                labels.append(self.equilibrium_label)

        def make_legend(handles, labels, loc, zorder, anchor=None, anchor_transform=None):
            if not handles:
                return None
            legend_kwargs = dict(
                handles=handles,
                labels=labels,
                loc=loc,
                fontsize=self.legend_fontsize,
                frameon=True,
                framealpha=self.legend_framealpha,
                borderpad=0.2,
                labelspacing=0.2,
                handletextpad=0.4,
                borderaxespad=0.2,
                handlelength=1.0,
            )
            if anchor is not None:
                legend_kwargs["bbox_to_anchor"] = anchor
                if anchor_transform == "data":
                    legend_kwargs["bbox_transform"] = plt.gca().transData
                elif anchor_transform == "axes":
                    legend_kwargs["bbox_transform"] = plt.gca().transAxes

            legend_obj = plt.legend(**legend_kwargs)
            legend_obj.set_zorder(zorder)
            plt.gca().add_artist(legend_obj)
            return legend_obj

        legend = None
        if self.legend_groups is not None:
            groups = list(self.legend_groups)
            group_locs = list(self.legend_group_locs) if self.legend_group_locs is not None else []
            group_anchors = (
                list(self.legend_group_anchors)
                if self.legend_group_anchors is not None
                else []
            )
            group_anchor_transforms = (
                list(self.legend_group_anchor_transforms)
                if self.legend_group_anchor_transforms is not None
                else []
            )
            for idx_group, group_indices in enumerate(groups):
                handles = []
                labels = []
                for layer_idx in group_indices:
                    append_layer_legend_entry(handles, labels, int(layer_idx))

                # non-layer legend items are attached once to the first group
                if idx_group == 0:
                    append_extra_legend_entries(handles, labels)

                loc = group_locs[idx_group] if idx_group < len(group_locs) else self.legend_loc
                anchor = group_anchors[idx_group] if idx_group < len(group_anchors) else None
                anchor_transform = (
                    group_anchor_transforms[idx_group]
                    if idx_group < len(group_anchor_transforms)
                    else None
                )
                legend_obj = make_legend(
                    handles,
                    labels,
                    loc,
                    100 + idx_group,
                    anchor=anchor,
                    anchor_transform=anchor_transform,
                )
                if legend is None and legend_obj is not None:
                    legend = legend_obj
        else:
            legend_handles = []
            legend_labels = []
            for i in order:
                append_layer_legend_entry(legend_handles, legend_labels, i)

            append_extra_legend_entries(legend_handles, legend_labels)
            legend = make_legend(legend_handles, legend_labels, self.legend_loc, 100)

        if self.domain_label_positions is not None:
            for idx, pos in enumerate(self.domain_label_positions, start=1):
                if pos is None or len(pos) != 2:
                    continue
                plt.text(
                    pos[0],
                    pos[1],
                    rf"$D_{idx}$",
                    fontsize=17,
                    ha="center",
                    va="center",
                    zorder=20,
                )

        if is_savanna_style:
            ax = plt.gca()
            plt.gcf().canvas.draw()

            for ticklab in ax.get_xticklabels():
                if ticklab.get_text() == r"$\mathbf{T_{1}}$":
                    ticklab.set_fontsize(15)

            for ticklab in ax.get_yticklabels():
                if ticklab.get_text() == r"$\mathbf{G_{1}}$":
                    ticklab.set_fontsize(15)



        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
        if show:
            plt.show()
        else:
            plt.close()
