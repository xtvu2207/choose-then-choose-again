"""common utility helpers

author: Xuan Tung VU
"""

import sys
import numpy as np


class ProgressBar:
    """simple terminal progress bar with in-place updates"""

    def __init__(self, label: str, total_steps: int, start_immediately: bool = True):
        """initialize progress bar state

        args:
            label: short text shown at the left of the bar
            total_steps: total number of steps to reach 100%
            start_immediately: if true, render bar right away
        """
        self.label = label
        self.total_steps = max(1, int(total_steps))
        self.current = 0
        if start_immediately:
            self._render()

    def _render(self):
        """render the current bar state to stdout"""
        width = 40
        ratio = min(1.0, self.current / self.total_steps)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        pct = int(ratio * 100)
        sys.stdout.write(f"\r[{self.label:<9}] [{bar}] {pct:3d}%")
        sys.stdout.flush()

    def advance(self, n: int = 1):
        """advance the progress by `n` steps and re-render"""
        self.current = min(self.total_steps, self.current + int(n))
        self._render()

    def set_total_steps(self, total_steps: int):
        """update the total number of steps while preserving current progress"""
        self.total_steps = max(1, int(total_steps))
        self.current = min(self.current, self.total_steps)
        self._render()

    def done(self):
        """mark progress as complete and print trailing newline"""
        self.current = self.total_steps
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()


def flatten_mask(mask: np.ndarray) -> np.ndarray:
    """return mask as contiguous flattened uint8 array for c++ bindings"""
    return np.ascontiguousarray(mask.astype(np.uint8).reshape(-1))


def rect_mask(L_vals, P_vals, bounds):
    """create a rectangular mask on the tensor product grid (l_vals x p_vals)"""
    lo_l, hi_l = bounds[0]
    lo_p, hi_p = bounds[1]
    mask_l = (L_vals >= lo_l) & (L_vals <= hi_l)
    mask_p = (P_vals >= lo_p) & (P_vals <= hi_p)
    return np.logical_and.outer(mask_l, mask_p).astype(np.uint8)
