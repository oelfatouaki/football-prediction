"""Turn expected goals into scoreline and win/draw/loss probabilities."""

from typing import Optional, Tuple

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10


def scoreline_grid(lam_h, lam_a, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Probability of every scoreline, shape (matches, max_goals+1, max_goals+1).

    Assumes the two teams score independently. Rows are home goals and columns
    are away goals.
    """
    goals = np.arange(max_goals + 1)
    p_h = poisson.pmf(goals[None, :], np.asarray(lam_h)[:, None])
    p_a = poisson.pmf(goals[None, :], np.asarray(lam_a)[:, None])
    return p_h[:, :, None] * p_a[:, None, :]


def result_probs(lam_h, lam_a, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Home win / draw / away win probabilities, shape (matches, 3)."""
    grid = scoreline_grid(lam_h, lam_a, max_goals)
    total = grid.sum(axis=(1, 2))
    goals = np.arange(max_goals + 1)
    i, j = np.meshgrid(goals, goals, indexing="ij")
    return np.column_stack([
        (grid * (i > j)).sum(axis=(1, 2)) / total,
        (grid * (i == j)).sum(axis=(1, 2)) / total,
        (grid * (i < j)).sum(axis=(1, 2)) / total,
    ])


def most_likely_score(lam_h: float, lam_a: float, result: Optional[str] = None,
                      max_goals: int = MAX_GOALS) -> Tuple[int, int]:
    """Most probable scoreline overall, or within a chosen H/D/A result."""
    grid = scoreline_grid([lam_h], [lam_a], max_goals)[0]
    if result is not None:
        goals = np.arange(max_goals + 1)
        i, j = np.meshgrid(goals, goals, indexing="ij")
        mask = {"H": i > j, "D": i == j, "A": i < j}[result]
        grid = np.where(mask, grid, 0)
    return tuple(int(x) for x in np.unravel_index(grid.argmax(), grid.shape))


def top_scores(lam_h: float, lam_a: float, n: int = 5,
               max_goals: int = MAX_GOALS):
    """The n most likely scorelines, with their probabilities."""
    grid = scoreline_grid([lam_h], [lam_a], max_goals)[0]
    flat = np.argsort(grid.ravel())[::-1][:n]
    return [(int(h), int(a), float(grid[h, a]))
            for h, a in (divmod(k, max_goals + 1) for k in flat)]
