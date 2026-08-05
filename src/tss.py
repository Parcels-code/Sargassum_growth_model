import numpy as np


def make_presence_grid(x, y, x_edges, y_edges, valid_mask):
    """Convert point coordinates to a boolean presence/absence grid."""
    valid = np.isfinite(x) & np.isfinite(y)
    h, _, _ = np.histogram2d(y[valid], x[valid], bins=[y_edges, x_edges])
    grid = h > 0
    grid[~valid_mask] = np.nan
    return grid


def tss(obs, pred):
    """Compute TSS = sensitivity + specificity - 1 from two boolean grids."""
    tp = np.logical_and(obs, pred).sum()
    fn = np.logical_and(obs, ~pred).sum()
    fp = np.logical_and(~obs, pred).sum()
    tn = np.logical_and(~obs, ~pred).sum()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity + specificity - 1


def compute_presence_grids(x_obs, y_obs, x_pred, y_pred, brightness, DX):
    """Compute presence grid for a given set of observed and predicted points."""
    # 1) Define grid edges
    x_edges = np.arange(np.nanmin(x_obs), np.nanmax(x_obs) + DX, DX)
    y_edges = np.arange(np.nanmin(y_obs), np.nanmax(y_obs) + DX, DX)

    # 2) Create a valid mask based on brightness
    valid_points = brightness[:, 2] > 1
    h, _, _ = np.histogram2d(brightness[valid_points, 1], brightness[valid_points, 0], bins=[y_edges, x_edges])
    valid_mask = h > 0

    # 3) Build presence/absence maps
    obs_grid = make_presence_grid(x_obs, y_obs, x_edges, y_edges, valid_mask)
    pred_grid = make_presence_grid(x_pred, y_pred, x_edges, y_edges, valid_mask)

    # 4) Compute agreement grid
    agreement = (pred_grid.astype(int) + 2 * obs_grid.astype(int)).astype(float)
    agreement[~valid_mask] = np.nan

    return obs_grid, pred_grid, agreement, x_edges, y_edges


def compute_tss(x_obs, y_obs, x_pred, y_pred, brightness, DX):
    """Compute TSS (Allouche et al. 2006) for a given set of observed and predicted points."""

    obs_grid, pred_grid, _, _, _ = compute_presence_grids(x_obs, y_obs, x_pred, y_pred, brightness, DX)

    return tss(obs_grid, pred_grid)
