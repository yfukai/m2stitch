import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_liklihood(prob_uniform, mu, sigma, t):
    t2 = -((t - mu) ** 2) / (2 * sigma ** 2)
    norm_liklihood = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(t2)
    uniform_liklihood = 1 / 100.0
    p = prob_uniform / 100.0
    return p * uniform_liklihood + (1 - p) * norm_liklihood


def compute_inv_liklihood(params, T):
    prob_uniform, mu, sigma = params
    return -np.sum(
        [np.log(np.abs(calc_liklihood(prob_uniform, mu, sigma, t))) for t in T]
    )


def compute_image_overlap(
    grid, direction, W, H, max_stall_count=100, prob_uniform_threshold=90
):
    if direction == "north":
        T = grid["north_y_first"].values / H * 100
    elif direction == "west":
        T = grid["west_x_first"].values / W * 100
    else:
        raise ValueError("direction must be either of north or west")
    T = T[np.isfinite(T)]
    #    print(T)
    #    assert np.all(0 <= T < 100)

    best_model = None
    for i in range(max_stall_count):
        init_guess = np.random.uniform(0, 100, size=(3,))
        model = minimize(
            compute_inv_liklihood,
            init_guess,
            args=(T,),
            bounds=[(0, 100)] * 3,
            method="trust-constr",
        )
        if model["x"][0] < prob_uniform_threshold:
            if best_model is None:
                best_model = model
            else:
                if np.isclose(best_model["fun"], model["fun"]):
                    return model["x"]
                elif model["fun"] < best_model["fun"]:
                    best_model = model
    return best_model["x"]


def filter_by_overlap_and_correlation(T, ncc, overlap, size, pou=3):
    r = (size * (100 - overlap - pou) / 100, size * (100 - overlap + pou) / 100)
    return (T.between(*r)) & (ncc > 0.5)


def filter_outliers(T, isvalid, w=1.5):
    q1, _, q3 = np.quantile(T[isvalid], (0.25, 0.5, 0.75))
    iqd = max(1, np.abs(q3 - q1))
    return isvalid & T.between(q1 - w * iqd, q3 + w * iqd)


def compute_repeatability(grid, overlap_n, overlap_w, W, H, pou):
    grid["north_valid1"] = filter_by_overlap_and_correlation(
        grid["north_y_first"], grid["north_ncc_first"], overlap_n, H, pou
    )
    grid["west_valid1"] = filter_by_overlap_and_correlation(
        grid["west_x_first"], grid["west_ncc_first"], overlap_w, W, pou
    )
    grid["north_valid2"] = filter_outliers(grid["north_y_first"], grid["north_valid1"])
    grid["west_valid2"] = filter_outliers(grid["west_x_first"], grid["west_valid1"])

    xs = grid[grid["north_valid2"]]["north_x_first"]
    rx_north = np.ceil((xs.max() - xs.min()) / 2)
    _, yss = zip(*grid[grid["north_valid2"]].groupby("col")["north_y_first"])
    ry_north = np.ceil(np.max([np.max(ys) - np.min(ys) for ys in yss]) / 2)
    r_north = max(rx_north, ry_north)

    ys = grid[grid["west_valid2"]]["west_y_first"]
    ry_west = np.ceil((ys.max() - ys.min()) / 2)
    _, xss = zip(*grid[grid["west_valid2"]].groupby("row")["west_x_first"])
    rx_west = np.ceil(np.max([np.max(xs) - np.min(xs) for xs in xss]) / 2)
    r_west = max(ry_west, rx_west)

    return grid, max(r_north, r_west)


def filter_by_repeatability(grid, r):
    for _, grp in grid.groupby("row"):
        isvalid = grp["north_valid2"].astype(bool)
        if not any(isvalid):
            grid.loc[grp.index, "north_valid3"] = False
        else:
            medx = grp[isvalid]["north_x_first"].median()
            medy = grp[isvalid]["north_y_first"].median()
            grid.loc[grp.index, "north_valid3"] = (
                grp["north_x_first"].between(medx - r, medx + r)
                & grp["north_y_first"].between(medy - r, medy + r)
                & (grp["north_ncc_first"] > 0.5)
            )
    for _, grp in grid.groupby("col"):
        isvalid = grp["west_valid2"]
        if not any(isvalid):
            grid.loc[grp.index, "west_valid3"] = False
        else:
            medx = grp[isvalid]["west_x_first"].median()
            medy = grp[isvalid]["west_y_first"].median()
            grid.loc[grp.index, "west_valid3"] = (
                grp["west_x_first"].between(medx - r, medx + r)
                & grp["west_y_first"].between(medy - r, medy + r)
                & (grp["west_ncc_first"] > 0.5)
            )
    return grid


def replace_invalid_translations(grid):
    for direction in ["north", "west"]:
        for key in ["x", "y", "ncc"]:
            isvalid = grid[f"{direction}_valid3"]
            grid.loc[isvalid, f"{direction}_{key}_second"] = grid.loc[
                isvalid, f"{direction}_{key}_first"
            ]
    for direction, rowcol in zip(["north", "west"], ["row", "col"]):
        for _, grp in grid.groupby(rowcol):
            isvalid = grp[f"{direction}_valid3"].astype(bool)
            if any(isvalid):
                assert all(
                    pd.isna(grid.loc[grp.index[~isvalid], f"{direction}_x_second"])
                )
                assert all(
                    pd.isna(grid.loc[grp.index[~isvalid], f"{direction}_y_second"])
                )
                grid.loc[grp.index[~isvalid], f"{direction}_x_second"] = grp[isvalid][
                    f"{direction}_x_first"
                ].median()
                grid.loc[grp.index[~isvalid], f"{direction}_y_second"] = grp[isvalid][
                    f"{direction}_y_first"
                ].median()
                grid.loc[grp.index[~isvalid], f"{direction}_ncc_second"] = -1
    for direction, xy in itertools.product(["north", "west"], ["x", "y"]):
        key = f"{direction}_{xy}_second"
        isna = pd.isna(grid[key])
        grid.loc[isna, key] = grid.loc[~isna, key].median()
        grid.loc[isna, f"{direction}_ncc_second"] = -1
    return grid
