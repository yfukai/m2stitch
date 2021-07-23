import itertools
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._typing_utils import Float
from ._typing_utils import FloatArray
from ._typing_utils import Int


def calc_liklihood(prob_uniform: Float, mu: Float, sigma: Float, t: Float) -> Float:
    """Calculate the liklihood of the translation.

    Parameters
    ----------
    prob_uniform : Float
        the probability of the uniform error
    mu : Float
        the mean of the Gaussian distribution
    sigma : Float
        the stdev of the Gaussian distribution
    t : Float
        the translation amplitude

    Returns
    -------
    liklihood : Float
        the calculated liklihood
    """
    t2 = -((t - mu) ** 2) / (2 * sigma ** 2)
    norm_liklihood = 1.0 / (np.sqrt(2 * np.pi) * sigma) * np.exp(t2)
    uniform_liklihood = 1 / 100.0
    p = prob_uniform / 100.0
    return p * uniform_liklihood + (1 - p) * norm_liklihood


def compute_inv_liklihood(
    params: Union[Tuple[Float, Float, Float], FloatArray], T: FloatArray
) -> Float:
    """Compute the negated sum of log liklihood.

    Parameters
    ----------
    params : Union[Tuple[Float, Float, Float], FloatArray]
        the parameters : prob_uniform, mu, sigma
    T : FloatArray
        the value of translations

    Returns
    -------
    inv_log_likelihood : Float
        the negated sum of log liklihood
    """
    prob_uniform, mu, sigma = params
    return -np.sum(
        [np.log(np.abs(calc_liklihood(prob_uniform, mu, sigma, t))) for t in T]
    )


def compute_image_overlap(
    grid: pd.DataFrame,
    direction: str,
    W: Int,
    H: Int,
    max_stall_count: Int = 100,
    prob_uniform_threshold: Float = 90,
) -> Tuple[float, ...]:
    """Compute the value of the image overlap.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{north|west}_{x|y}_first"
    direction : str
        the direction of the overlap, either of "north" or "west"
    W : Int
        the image width
    H : Int
        the image height
    max_stall_count : Int, optional
        the maximum count of optimization, by default 100
    prob_uniform_threshold : Float, optional
        the threshold for the probabillity for uniform error, by default 90

    Returns
    -------
    x : Tuple[float, float, float]
        the computed parameters : prob_uniform, mu, sigma

    Raises
    ------
    ValueError
        when direction is not in ["north","west"], raises ValueError
    """
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
    for _ in range(max_stall_count):
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
                    assert len(model["x"]) == 3
                    return tuple(map(float, model["x"]))
                elif model["fun"] < best_model["fun"]:
                    best_model = model
    assert not best_model is None
    #    best_model_params : Tuple[Float,Float,Float] = tuple(best_model["x"])
    assert len(best_model["x"]) == 3
    return tuple(map(float, best_model["x"]))


def filter_by_overlap_and_correlation(
    T: pd.Series, ncc: pd.Series, overlap: Float, size: Int, pou: Float = 3
) -> pd.Series:
    """Filter the translation values by estimated overlap.

    Parameters
    ----------
    T : pd.Series
        the translation
    ncc : pd.Series
        the normalized cross correlation, this value should be > 0.5 to be valid
    overlap : Float
        the estimated overlap
    size : Int
        the size of image dimension
    pou : Float, optional
        the percentile margin for error, by default 3

    Returns
    -------
    isvalid : pd.Series
        whether the translation is within the estimated limit
    """
    r = (size * (100 - overlap - pou) / 100, size * (100 - overlap + pou) / 100)
    return (T.between(*r)) & (ncc > 0.5)


def filter_outliers(T: pd.Series, isvalid: pd.Series, w: Float = 1.5) -> pd.Series:
    """Filter the translation outside the 25% and 75% percentiles * w.

    Parameters
    ----------
    T : pd.Series
        the translation
    isvalid : pd.Series
        whether the translation is valid
    w : Float, optional
        the coef for the percentiles, by default 1.5

    Returns
    -------
    isvalid : pd.Series
        whether the translation is within the estimated limit
    """
    q1, _, q3 = np.quantile(T[isvalid].values, (0.25, 0.5, 0.75))
    iqd = max(1, np.abs(q3 - q1))
    return isvalid & T.between(q1 - w * iqd, q3 + w * iqd)


def compute_repeatability(
    grid: pd.DataFrame, overlap_n: Float, overlap_w: Float, W: Int, H: Int, pou: Float
) -> Tuple[pd.DataFrame, Float]:
    """Compute the repeatability of the stage motion.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{north|west}_{x|y|ncc}_first"
    overlap_n : Float
        the estimated overlap for north direction
    overlap_w : Float
        the estimated overlap for west direction
    W : Int
        the width of the height images
    H : Int
        the width of the tile images
    pou : Float
        the percentile margin for error, by default 3

    Returns
    -------
    grid : pd.DataFrame
        the updated dataframe for the grid position
    repeatability : Float
        the repeatability of the stage motion
    """
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


def filter_by_repeatability(grid: pd.DataFrame, r: Float) -> pd.DataFrame:
    """Filter the stage translation by repeatability.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{north|west}_{x|y|ncc}_first"
    r : Float
        the repeatability value

    Returns
    -------
    grid : pd.DataFrame
        the updated dataframe for the grid position
    """
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


def replace_invalid_translations(grid: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid translations by estimated values.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, 
        with columns "{north|west}_{x|y}_second" and "{north|west}_valid3"

    Returns
    -------
    grid : pd.DataFrame
        the updatd dataframe for the grid position
    """
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
