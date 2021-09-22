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
    prob_uniform_threshold: Float = 80,
) -> Tuple[float, ...]:
    """Compute the value of the image overlap.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{top|left}_{x|y}_first"
    direction : str
        the direction of the overlap, either of "top" or "left"
    W : Int
        the image width
    H : Int
        the image height
    max_stall_count : Int, optional
        the maximum count of optimization, by default 100
    prob_uniform_threshold : Float, optional
        the threshold for the probabillity for uniform error, by default 80

    Returns
    -------
    x : Tuple[float, float, float]
        the computed parameters : prob_uniform, mu, sigma

    Raises
    ------
    ValueError
        when direction is not in ["top","left"], raises ValueError
    """
    if direction == "top":
        T = grid["top_y_first"].values / H * 100
    elif direction == "left":
        T = grid["left_x_first"].values / W * 100
    else:
        raise ValueError("direction must be either of top or left")
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
    assert (
        not best_model is None
    ), "Overlap model estimation failed, try raising the value of overlap_prob_uniform_threshold"  # noqa :
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
    valid_T = T[isvalid].values
    if len(valid_T) < 1:
        return isvalid
    q1, _, q3 = np.quantile(valid_T, (0.25, 0.5, 0.75))
    iqd = max(1, np.abs(q3 - q1))
    return isvalid & T.between(q1 - w * iqd, q3 + w * iqd)


def compute_repeatability(
    grid: pd.DataFrame, overlap_n: Float, overlap_w: Float, W: Int, H: Int, pou: Float
) -> Tuple[pd.DataFrame, Float]:
    """Compute the repeatability of the stage motion.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{top|left}_{x|y|ncc}_first"
    overlap_n : Float
        the estimated overlap for top direction
    overlap_w : Float
        the estimated overlap for left direction
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
    grid["top_valid1"] = filter_by_overlap_and_correlation(
        grid["top_y_first"], grid["top_ncc_first"], overlap_n, H, pou
    )
    grid["left_valid1"] = filter_by_overlap_and_correlation(
        grid["left_x_first"], grid["left_ncc_first"], overlap_w, W, pou
    )
    grid["top_valid2"] = filter_outliers(grid["top_y_first"], grid["top_valid1"])
    grid["left_valid2"] = filter_outliers(grid["left_x_first"], grid["left_valid1"])

    if np.any(grid["top_valid2"]):
        xs = grid[grid["top_valid2"]]["top_x_first"]
        rx_top = np.ceil((xs.max() - xs.min()) / 2)
        _, yss = zip(*grid[grid["top_valid2"]].groupby("col")["top_y_first"])
        ry_top = np.ceil(np.max([np.max(ys) - np.min(ys) for ys in yss]) / 2)
        r_top = max(rx_top, ry_top)
    else:
        r_top = 0  # better than failing

    if np.any(grid["left_valid2"]):
        ys = grid[grid["left_valid2"]]["left_y_first"]
        ry_left = np.ceil((ys.max() - ys.min()) / 2)
        _, xss = zip(*grid[grid["left_valid2"]].groupby("row")["left_x_first"])
        rx_left = np.ceil(np.max([np.max(xs) - np.min(xs) for xs in xss]) / 2)
        r_left = max(ry_left, rx_left)
    else:
        r_left = 0  # better than failing

    return grid, max(r_top, r_left)


def filter_by_repeatability(grid: pd.DataFrame, r: Float) -> pd.DataFrame:
    """Filter the stage translation by repeatability.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{top|left}_{x|y|ncc}_first"
    r : Float
        the repeatability value

    Returns
    -------
    grid : pd.DataFrame
        the updated dataframe for the grid position
    """
    for _, grp in grid.groupby("row"):
        isvalid = grp["top_valid2"].astype(bool)
        if not any(isvalid):
            grid.loc[grp.index, "top_valid3"] = False
        else:
            medx = grp[isvalid]["top_x_first"].median()
            medy = grp[isvalid]["top_y_first"].median()
            grid.loc[grp.index, "top_valid3"] = (
                grp["top_x_first"].between(medx - r, medx + r)
                & grp["top_y_first"].between(medy - r, medy + r)
                & (grp["top_ncc_first"] > 0.5)
            )
    for _, grp in grid.groupby("col"):
        isvalid = grp["left_valid2"]
        if not any(isvalid):
            grid.loc[grp.index, "left_valid3"] = False
        else:
            medx = grp[isvalid]["left_x_first"].median()
            medy = grp[isvalid]["left_y_first"].median()
            grid.loc[grp.index, "left_valid3"] = (
                grp["left_x_first"].between(medx - r, medx + r)
                & grp["left_y_first"].between(medy - r, medy + r)
                & (grp["left_ncc_first"] > 0.5)
            )
    return grid


def replace_invalid_translations(grid: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid translations by estimated values.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position,
        with columns "{top|left}_{x|y}_second" and "{top|left}_valid3"

    Returns
    -------
    grid : pd.DataFrame
        the updatd dataframe for the grid position
    """
    for direction in ["top", "left"]:
        for key in ["x", "y", "ncc"]:
            isvalid = grid[f"{direction}_valid3"]
            grid.loc[isvalid, f"{direction}_{key}_second"] = grid.loc[
                isvalid, f"{direction}_{key}_first"
            ]
    for direction, rowcol in zip(["top", "left"], ["row", "col"]):
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
    for direction, xy in itertools.product(["top", "left"], ["x", "y"]):
        key = f"{direction}_{xy}_second"
        isna = pd.isna(grid[key])
        grid.loc[isna, key] = grid.loc[~isna, key].median()
        grid.loc[isna, f"{direction}_ncc_second"] = -1
    for direction, xy in itertools.product(["top", "left"], ["x", "y"]):
        assert np.all(np.isfinite(grid[f"{direction}_{xy}_second"]))

    return grid
