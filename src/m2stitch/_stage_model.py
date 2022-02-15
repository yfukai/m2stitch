import itertools
from typing import Callable
from typing import Tuple

import numpy as np
import pandas as pd

from ._typing_utils import Float
from ._typing_utils import Int


def compute_image_overlap2(
    grid: pd.DataFrame, direction: str, sizeY: Int, sizeX: Int, predictor: Callable
) -> Tuple[Float, ...]:
    """Compute the value of the image overlap.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{left|top}_{x|y}_first"
    direction : str
        the direction of the overlap, either of "left" or "top"
    sizeY : Int
        the image width
    sizeX : Int
        the image height

    Returns
    -------
    x : Tuple[float, float]
        the computed y and x displacement

    Raises
    ------
    ValueError
        when direction is not in ["left","top"], raises ValueError
    """
    translation = np.array(
        [
            grid[f"{direction}_y_first"].values / sizeY,
            grid[f"{direction}_x_first"].values / sizeX,
        ]
    )
    translation = translation[:, np.all(np.isfinite(translation), axis=0)]
    c = predictor(translation.T)
    res = np.median(translation[:, c == 1], axis=1)
    assert len(res) == 2
    return tuple(res)


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


def filter_by_repeatability(grid: pd.DataFrame, r: Float) -> pd.DataFrame:
    """Filter the stage translation by repeatability.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position, with columns "{left|top}_{x|y|ncc}_first"
    r : Float
        the repeatability value

    Returns
    -------
    grid : pd.DataFrame
        the updated dataframe for the grid position
    """
    for _, grp in grid.groupby("col"):
        isvalid = grp["left_valid2"].astype(bool)
        if not any(isvalid):
            grid.loc[grp.index, "left_valid3"] = False
        else:
            medx = grp[isvalid]["left_y_first"].median()
            medy = grp[isvalid]["left_x_first"].median()
            grid.loc[grp.index, "left_valid3"] = (
                grp["left_y_first"].between(medx - r, medx + r)
                & grp["left_x_first"].between(medy - r, medy + r)
                & (grp["left_ncc_first"] > 0.5)
            )
    for _, grp in grid.groupby("row"):
        isvalid = grp["top_valid2"]
        if not any(isvalid):
            grid.loc[grp.index, "top_valid3"] = False
        else:
            medx = grp[isvalid]["top_y_first"].median()
            medy = grp[isvalid]["top_x_first"].median()
            grid.loc[grp.index, "top_valid3"] = (
                grp["top_y_first"].between(medx - r, medx + r)
                & grp["top_x_first"].between(medy - r, medy + r)
                & (grp["top_ncc_first"] > 0.5)
            )
    return grid


def replace_invalid_translations(grid: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid translations by estimated values.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position,
        with columns "{left|top}_{x|y}_second" and "{left|top}_valid3"

    Returns
    -------
    grid : pd.DataFrame
        the updatd dataframe for the grid position
    """
    for direction in ["left", "top"]:
        for key in ["x", "y", "ncc"]:
            isvalid = grid[f"{direction}_valid3"]
            grid.loc[isvalid, f"{direction}_{key}_second"] = grid.loc[
                isvalid, f"{direction}_{key}_first"
            ]
    for direction, rowcol in zip(["left", "top"], ["col", "row"]):
        for _, grp in grid.groupby(rowcol):
            isvalid = grp[f"{direction}_valid3"].astype(bool)
            if any(isvalid):
                assert all(
                    pd.isna(grid.loc[grp.index[~isvalid], f"{direction}_y_second"])
                )
                assert all(
                    pd.isna(grid.loc[grp.index[~isvalid], f"{direction}_x_second"])
                )
                grid.loc[grp.index[~isvalid], f"{direction}_y_second"] = grp[isvalid][
                    f"{direction}_y_first"
                ].median()
                grid.loc[grp.index[~isvalid], f"{direction}_x_second"] = grp[isvalid][
                    f"{direction}_x_first"
                ].median()
                grid.loc[grp.index[~isvalid], f"{direction}_ncc_second"] = -1
    for direction, xy in itertools.product(["left", "top"], ["x", "y"]):
        key = f"{direction}_{xy}_second"
        isna = pd.isna(grid[key])
        grid.loc[isna, key] = grid.loc[~isna, key].median()
        grid.loc[isna, f"{direction}_ncc_second"] = -1
    for direction, xy in itertools.product(["left", "top"], ["x", "y"]):
        assert np.all(np.isfinite(grid[f"{direction}_{xy}_second"]))

    return grid
