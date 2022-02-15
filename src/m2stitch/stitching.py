"""This module provides microscope image stitching with the algorithm by MIST."""
import itertools
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm

from ._constrained_refinement import refine_translations
from ._global_optimization import compute_final_position
from ._global_optimization import compute_maximum_spanning_tree
from ._stage_model import compute_image_overlap2
from ._stage_model import filter_by_overlap_and_correlation
from ._stage_model import filter_by_repeatability
from ._stage_model import filter_outliers
from ._stage_model import replace_invalid_translations
from ._translation_computation import interpret_translation
from ._translation_computation import multi_peak_max
from ._translation_computation import pcm
from ._typing_utils import BoolArray
from ._typing_utils import Float
from ._typing_utils import NumArray


@dataclass
class ElipticEnvelopPredictor:
    contamination: float
    epsilon: float
    random_seed: int

    def __call__(self, X: NumArray) -> BoolArray:
        ee = EllipticEnvelope(contamination=self.contamination)
        rng = np.random.default_rng(self.random_seed)
        X = rng.normal(size=X.shape) * self.epsilon + X
        return ee.fit_predict(X) > 0


def stitch_images(
    images: Union[Sequence[NumArray], NumArray],
    rows: Optional[Sequence[Any]] = None,
    cols: Optional[Sequence[Any]] = None,
    position_indices: Optional[NumArray] = None,
    position_initial_guess: Optional[NumArray] = None,
    overlap_diff_threshold: Float = 10,
    pou: Float = 3,
    full_output: bool = False,
    row_col_transpose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Compute image positions for stitching.

    Parameters
    ---------
    images : np.ndarray
        the images to stitch.

    rows : list, optional
        the row indices (tile position in the second last dimension) of the images.

    cols : list, optional
        the column indices (tile position in the last dimension) of the images

    position_indices : np.ndarray, optional
        the tile position indices in each dimension.
        the dimensions corresponds to (image, index)
        ignored if rows and cols are not None.

    position_initial_guess : np.ndarray, optional
        the initial guess for the positions of the images, in the unit of pixels.

    overlap_diff_threshold : 10
        the allowed difference from the initial guess, in percentage of the image size.
        ignored if position_initial_guess is None

    pou : Float, default 3
        the "percent overlap uncertainty" parameter

    full_output : bool, default False
        if True, returns the full comptutation result in the pd.DataFrame

    row_col_transpose : bool, default True
        if True, row and col indices are switched.
        only for compatibility and the default value will be False in the future.

    Returns
    -------
    grid : pd.DataFrame
        the result dataframe with the rows "x_pos" and "y_pos" whose values are
        the absolute positions.

    prop_dict : dict
        the dict of estimated parameters. (to be documented)
    """
    images = np.array(images)
    assert (position_indices is not None) or (rows is not None and cols is not None)
    if position_indices is None:
        if row_col_transpose:
            warnings.warn(
                "row_col_transpose is True. The default value will be changed to False in the major release."
            )
            position_indices = np.array([cols, rows]).T
        else:
            position_indices = np.array([rows, cols]).T
    position_indices = np.array(position_indices)
    assert images.shape[0] == position_indices.shape[0]
    assert position_indices.shape[1] == images.ndim - 1
    if position_initial_guess is not None:
        position_initial_guess = np.array(position_initial_guess)
        assert images.shape[0] == position_indices.shape[0]
        assert position_initial_guess.shape[1] == images.ndim - 1
    assert 0 <= overlap_diff_threshold and overlap_diff_threshold <= 100
    _rows, _cols = position_indices.T

    sizeY, sizeX = images.shape[1:]

    grid = pd.DataFrame(
        {
            "col": _cols,
            "row": _rows,
        },
        index=np.arange(len(_cols)),
    )

    def get_index(col, row):
        df = grid[(grid["col"] == col) & (grid["row"] == row)]
        assert len(df) < 2
        if len(df) == 1:
            return df.index[0]
        else:
            return None

    grid["top"] = grid.apply(
        lambda g: get_index(g["col"], g["row"] - 1), axis=1
    ).astype(pd.Int32Dtype())
    grid["left"] = grid.apply(
        lambda g: get_index(g["col"] - 1, g["row"]), axis=1
    ).astype(pd.Int32Dtype())

    ### dimension order ... m.y.x
    if position_initial_guess is not None:
        for j, dimension in enumerate(["y", "x"]):
            grid[f"{dimension}_pos_init_guess"] = position_initial_guess[:, j]
        for direction, dimension in itertools.product(["left", "top"], ["y", "x"]):
            for ind, g in grid.iterrows():
                i1 = g[direction]
                if pd.isna(i1):
                    continue
                g2 = grid.loc[i1]
                grid.loc[ind, f"{direction}_{dimension}_init_guess"] = (
                    g[f"{dimension}_pos_init_guess"] - g2[f"{dimension}_pos_init_guess"]
                )

    ###### translationComputation ######
    for direction in ["left", "top"]:
        for i2, g in tqdm(grid.iterrows(), total=len(grid)):
            i1 = g[direction]
            if pd.isna(i1):
                continue
            image1 = images[i1]
            image2 = images[i2]

            PCM = pcm(image1, image2).real
            if position_initial_guess is not None:

                def get_lims(dimension, size):
                    val = g[f"{direction}_{dimension}_init_guess"]
                    r = size * overlap_diff_threshold / 100.0
                    return np.round([val - r, val + r]).astype(np.int64)

                lims = np.array(
                    [
                        get_lims(dimension, size)
                        for dimension, size in zip("yx", [sizeY, sizeX])
                    ]
                )
            else:
                lims = np.array([[-sizeY, sizeY], [-sizeX, sizeX]])
            yins, xins, _ = multi_peak_max(PCM)
            max_peak = interpret_translation(
                image1, image2, yins, xins, *lims[0], *lims[1]
            )
            for j, key in enumerate(["ncc", "y", "x"]):
                grid.loc[i2, f"{direction}_{key}_first"] = max_peak[j]

    # TODO make threshold adjustable
    assert np.any(grid["top_ncc_first"] > 0.5), "there is no good top pair"
    assert np.any(grid["left_ncc_first"] > 0.5), "there is no good left pair"
    predictor = ElipticEnvelopPredictor(contamination=0.4, epsilon=0.01, random_seed=0)
    left_displacement = compute_image_overlap2(
        grid[grid["left_ncc_first"] > 0.5], "left", sizeY, sizeX, predictor
    )
    top_displacement = compute_image_overlap2(
        grid[grid["top_ncc_first"] > 0.5], "top", sizeY, sizeX, predictor
    )
    overlap_top = np.clip(100 - top_displacement[0] * 100, pou, 100 - pou)
    overlap_left = np.clip(100 - left_displacement[1] * 100, pou, 100 - pou)

    ### compute_repeatability ###
    grid["top_valid1"] = filter_by_overlap_and_correlation(
        grid["top_y_first"], grid["top_ncc_first"], overlap_top, sizeY, pou
    )
    grid["top_valid2"] = filter_outliers(grid["top_y_first"], grid["top_valid1"])
    grid["left_valid1"] = filter_by_overlap_and_correlation(
        grid["left_x_first"], grid["left_ncc_first"], overlap_left, sizeX, pou
    )
    grid["left_valid2"] = filter_outliers(grid["left_x_first"], grid["left_valid1"])

    rs = []
    for direction, dims, rowcol in zip(["top", "left"], ["yx", "xy"], ["col", "row"]):
        valid_key = f"{direction}_valid2"
        valid_grid = grid[grid[valid_key]]
        if len(valid_grid) > 0:
            w1s = valid_grid[f"{direction}_{dims[0]}_first"]
            r1 = np.ceil((w1s.max() - w1s.min()) / 2)
            _, w2s = zip(*valid_grid.groupby(rowcol)[f"{direction}_{dims[1]}_first"])
            r2 = np.ceil(np.max([np.max(w2) - np.min(w2) for w2 in w2s]) / 2)
            rs.append(max(r1, r2))
        rs.append(0)
    r = np.max(rs)

    grid = filter_by_repeatability(grid, r)
    grid = replace_invalid_translations(grid)

    grid = refine_translations(images, grid, r)

    tree = compute_maximum_spanning_tree(grid)
    grid = compute_final_position(grid, tree)

    prop_dict = {
        "W": sizeY,
        "H": sizeX,
        "overlap_left": overlap_left,
        "overlap_top": overlap_top,
        "repeatability": r,
    }
    if row_col_transpose:
        grid = grid.rename(columns={"x_pos": "y_pos", "y_pos": "x_pos"})
    if full_output:
        return grid, prop_dict
    else:
        return grid[["row", "col", "y_pos", "x_pos"]], prop_dict
