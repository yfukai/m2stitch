"""This module provides microscope image stitching with the algorithm by MIST."""
import itertools
import warnings
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._constrained_refinement import refine_translations
from ._global_optimization import compute_final_position
from ._global_optimization import compute_maximum_spanning_tree
from ._stage_model import compute_image_overlap
from ._stage_model import compute_repeatability
from ._stage_model import filter_by_repeatability
from ._stage_model import replace_invalid_translations
from ._translation_computation import interpret_translation
from ._translation_computation import multi_peak_max
from ._translation_computation import pcm
from ._typing_utils import Float
from ._typing_utils import NumArray


def stitch_images(
    images: Union[Sequence[NumArray], NumArray],
    rows: Optional[Sequence[Any]] = None,
    cols: Optional[Sequence[Any]] = None,
    position_indices: Optional[NumArray] = None,
    position_initial_guess: Optional[NumArray] = None,
    overlap_diff_threshold: Float = 10,
    pou: Float = 3,
    overlap_prob_uniform_threshold: Float = 80,
    full_output: bool = False,
    row_col_transpose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Compute image positions for stitching.

    Parameters
    ---------
    images : np.ndarray
        the images to stitch.

    rows : list, optional
        the row indices (tile position in the last dimension) of the images.

    cols : list, optional
        the column indices (tile position in the second last dimension) of the images

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

    overlap_prob_uniform_threshold : Float, default 80
        the upper threshold for "uniform probability".
        raise this value to ease assumption
        that the displacement is following non-uniform distribution.

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
            position_indices = np.array([rows, cols]).T
        else:
            position_indices = np.array([cols, rows]).T
    position_indices = np.array(position_indices)
    assert images.shape[0] == position_indices.shape[0]
    assert position_indices.shape[1] == images.ndim - 1
    if position_initial_guess is not None:
        position_initial_guess = np.array(position_initial_guess)
        assert images.shape[0] == position_indices.shape[0]
        assert position_initial_guess.shape[1] == images.ndim - 1
    assert 0 <= overlap_prob_uniform_threshold and overlap_prob_uniform_threshold <= 100
    assert 0 <= overlap_diff_threshold and overlap_diff_threshold <= 100
    _cols, _rows = position_indices.T

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
        lambda g: get_index(g["col"] - 1, g["row"]), axis=1
    ).astype(pd.Int32Dtype())
    grid["left"] = grid.apply(
        lambda g: get_index(g["col"], g["row"] - 1), axis=1
    ).astype(pd.Int32Dtype())

    ### dimension order ... m.y.x
    if position_initial_guess is not None:
        for j, dimension in enumerate(["y", "x"]):
            grid[f"{dimension}_pos_init_guess"] = position_initial_guess[:, j]
        for direction, dimension in itertools.product(["top", "left"], ["y", "x"]):
            for ind, g in grid.iterrows():
                i1 = g[direction]
                if pd.isna(i1):
                    continue
                g2 = grid.loc[i1]
                grid.loc[ind, f"{direction}_{dimension}_init_guess"] = (
                    g[f"{dimension}_pos_init_guess"] - g2[f"{dimension}_pos_init_guess"]
                )

    ###### translationComputation ######
    for direction in ["top", "left"]:
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
                    return (val - r, val + r)

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
            for j, key in enumerate(["ncc", "x", "y"]):
                grid.loc[i2, f"{direction}_{key}_first"] = max_peak[j]

    prob_uniform_n, mu_n, sigma_n = compute_image_overlap(
        grid, "top", sizeY, sizeX, prob_uniform_threshold=overlap_prob_uniform_threshold
    )
    overlap_n = 100 - mu_n
    prob_uniform_w, mu_w, sigma_w = compute_image_overlap(
        grid,
        "left",
        sizeY,
        sizeX,
        prob_uniform_threshold=overlap_prob_uniform_threshold,
    )
    overlap_w = 100 - mu_w

    overlap_n = np.clip(overlap_n, pou, 100 - pou)
    overlap_w = np.clip(overlap_w, pou, 100 - pou)

    grid, r = compute_repeatability(grid, overlap_n, overlap_w, sizeY, sizeX, pou)
    grid = filter_by_repeatability(grid, r)
    grid = replace_invalid_translations(grid)

    grid = refine_translations(images, grid, r)

    tree = compute_maximum_spanning_tree(grid)
    grid = compute_final_position(grid, tree)

    prop_dict = {
        "W": sizeY,
        "H": sizeX,
        "overlap_top": overlap_n,
        "overlap_left": overlap_w,
        "overlap_top_results": {
            "prob_uniform": prob_uniform_n,
            "mu": mu_n,
            "sigma": sigma_n,
        },
        "overlap_left_results": {
            "prob_uniform": prob_uniform_w,
            "mu": mu_w,
            "sigma": sigma_w,
        },
        "repeatability": r,
    }
    if full_output:
        return grid, prop_dict
    else:
        return grid[["row", "col", "y_pos", "x_pos"]], prop_dict
