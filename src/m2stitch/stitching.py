"""This module provides microscope image stitching with the algorithm by MIST."""
from typing import Any
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
    rows: Sequence[Any],
    cols: Sequence[Any],
    pou: Float = 3,
    overlap_prob_uniform_threshold: Float = 80,
    full_output: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Compute image positions for stitching.

    Parameters
    ---------
    images : np.ndarray
        the images to stitch.

    rows : list
        the row indices of the images

    cols : list
        the col indices of the images

    pou : Float, default 3
        the "percent overlap uncertainty" parameter

    overlap_prob_uniform_threshold : Float, default 80
        the upper threshold for "uniform probability".
        raise this value to ease assumption
        that the displacement is following non-uniform distribution.

    full_output : bool, default False
        if True, returns the full computation result in the pd.DataFrame

    Returns
    -------
    grid : pd.DataFrame
        the result dataframe with the rows "x_pos" and "y_pos" whose values are
        the absolute positions.

    prop_dict : dict
        the dict of estimated parameters. (to be documented)
    """
    images = np.array(images)
    assert images.shape[0] == len(rows)
    assert images.shape[0] == len(cols)
    assert 0 <= overlap_prob_uniform_threshold and overlap_prob_uniform_threshold <= 100

    W, H = images.shape[1:]

    grid = pd.DataFrame(
        {
            "row": rows,
            "col": cols,
        },
        index=np.arange(len(rows)),
    )

    def get_index(row, col):
        df = grid[(grid["row"] == row) & (grid["col"] == col)]
        assert len(df) < 2
        if len(df) == 1:
            return df.index[0]
        else:
            return None

    grid["top"] = grid.apply(
        lambda g: get_index(g["row"] - 1, g["col"]), axis=1
    ).astype(pd.Int32Dtype())
    grid["left"] = grid.apply(
        lambda g: get_index(g["row"], g["col"] - 1), axis=1
    ).astype(pd.Int32Dtype())

    ###### translationComputation ######
    for direction in ["top", "left"]:
        for i2, g in tqdm(grid.iterrows(), total=len(grid)):
            i1 = g[direction]
            if pd.isna(i1):
                continue
            image1 = images[i1]
            image2 = images[i2]

            PCM = pcm(image1, image2).real
            found_peaks = list(zip(*multi_peak_max(PCM)))

            interpreted_peaks = []
            for r, c, _ in found_peaks:
                interpreted_peaks.append(interpret_translation(image1, image2, r, c))
            max_peak = interpreted_peaks[np.argmax(np.array(interpreted_peaks)[:, 0])]
            for j, key in enumerate(["ncc", "x", "y"]):
                grid.loc[i2, f"{direction}_{key}_first"] = max_peak[j]

    prob_uniform_n, mu_n, sigma_n = compute_image_overlap(
        grid, "top", W, H, prob_uniform_threshold=overlap_prob_uniform_threshold
    )
    overlap_n = 100 - mu_n
    prob_uniform_w, mu_w, sigma_w = compute_image_overlap(
        grid, "left", W, H, prob_uniform_threshold=overlap_prob_uniform_threshold
    )
    overlap_w = 100 - mu_w

    overlap_n = np.clip(overlap_n, pou, 100 - pou)
    overlap_w = np.clip(overlap_w, pou, 100 - pou)

    grid, r = compute_repeatability(grid, overlap_n, overlap_w, W, H, pou)
    grid = filter_by_repeatability(grid, r)
    grid = replace_invalid_translations(grid)

    grid = refine_translations(images, grid, r)

    tree = compute_maximum_spanning_tree(grid)
    grid = compute_final_position(grid, tree)

    prop_dict = {
        "W": W,
        "H": H,
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
        return grid[["row", "col", "x_pos", "y_pos"]], prop_dict
