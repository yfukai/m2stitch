import numpy as np
import pandas as pd
from tqdm import tqdm

from .constrained_refinement import refine_translations
from .global_optimization import compute_final_position
from .global_optimization import compute_maximum_spanning_tree
from .stage_model import compute_image_overlap
from .stage_model import compute_repeatability
from .stage_model import filter_by_repeatability
from .stage_model import replace_invalid_translations
from .translation_computation import interpret_translation
from .translation_computation import multi_peak_max
from .translation_computation import pcm


def compute_stitching(images, rows, cols, pou=3, full_output=False):
    images = np.array(images)
    assert images.shape[0] == len(rows)
    assert images.shape[0] == len(cols)

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

    grid["north"] = grid.apply(
        lambda g: get_index(g["row"] - 1, g["col"]), axis=1
    ).astype(pd.Int32Dtype())
    grid["west"] = grid.apply(
        lambda g: get_index(g["row"], g["col"] - 1), axis=1
    ).astype(pd.Int32Dtype())

    ###### translationComputation ######
    for direction in ["north", "west"]:
        for i2, g in tqdm(grid.iterrows(), total=len(grid)):
            i1 = g[direction]
            if pd.isna(i1):
                continue
            image1 = images[i1]
            image2 = images[i2]

            PCM = pcm(image1, image2).real
            found_peaks = list(zip(*multi_peak_max(PCM)))

            interpreted_peaks = []
            for r, c, v in found_peaks:
                interpreted_peaks.append(interpret_translation(image1, image2, r, c))
            max_peak = interpreted_peaks[np.argmax(np.array(interpreted_peaks)[:, 0])]
            for j, key in enumerate(["ncc", "x", "y"]):
                grid.loc[i2, f"{direction}_{key}_first"] = max_peak[j]

    prob_uniform_n, mu_n, sigma_n = compute_image_overlap(grid, "north", W, H)
    overlap_n = 100 - mu_n
    prob_uniform_w, mu_w, sigma_w = compute_image_overlap(grid, "west", W, H)
    overlap_w = 100 - mu_w

    overlap_n = np.clip(overlap_n, pou, 100 - pou)
    overlap_w = np.clip(overlap_w, pou, 100 - pou)

    grid, r = compute_repeatability(grid, overlap_n, overlap_w, W, H, pou)
    grid = filter_by_repeatability(grid, r)
    grid = replace_invalid_translations(grid)

    grid = refine_translations(images,grid, r)

    tree = compute_maximum_spanning_tree(grid)
    grid = compute_final_position(grid, tree)

    if full_output:
        return grid
    else:
        return grid[["row","col","x_pos","y_pos"]]