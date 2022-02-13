"""Test cases for the __main__ module."""
from os import path
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from m2stitch import stitch_images


@pytest.fixture
def test_image_path(shared_datadir: str) -> Tuple[npt.NDArray, pd.DataFrame]:
    testimages = np.load(path.join(shared_datadir, "testimages.npy"))
    props = pd.read_csv(path.join(shared_datadir, "testimages_props.csv"), index_col=0)
    assert np.array_equal(props.index, np.arange(testimages.shape[0]))
    return (testimages, props)


def test_stitching(test_image_path: Tuple[npt.NDArray, pd.DataFrame]) -> None:
    testimages, props = test_image_path
    """It exits with a status code of zero."""
    cols = props["col"].to_list()
    rows = props["row"].to_list()
    result_df, _ = stitch_images(testimages, rows, cols, row_col_transpose=False)
    assert np.array_equal(result_df.index, props.index)
    assert all(np.abs(result_df["y_pos"] - props["y_pos"]) <= props["allowed_error"])
    assert all(np.abs(result_df["x_pos"] - props["x_pos"]) <= props["allowed_error"])


def test_stitching_init_guess(
    test_image_path: Tuple[npt.NDArray, pd.DataFrame]
) -> None:
    np.random.seed(0)
    testimages, props = test_image_path
    """It exits with a status code of zero."""
    cols = props["col"].to_list()
    rows = props["row"].to_list()
    pos_guess = props[["y_pos", "x_pos"]].values
    result_df, _ = stitch_images(
        testimages,
        rows,
        cols,
        row_col_transpose=False,
        position_initial_guess=pos_guess,
    )
    assert np.array_equal(result_df.index, props.index)
    assert all(np.abs(result_df["y_pos"] - props["y_pos"]) <= props["allowed_error"])
    assert all(np.abs(result_df["x_pos"] - props["x_pos"]) <= props["allowed_error"])
