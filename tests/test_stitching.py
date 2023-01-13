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


@pytest.fixture
def test_image_path_mimuelle2212(
    shared_datadir: str,
) -> Tuple[npt.NDArray, pd.DataFrame]:
    testimages = np.load(path.join(shared_datadir, "images_mimuelle2212_2.npz"))[
        "arr_0"
    ]
    props = pd.read_csv(path.join(shared_datadir, "images_mimuelle2212.csv"))
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


def test_stitching_mimuelle2212(
    test_image_path_mimuelle2212: Tuple[npt.NDArray, pd.DataFrame]
) -> None:
    testimages, props = test_image_path_mimuelle2212
    """It exits with a status code of zero."""
    cols = props["col"].to_list()
    rows = props["row"].to_list()
    result_df, _ = stitch_images(
        testimages, rows, cols, row_col_transpose=False, ncc_threshold=0.1
    )


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


def test_stitching_with_pos(test_image_path: Tuple[npt.NDArray, pd.DataFrame]) -> None:
    testimages, props = test_image_path
    """It exits with a status code of zero."""
    poss = props[["y_pos", "x_pos"]].values

    def get_pos(vals):
        vals2 = np.round(np.array(vals) / 500).astype(np.int64)
        sorted_vals = np.sort(np.unique(vals2))
        d = dict(zip(sorted_vals, np.arange(len(sorted_vals))))
        return list(map(d.get, vals2))

    pos_indices = np.array(
        [
            get_pos(poss[:, 0]),
            get_pos(poss[:, 1]),
        ]
    ).T
    assert all(pos_indices[:, 0] == props["row"])
    assert all(pos_indices[:, 1] == props["col"])
    result_df, _ = stitch_images(
        testimages, position_indices=pos_indices, row_col_transpose=False
    )
    assert np.array_equal(result_df.index, props.index)
    assert all(np.abs(result_df["y_pos"] - props["y_pos"]) <= props["allowed_error"])
    assert all(np.abs(result_df["x_pos"] - props["x_pos"]) <= props["allowed_error"])


def test_stitching_with_varied_ncc(
    test_image_path: Tuple[npt.NDArray, pd.DataFrame]
) -> None:
    testimages, props = test_image_path
    """It exits with a status code of zero."""
    cols = props["col"].to_list()
    rows = props["row"].to_list()

    inds = np.isin(cols, [1, 2, 3]) & np.isin(rows, [3, 4])
    cols = list(np.array(cols)[inds])
    rows = list(np.array(rows)[inds])
    testimages = testimages[inds]

    with pytest.raises(AssertionError):
        result_df, _ = stitch_images(
            testimages, rows, cols, row_col_transpose=False, pou=100
        )

    result_df, _ = stitch_images(
        testimages, rows, cols, row_col_transpose=False, ncc_threshold=0.01, pou=100
    )


#    assert np.array_equal(result_df.index, props.index)
#    assert all(np.abs(result_df["y_pos"] - props["y_pos"]) <= props["allowed_error"])
#    assert all(np.abs(result_df["x_pos"] - props["x_pos"]) <= props["allowed_error"])
