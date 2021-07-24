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
    rows = props["row"].to_list()
    cols = props["col"].to_list()
    result_df, _ = stitch_images(testimages, rows, cols)
    assert np.array_equal(result_df.index, props.index)
    assert np.max(np.abs(result_df["x_pos"] - props["x_pos"])) < 2
    assert np.max(np.abs(result_df["y_pos"] - props["y_pos"])) < 2
