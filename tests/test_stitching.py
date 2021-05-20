"""Test cases for the __main__ module."""
from operator import index
from os import path

import numpy as np
import pandas as pd
import pytest
import zarr

from m2stitch import compute_stitching


@pytest.fixture
def test_image_path(shared_datadir):
    testimages = zarr.open(path.join(shared_datadir, "testimages.zarr"))
    props = pd.read_csv(path.join(shared_datadir, "testimages_props.csv"), index_col=0)
    assert np.array_equal(props.index, np.arange(testimages.shape[0]))
    return (testimages, props)


def test_stitching(test_image_path) -> None:
    testimages, props = test_image_path
    """It exits with a status code of zero."""
    rows = props["row"].to_list()
    cols = props["col"].to_list()
    result_df = compute_stitching(testimages, rows, cols)
    assert np.array_equal(result_df.index, props.index)
    assert np.max(np.abs(result_df["x_pos"] - props["x_pos"])) < 2
    assert np.max(np.abs(result_df["y_pos"] - props["y_pos"])) < 2
