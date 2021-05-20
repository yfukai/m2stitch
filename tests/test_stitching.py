"""Test cases for the __main__ module."""
import pytest

#from m2stitch import stitching
#import zarr
#
#
#@pytest.fixture
#def test_image_path(shared_datadir):
#
#
#
#
#def test_stitching() -> None:
#    """It exits with a status code of zero."""
#    result = runner.invoke(__main__.main)
#    assert result.exit_code == 0
#rowcol_df=pd.read_csv("../tests/data/testimages_rowcol.csv",index_col=0)#
#rows=rowcol_df["row"].to_list() ; cols=rowcol_df["col"].to_list()
#testimg=zarr.open("../tests/data/testimages.zarr","r")
#m2stitch.compute_stitching(testimg,rows,cols)