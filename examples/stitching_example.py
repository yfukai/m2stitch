# coding: utf-8
from os import path

import numpy as np
import pandas as pd

import m2stitch

script_path = path.dirname(path.realpath(__file__))

image_file_path = path.join(script_path, "../tests/data/testimages.npy")
props_file_path = path.join(script_path, "../tests/data/testimages_props.csv")
images = np.load(image_file_path)
props = pd.read_csv(props_file_path, index_col=0)

rows = props["row"].to_list()
cols = props["col"].to_list()

print(images.shape)
# must be 3-dim, with each dimension meaning (tile_index,x,y)
print(rows)
# the row (second-last dim.) indices for each tile index. for example, [1,1,2,2,2,...]
print(cols)
# the column (last dim.) indices for each tile index. for example, [2,3,1,2,3,...]

# Note : the row_col_transpose=True is kept only for the sake of version compatibility.
# In the mejor version, the row_col_transpose=False will be the default.
result_df, _ = m2stitch.stitch_images(images, rows, cols, row_col_transpose=False)

print(result_df["y_pos"])
# the absolute y (second last dim.) positions of the tiles
print(result_df["x_pos"])
# the absolute x (last dim.) positions of the tiles

# stitching example
result_df["y_pos2"] = result_df["y_pos"] - result_df["y_pos"].min()
result_df["x_pos2"] = result_df["x_pos"] - result_df["x_pos"].min()

size_y = images.shape[1]
size_x = images.shape[2]

stitched_image_size = (
    result_df["y_pos2"].max() + size_y,
    result_df["x_pos2"].max() + size_x,
)
stitched_image = np.zeros_like(images, shape=stitched_image_size)
for i, row in result_df.iterrows():
    stitched_image[
        row["y_pos2"] : row["y_pos2"] + size_y,
        row["x_pos2"] : row["x_pos2"] + size_x,
    ] = images[i]

result_image_file_path = path.join(script_path, "stitched_image.npy")
np.save(result_image_file_path, stitched_image)
