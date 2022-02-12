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
# the row indices (direction in the last dimension) for each tile index. for example, [1,1,2,2,2,...]
print(cols)
# the column indices (direction in the second last dinemsion) for each tile index. for example, [2,3,1,2,3,...]

result_df, _ = m2stitch.stitch_images(images, rows, cols, row_col_transpose=False)
# note: the previous default row_col_transpose=True is deprecated and will be removed

print(result_df["y_pos"])
# the absolute y positions (second last dimension) of the tiles
print(result_df["x_pos"])
# the absolute x positions (last dimension) of the tiles

# stitching example
result_df["y_pos2"] = result_df["y_pos"] - result_df["y_pos"].min()
result_df["x_pos2"] = result_df["x_pos"] - result_df["x_pos"].min()

size_y = images.shape[-2]
size_x = images.shape[-1]

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
