import networkx as nx
import numpy as np
import pandas as pd


def compute_maximum_spanning_tree(grid):
    connection_graph = nx.Graph()
    for i, g in grid.iterrows():
        for direction in ["north", "west"]:
            if not pd.isna(g[direction]):
                weight = g[f"{direction}_ncc"]
                if g[f"{direction}_valid3"]:
                    weight = weight + 10
                connection_graph.add_edge(
                    i,
                    g[direction],
                    weight=weight,
                    direction=direction,
                    f=i,
                    t=g[direction],
                    x=g[f"{direction}_x"],
                    y=g[f"{direction}_y"],
                )
    return nx.maximum_spanning_tree(connection_graph)


def compute_final_position(grid, tree, source_index=0):
    grid.loc[source_index, "x_pos"] = 0
    grid.loc[source_index, "y_pos"] = 0

    nodes = [source_index]
    walked_nodes = []
    while len(nodes) > 0:
        node = nodes.pop()
        walked_nodes.append(node)
        for adj, props in tree.adj[node].items():
            if not adj in walked_nodes:
                assert (props["f"] == node) & (props["t"] == adj) or (
                    props["t"] == node
                ) & (props["f"] == adj)
                nodes.append(adj)
                x_pos = grid.loc[node, "x_pos"]
                y_pos = grid.loc[node, "y_pos"]

                if node == props["t"]:
                    grid.loc[adj, "x_pos"] = x_pos + props["x"]
                    grid.loc[adj, "y_pos"] = y_pos + props["y"]
                else:
                    grid.loc[adj, "x_pos"] = x_pos - props["x"]
                    grid.loc[adj, "y_pos"] = y_pos - props["y"]
    assert not any(pd.isna(grid["x_pos"]))
    assert not any(pd.isna(grid["y_pos"]))
    grid["x_pos"] = grid["x_pos"] - grid["x_pos"].min()
    grid["y_pos"] = grid["y_pos"] - grid["y_pos"].min()
    grid["x_pos"] = grid["x_pos"].astype(np.int32)
    grid["y_pos"] = grid["y_pos"].astype(np.int32)

    return grid
