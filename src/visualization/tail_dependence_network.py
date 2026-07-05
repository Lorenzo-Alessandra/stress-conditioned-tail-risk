from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def ensure_output_directory(path: str | Path) -> Path:
    """
    Create output directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        Output directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_tail_dependence_network(
    matrix: pd.DataFrame,
    threshold: float,
) -> nx.Graph:
    """
    Build an undirected tail-dependence network from a symmetric matrix.

    Parameters
    ----------
    matrix:
        Tail-dependence matrix.
    threshold:
        Minimum off-diagonal tail-dependence value required to include an edge.

    Returns
    -------
    nx.Graph
        Tail-dependence network.
    """
    graph = nx.Graph()

    assets = list(matrix.index)

    for asset in assets:
        graph.add_node(asset)

    for row_position, asset_i in enumerate(assets):
        for column_position, asset_j in enumerate(assets):
            if column_position <= row_position:
                continue

            value = float(matrix.loc[asset_i, asset_j])

            if value >= threshold:
                graph.add_edge(
                    asset_i,
                    asset_j,
                    weight=value,
                )

    return graph


def summarize_tail_dependence_network(graph: nx.Graph) -> pd.DataFrame:
    """
    Summarize network node statistics.

    Parameters
    ----------
    graph:
        Tail-dependence network.

    Returns
    -------
    pd.DataFrame
        Node-level network summary.
    """
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))

    summary = pd.DataFrame(
        {
            "asset": list(graph.nodes()),
            "degree": [degree[node] for node in graph.nodes()],
            "weighted_degree": [weighted_degree[node] for node in graph.nodes()],
        }
    )

    summary = summary.sort_values(
        ["weighted_degree", "degree"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return summary


def extract_tail_dependence_edges(graph: nx.Graph) -> pd.DataFrame:
    """
    Extract graph edges as a dataframe.

    Parameters
    ----------
    graph:
        Tail-dependence network.

    Returns
    -------
    pd.DataFrame
        Edge list with tail-dependence weights.
    """
    records = []

    for asset_i, asset_j, data in graph.edges(data=True):
        records.append(
            {
                "asset_i": asset_i,
                "asset_j": asset_j,
                "tail_dependence": data["weight"],
            }
        )

    edges = pd.DataFrame(records)

    if edges.empty:
        return pd.DataFrame(columns=["asset_i", "asset_j", "tail_dependence"])

    edges = edges.sort_values("tail_dependence", ascending=False).reset_index(
        drop=True
    )

    return edges


def plot_tail_dependence_network(
    graph: nx.Graph,
    threshold: float,
    output_path: str | Path,
) -> None:
    """
    Plot the tail-dependence network.

    Parameters
    ----------
    graph:
        Tail-dependence network.
    threshold:
        Edge inclusion threshold.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 8))

    if graph.number_of_edges() == 0:
        raise ValueError("Graph has no edges. Lower the threshold.")

    positions = nx.spring_layout(
        graph,
        seed=42,
        weight="weight",
    )

    edge_weights = [
        graph.edges[edge]["weight"]
        for edge in graph.edges()
    ]

    edge_widths = [
        1.0 + 6.0 * (weight - threshold) / (max(edge_weights) - threshold + 1e-12)
        for weight in edge_weights
    ]

    node_sizes = [
        700 + 250 * graph.degree(node)
        for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        alpha=0.7,
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=10,
        ax=ax,
    )

    edge_labels = {
        (asset_i, asset_j): f"{data['weight']:.2f}"
        for asset_i, asset_j, data in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax,
    )

    ax.set_title(f"Tail-Dependence Network, q=0.95, threshold={threshold}")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()