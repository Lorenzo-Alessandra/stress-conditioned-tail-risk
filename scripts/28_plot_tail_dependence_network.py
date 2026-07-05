import pandas as pd

from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex
from src.visualization.tail_dependence_network import (
    build_tail_dependence_network,
    extract_tail_dependence_edges,
    plot_tail_dependence_network,
    summarize_tail_dependence_network,
)


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    matrix_path = paths_config["processed_files"]["tail_dependence_matrix_q95"]

    threshold = model_config["tail_dependence"]["network_threshold"]

    print("Reading q=0.95 tail-dependence matrix...")
    matrix = pd.read_parquet(matrix_path)

    print(f"Building tail-dependence network with threshold={threshold}...")
    graph = build_tail_dependence_network(
        matrix=matrix,
        threshold=threshold,
    )

    print("Extracting network edges...")
    edges = extract_tail_dependence_edges(graph)

    print("Summarizing network nodes...")
    summary = summarize_tail_dependence_network(graph)

    edges.to_parquet(
        paths_config["processed_files"]["tail_dependence_network_edges_q95"]
    )
    summary.to_parquet(
        paths_config["processed_files"]["tail_dependence_network_summary_q95"]
    )

    edges_output = edges.copy()
    summary_output = summary.copy()

    for df in [edges_output, summary_output]:
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(6)

    edges_csv_path = "outputs/tables/tail_dependence_network_edges_q95.csv"
    summary_csv_path = "outputs/tables/tail_dependence_network_summary_q95.csv"

    edges_latex_path = "outputs/tables/tail_dependence_network_edges_q95.tex"
    summary_latex_path = "outputs/tables/tail_dependence_network_summary_q95.tex"

    save_table_csv(edges_output, edges_csv_path)
    save_table_csv(summary_output, summary_csv_path)

    save_table_latex(
        edges_output,
        edges_latex_path,
        caption="Tail-dependence network edges at the 95 percent loss threshold.",
        label="tab:tail_dependence_network_edges_q95",
    )

    save_table_latex(
        summary_output,
        summary_latex_path,
        caption="Tail-dependence network node summary at the 95 percent loss threshold.",
        label="tab:tail_dependence_network_summary_q95",
    )

    print("Plotting tail-dependence network...")
    plot_tail_dependence_network(
        graph=graph,
        threshold=threshold,
        output_path="outputs/figures/tail_dependence_network_q95.png",
    )

    print("\nTail-dependence network edges:")
    print(edges_output.to_string(index=False))

    print("\nTail-dependence network summary:")
    print(summary_output.to_string(index=False))

    print("\nTail-dependence network outputs saved successfully.")
    print(edges_csv_path)
    print(summary_csv_path)
    print("outputs/figures/tail_dependence_network_q95.png")


if __name__ == "__main__":
    main()