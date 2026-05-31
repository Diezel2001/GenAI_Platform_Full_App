"""
langgraph_visualizer.py

Visualize a compiled LangGraph and save it as PNG.

Requirements:
    pip install langgraph
"""

from pathlib import Path
from typing import Optional


def visualize_langgraph(
    compiled_graph,
    output_file: str = "langgraph.png",
    show_mermaid: bool = False,
) -> str:
    """
    Visualize a compiled LangGraph.

    Args:
        compiled_graph: Result of workflow.compile()
        output_file: Output PNG filename
        show_mermaid: Print Mermaid source to console

    Returns:
        Path to generated file
    """

    try:
        graph = compiled_graph.get_graph()

        if show_mermaid:
            print("\n=== Mermaid Diagram ===\n")
            print(graph.draw_mermaid())
            print("\n=======================\n")

        png_bytes = graph.draw_mermaid_png()

        output_path = Path(output_file)
        output_path.write_bytes(png_bytes)

        print(f"✓ Graph image saved: {output_path.resolve()}")
        return str(output_path.resolve())

    except Exception as exc:
        print(f"Failed to generate PNG: {exc}")

        try:
            mermaid_text = compiled_graph.get_graph().draw_mermaid()

            fallback_path = Path(output_file).with_suffix(".mmd")
            fallback_path.write_text(mermaid_text, encoding="utf-8")

            print(
                f"✓ Mermaid source saved instead: "
                f"{fallback_path.resolve()}"
            )

            return str(fallback_path.resolve())

        except Exception as inner_exc:
            raise RuntimeError(
                f"Could not visualize graph.\n"
                f"PNG Error: {exc}\n"
                f"Mermaid Error: {inner_exc}"
            ) from inner_exc


def print_graph_structure(compiled_graph) -> None:
    """
    Print Mermaid representation to terminal.
    Useful when image generation is unavailable.
    """

    print(compiled_graph.get_graph().draw_mermaid())


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

# if __name__ == "__main__":
#     from typing import TypedDict
#     from langgraph.graph import StateGraph, END

#     class State(TypedDict):
#         message: str

#     def start_node(state: State):
#         return state

#     def process_node(state: State):
#         return state

#     workflow = StateGraph(State)

#     workflow.add_node("start", start_node)
#     workflow.add_node("process", process_node)

#     workflow.set_entry_point("start")
#     workflow.add_edge("start", "process")
#     workflow.add_edge("process", END)

#     compiled_graph = workflow.compile()

#     visualize_langgraph(
#         compiled_graph,
#         output_file="workflow.png",
#         show_mermaid=True,
#     )