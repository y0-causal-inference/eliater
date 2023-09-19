import y0

from .multi_med_confounder import generate_data_for_multi_med_confounder
from y0.graph import NxMixedGraph
from y0.examples import Example
from y0.dsl import Variable, X, Y, Z1, Z2, Z3
from y0.algorithm.identify import Query

M1 = y0.dsl.Variable("M1")
M2 = y0.dsl.Variable("M2")
R1 = y0.dsl.Variable("R1")
R2 = y0.dsl.Variable("R2")
R3 = y0.dsl.Variable("R3")

#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
multi_med_confounder = NxMixedGraph.from_edges(
    directed=[
        (Z1, X),
        (X, M1),
        (M1, M2),
        (M2, Y),
        (Z1, Z2),
        (Z2, Z3),
        (Z3, Y),
        (M1, R1),
        (R1, R2),
        (R2, R3),
        (Y, R3),
    ],
    undirected=[
        (Z1, X),
    ],
)

multi_med_confounder_example = Example(
    name="Multi_mediators_confounders",
    reference='Causal workflow paper, figure 4',
    graph=multi_med_confounder,
    generate_data=generate_data_for_multi_med_confounder,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)