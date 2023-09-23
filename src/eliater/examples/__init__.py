import y0
from y0.algorithm.identify import Query
from y0.dsl import Z1, Z2, Z3, Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

from .multi_med import generate_data_for_multi_med
from .multi_med_confounder import generate_data_for_multi_med_confounder
from .multi_med_confounder_nuisance_var import generate_data_for_multi_med_confounder_nuisance_var

M1 = y0.dsl.Variable("M1")
M2 = y0.dsl.Variable("M2")
R1 = y0.dsl.Variable("R1")
R2 = y0.dsl.Variable("R2")
R3 = y0.dsl.Variable("R3")

#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
multi_med = NxMixedGraph.from_edges(
    directed=[
        (X, M1),
        (M1, M2),
        (M2, Y),
    ],
    undirected=[
        (X, Y),
        # (M1, Y) We generated data for this graph with the assumption that the bi-directed edge between
        # M1 and Y is present. However, we assume that the prior knowledge graph does not have this information.
    ],
)

multi_med_example = Example(
    name="Multi_mediators",
    reference="Inspired by the frontdoor example, but with multiple mediators.",
    graph=multi_med,
    generate_data=generate_data_for_multi_med,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)

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
    ],
    undirected=[
        (Z1, X),
        # (Y, Z2)
        # We are generating data with the assumption that there is a bi-directed edge between
        # Y and Z2, but that bi-directed edge is missed from this prior knowledge graph.
    ],
)

multi_med_confounder_example = Example(
    name="Multi_mediators_confounders",
    reference="Causal workflow paper, figure 4 (b). The query can be estimated with both front-door and back-door approaches",
    graph=multi_med_confounder,
    generate_data=generate_data_for_multi_med_confounder,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)


#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
multi_med_confounder_nuisance_var = NxMixedGraph.from_edges(
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
        # (Y, Z2)
        # We are generating data with the assumption that there is a bi-directed edge between
        # Y and Z2, but that bi-directed edge is missed from this prior knowledge graph.
    ],
)

multi_med_confounder_nuisance_var_example = Example(
    name="Multi_mediators_confounders_nuisance_var",
    reference="Causal workflow paper, figure 4 (a). The query can be estimated with both front-door and back-door approaches",
    graph=multi_med_confounder_nuisance_var,
    generate_data=generate_data_for_multi_med_confounder_nuisance_var,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)

continuous = NxMixedGraph.from_str_edges(
    directed=[
        ("W", "X"),
        ("X", "Z"),
        ("Z", "Y"),
        ("W", "Y")
    ]
)
