import unittest


class TestWorkflow(unittest.TestCase):

    def test_fix_graph(self):
        # TODO: Add graphs and examples
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
