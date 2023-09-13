from typing import Iterable, Tuple, Optional

from y0.graph import NxMixedGraph
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators.CITests import pearsonr, chi_square, cressie_read, freeman_tuckey, g_sq, log_likelihood, modified_log_likelihood, power_divergence, neyman
from pgmpy.independencies import IndependenceAssertion


tests = {
    "pearson": pearsonr,
    "chi-square": chi_square,
    "cressie_read": cressie_read,
    "freeman_tuckey": freeman_tuckey,
    "g_sq": g_sq,
    "log_likelihood": log_likelihood,
    "modified_log_likelihood": modified_log_likelihood,
    "power_divergence": power_divergence,
    "neyman": neyman
}


def to_pgmpy_dag(graph: NxMixedGraph) -> DAG:
    """Converts a y0 graph to its equivalent pgmpy representation"""
    import networkx as nx
    latent_variable_dag = graph.to_latent_variable_dag()
    latents = {node for node, is_hidden in nx.get_node_attributes(latent_variable_dag, "hidden").items() if is_hidden}
    dag = DAG(ebunch=latent_variable_dag.edges, latents=latents)
    return dag


def evaluate_test(graph: NxMixedGraph,
                  data: pd.DataFrame,
                  significance_level: Optional[float] = 0.05,
                  test: Optional[str] = None
                  ) -> Iterable[Tuple[IndependenceAssertion, bool]]:
    """Runs the given conditional independency test on the given network"""
    # TODO: Use a default test for a particular type of data
    dag = to_pgmpy_dag(graph)
    conditional_independencies = dag.get_independencies().get_assertions()
    conditional_independencies = conditional_independencies[:len(conditional_independencies) // 2]
    result = []
    for conditional_independency in conditional_independencies:
        conditional_independency = conditional_independency.get_assertion()
        test_func = tests[test]
        for X in conditional_independency[0]:
            for Y in conditional_independency[1]:
                Z = conditional_independency[2]
                test_result = test_func(X=X,
                                        Y=Y,
                                        Z=Z,
                                        data=data,
                                        boolean=True,
                                        significance_level=significance_level)
                result.append((IndependenceAssertion(X, Y, Z), test_result))
    return result
