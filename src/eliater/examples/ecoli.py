"""Examples for transcriptional Escherichia coli K-12 regulatory network.

The E. Coli regulatory network was extracted manually (by hand) from the EcoCyc
database [Keseler2021]_ . The nodes represent genes, and the edges represent
regulatory relationships.

.. [Keseler2021] `The EcoCyc database in 2021 <https://doi.org/10.3389/fmicb.2021.711077>`_
"""

# FIXME what kind of regulatory relationships?
# FIXME What kinds of experiments did they come from?
# FIXME What was the method that these experiments got into the database?
# FIXME How was the network extracted manually? What was the thought process?

from y0.algorithm.identify import Query
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "ecoli_transcription_example",
]

graph = NxMixedGraph.from_str_adj(
    directed={
        "appY": ["appA", "appB", "hyaA", "hyaB", "hyaF"],
        "arcA": [
            "rpoS",
            "fnr",
            "dpiA",
            "aceE",
            "appY",
            "dpiB",
            "cydD",
            "hyaA",
            "hyaB",
            "hyaF",
            "mdh",
            "lrp",
            "ydeO",
            "oxyR",
        ],
        "btsR": ["mdh"],
        "cra": ["cyoA"],
        "crp": [
            "dpiA",
            "cirA",
            "dcuR",
            "oxyR",
            "fis",
            "fur",
            "aceE",
            "dpiB",
            "cyoA",
            "exuT",
            "gadX",
            "mdh",
            "gutM",
        ],
        "cspA": ["hns"],
        "dcuR": ["dpiA", "dpiB"],
        "dpiA": ["appY", "citC", "dpiB", "exuT", "mdh"],
        "fis": ["cyoA", "gadX", "hns", "hyaA", "hyaB", "hyaF"],
        "fnr": [
            "dcuR",
            "dpiA",
            "narL",
            "aceE",
            "amtB",
            "aspC",
            "dpiB",
            "cydD",
            "cyoA",
            "gadX",
            "hcp",
        ],
        "fur": ["fnr", "amtB", "aspC", "cirA", "cyoA"],
        "gadX": ["amtB", "hns"],
        "hns": ["appY", "ydeO", "gutM"],
        "ihfA": ["crp", "fnr", "ihfB"],
        "ihfB": ["fnr"],
        "iscR": ["hyaA", "hyaB"],
        "lrp": ["soxS", "aspC"],
        "modE": ["narL"],
        "narL": ["dpiB", "cydD", "hcp", "hyaA", "hyaB", "hyaF", "dcuR", "dpiA"],
        "narP": ["hyaA", "hyaB", "hyaF"],
        "oxyR": ["fur", "hcp"],
        "phoB": ["cra"],
        "rpoD": [
            "arcA",
            "cirA",
            "crp",
            "dcuR",
            "fis",
            "fnr",
            "fur",
            "ihfB",
            "lrp",
            "narL",
            "oxyR",
            "phoB",
            "rpoS",
            "soxS",
            "aceE",
            "ydeO",
            "hns",
        ],
        "rpoH": ["cra"],
        "rpoS": ["aceE", "appY", "hyaA", "hyaB", "hyaF", "ihfA", "ihfB", "oxyR"],
        "soxS": ["fur"],
        "ydeO": ["hyaA", "hyaF", "hyaB"],
    },
    undirected={
        "dpiA": ["dpiB"],
        "hns": ["rpoS", "lrp"],
        "rpoS": ["lrp"],
        "lrp": ["ydeO", "oxyR"],
        "oxyR": ["ydeO"],
    },
)

ecoli_transcription_example = Example(
    name="E. coli Transcriptional Regulatory Graph",
    reference="Mohammad-Taheri, S., Tewari, V., Kapre, R., Rahiminasab, E., Sachs, K., Tapley Hoyt, C.,"
    " ... & Vitek, O. (2023). Optimal adjustment sets for causal query estimation in partially"
    " observed biomolecular networks. Bioinformatics, 39(Supplement_1), i494-i503.",
    graph=graph,
    description="This is the transcriptional E. Coli regulatory network obtained from EcoCyc database. "
    "The experimental data were 260 RNA-seq normalized expression profiles of E. coli K-12"
    " MG1655 and BW25113 across 154 unique experimental conditions, extracted from the PRECISE"
    " database by (Sastry et al., 2019) from this paper: 'The Escherichia coli transcriptome mostly"
    " consists of independently regulated modules' ",
    example_queries=[Query.from_str(treatments="fur", outcomes="dpiA")],
)

ecoli_transcription_example.__doc__ = ecoli_transcription_example.description
