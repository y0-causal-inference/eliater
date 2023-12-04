"""Examples for transcriptional Escherichia coli K-12 regulatory network."""

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
    description="This is the transcriptional E. Coli regulatory network"
    " obtained from EcoCyc database ",
    example_queries=[Query.from_str(treatments="fur", outcomes="dpiA")],
)

ecoli_transcription_example.__doc__ = ecoli_transcription_example.description
