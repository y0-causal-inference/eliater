"""Examples for E. coli K-12.

.. todo::

    1. wrap these with the :class:`y0.example.Example` class and detailed biological context.
    2. Where did this network come from? Give reference
    3. What is E. coli K-12?
    4. What is the biological phenomena described here?
    5. How was this network constructed?
    6. Is there associated data to go with this graph?
"""

from y0.graph import NxMixedGraph

graph = NxMixedGraph.from_str_adj(
    directed={
        "appY": ["appA", "appB", "appX", "hyaA", "hyaB", "hyaF"],
        "arcA": [
            "rpoS",
            "fnr",
            "dpiA",
            "aceE",
            "appY",
            "citX",
            "cydD",
            "dpiB",
            "gcvB",
            "hyaA",
            "hyaB",
            "hyaF",
            "mdh",
        ],
        "btsR": ["mdh"],
        "chiX": ["dpiA", "dpiB"],
        "citX": ["dpiB"],
        "cra": ["cyoA"],
        "crp": [
            "dpiA",
            "cirA",
            "dcuR",
            "oxyR",
            "fis",
            "fur",
            "aceE",
            "citX",
            "cyoA",
            "dpiB",
            "exuT",
            "gadX",
            "mdh",
            "srIR",
        ],
        "cspA": ["hns"],
        "dcuR": ["dpiA", "dpiB"],
        "dpiA": ["appY", "citC", "citD", "citX", "dpiB", "exuT", "mdh"],
        "dsrA": ["hns", "lrp", "rpoS"],
        "fis": ["cyoA", "gadX", "hns", "hyaA", "hyaB", "hyaF"],
        "fnr": [
            "dcuR",
            "dpiA",
            "narL",
            "aceE",
            "amtB",
            "aspC",
            "citX",
            "cydD",
            "cyoA",
            "dpiB",
            "gadX",
            "hcp",
        ],
        "fur": ["fnr", "amtB", "aspC", "cirA", "cyoA"],
        "gadX": ["amtB", "hns"],
        "gcvB": ["lrp", "oxyR", "ydeO"],
        "hns": ["appY", "srIR", "ydeO", "yjjQ"],
        "ihfA": ["crp", "fnr", "ihfB"],
        "ihfB": ["fnr"],
        "iscR": ["hyaA", "hyaB", "appX"],
        "lrp": ["soxS", "aspC"],
        "modE": ["narL"],
        "narL": ["citX", "cydD", "dpiB", "hcp", "hyaA", "hyaB", "hyaF", "dcuR", "dpiA"],
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
            "yjjQ",
        ],
        "rpoH": ["cra"],
        "rpoS": ["aceE", "appY", "hyaA", "hyaB", "hyaF", "ihfA", "ihfB", "oxyR"],
        "soxS": ["fur"],
        "srIR": ["gutM"],
        "ydeO": ["hyaA", "hyaF", "hyaB"],
    }
)
