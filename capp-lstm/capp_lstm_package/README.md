# CAPP LSTM Package
This package provides an LSTM-based neural network architecture for Computer-Aided Process Planning (CAPP) in manufacturing process generation.

The system receives a JSON-formatted input file that describes the part to be manufactured and produces up to four equivalent process chains as output using a LSTM-based network architecture with multi-head output.

The input JSON file must describe the part using the following classes:
* Geometry: Describes the overall shape and complexity of the part (e.g., prismatic, freeform).
* Holes: Describes the type and size of holes in the part (e.g., none, normal, large threaded).
* External Threads: Indicates whether the part has external threading (yes/no).
* Surface Finish: Specifies the required surface finish quality (rough to very good).
* Tolerance: Describes the part's dimensional tolerance (from rough to tight).
* Batch Size: Indicates the quantity to be manufactured (e.g., prototype, mass production).

Specifically:
```JSON
{
    "part_encoding": {
        "geometry": [
            "pure_axisymmetric",
            "axisymmetric_with_prismatic_features",
            "prismatic",
            "prismatic_with_axisymmetric_features",
            "prismatic_with_freeform_features",
            "freeform",
            "unconventional"
        ],
        "holes": [
            "none",
            "normal",
            "normal_threaded",
            "normal_functional",
            "large",
            "large_threaded",
            "large_functional"
        ],
        "external_threads": [
            "yes",
            "no"
        ],
        "surface_finish": [
            "rough",
            "normal",
            "good",
            "very_good"
        ],
        "tolerance": [
            "rough",
            "normal",
            "medium",
            "tight"
        ],
        "batch_size": [
            "prototype",
            "small",
            "medium",
            "large",
            "mass"
        ]
    }
}
```

The system can choose from the following manufacturing processes for each output process chain. Each process in the list may be selected only once per process chain.
The generated process chains may differ but will all represent valid sequences for producing the part described by the input.

```JSON
{
    "process_list": [
        "Turning",
        "Milling",
        "5-axis Milling",
        "SLM",
        "Sand Casting",
        "High Pressure Die Casting",
        "Investment Casting",
        "Turning Secondary",
        "Milling Secondary",
        "Hole Milling",
        "5-axis Milling Secondary",
        "Thread Milling",
        "Tapping",
        "Grinding",
        "5-axis Grinding",
        "Superfinishing",
        "Drilling",
        "Boring",
        "Reaming",
        "Special Finishing"
    ]
}
```