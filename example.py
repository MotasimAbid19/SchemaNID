"""Minimal forward-pass example for the SchemaNID architecture."""

import numpy as np
import pandas as pd
import torch

from schemanid import SchemaNID, build_registry


schemas = {
    "cic": ["Flow Duration", "Total Fwd Packets", "SYN Flag Count", "Protocol"],
    "unsw": ["dur", "spkts", "dbytes", "proto"],
}
registry = build_registry(schemas)

flows = pd.DataFrame(
    {
        "Flow Duration": [1.25, 0.08],
        "Total Fwd Packets": [12, 420],
        "SYN Flag Count": [1, 18],
        "Protocol": ["tcp", "udp"],
    }
)
values, present = registry.transform(flows, "cic")

model = SchemaNID(registry)
with torch.no_grad():
    output = model(
        torch.from_numpy(values),
        torch.from_numpy(present),
        torch.from_numpy(np.zeros(len(flows), dtype=np.int64)),
    )

print("logits shape:", tuple(output["logits"].shape))
print("flow representation shape:", tuple(output["representation"].shape))
print("parameters:", sum(parameter.numel() for parameter in model.parameters()))
print("This is an architecture check; train the model before using its predictions.")
