"""Basic checks that the public code runs. Run with: python -m pytest"""

import numpy as np
import pandas as pd
import torch

from schemanid import (
    SchemaNID,
    TrainingConfig,
    binary_labels,
    build_registry,
    deterministic_partitions,
    train,
)
from schemanid.evaluation import binary_metrics, predict_probabilities
from schemanid.preprocessing import concept_id, name_fingerprint

NAMES = ["Flow Duration", "Total Fwd Packets", "SYN Flag Count", "Protocol"]


def _flows(rows=64):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.exponential(5.0, (rows, 4)), columns=NAMES)
    frame["Protocol"] = rng.choice(["tcp", "udp", "icmp"], rows)
    return frame


def test_forward_shapes_and_ablations():
    registry = build_registry({"a": NAMES})
    values, present = registry.transform(_flows(8), "a")
    schema = torch.zeros(8, dtype=torch.long)
    for flags in (
        {},
        {"use_name_fingerprints": False},
        {"use_concept_embeddings": False},
        {"use_presence_type": False},
    ):
        out = SchemaNID(registry, **flags)(
            torch.from_numpy(values), torch.from_numpy(present), schema
        )
        assert out["logits"].shape == (8,)
        assert out["representation"].shape == (8, 64)


def test_row_with_no_features_gives_finite_output():
    registry = build_registry({"a": NAMES})
    values, present = registry.transform(_flows(3), "a")
    present[0] = 0
    out = SchemaNID(registry)(
        torch.from_numpy(values),
        torch.from_numpy(present),
        torch.zeros(3, dtype=torch.long),
    )
    assert torch.isfinite(out["logits"]).all()


def test_preprocessing_is_fixed():
    assert concept_id("Flow Duration") == 0
    assert concept_id("zzz") == -1
    assert np.array_equal(name_fingerprint("Flow Duration"), name_fingerprint("flow_duration"))
    registry = build_registry({"a": NAMES})
    values, present = registry.transform(_flows(5), "a")
    assert values.shape == (5, 128) and present.sum() == 5 * 4
    assert np.abs(values).max() <= 1.0


def test_binary_labels():
    assert binary_labels(pd.Series(["BENIGN", "DoS", "Normal"])).tolist() == [0, 1, 0]
    assert binary_labels(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]


def test_partitions_are_deterministic_and_disjoint():
    labels = (np.random.default_rng(1).random(160_000) < 0.2).astype(np.int64)
    a = deterministic_partitions(labels, 9001)
    b = deterministic_partitions(labels, 9001)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert (len(a["train"]), len(a["validation"]), len(a["confirmation"])) == (96_000, 24_000, 30_000)
    assert not set(a["train"]) & set(a["confirmation"])
    assert not set(a["validation"]) & set(a["confirmation"])


def test_training_and_evaluation_run():
    frame = _flows(400)
    labels = (frame["Flow Duration"] > 5).astype(np.int64).to_numpy()
    registry = build_registry({"a": NAMES})
    values, present = registry.transform(frame, "a")
    schema = np.zeros(len(frame), dtype=np.int64)
    split = 300
    result = train(
        SchemaNID(registry),
        (values[:split], present[:split], labels[:split], schema[:split]),
        (values[split:], present[split:], labels[split:], schema[split:]),
        TrainingConfig(epochs=2, patience=1, device="cpu"),
    )
    probabilities = predict_probabilities(
        result["model"], values[split:], present[split:], schema[split:]
    )
    metrics = binary_metrics(labels[split:], probabilities, result["threshold"])
    assert 0.0 <= metrics["macro_f1"] <= 1.0
