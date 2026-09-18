"""Schema-aware preprocessing used by SchemaNID.

The transformations are fixed: they do not fit means, variances, quantiles, or
other statistics on a target dataset. This is important for source-only
cross-domain evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CONCEPTS = (
    "duration",
    "forward_volume",
    "backward_volume",
    "packet_length",
    "timing",
    "rate",
    "tcp_flags",
    "header",
    "protocol_service",
    "activity",
)

CONCEPT_PATTERNS = {
    "duration": ("duration", "dur"),
    "forward_volume": ("fwd", "forward", "spkts", "sbytes", "src"),
    "backward_volume": ("bwd", "backward", "dpkts", "dbytes", "dst"),
    "packet_length": (
        "packet length",
        "pkt len",
        "mean",
        "std",
        "min",
        "max",
        "bytes",
    ),
    "timing": ("iat", "jitter", "time"),
    "rate": ("/s", "rate", "load"),
    "tcp_flags": ("flag", "syn", "ack", "fin", "rst", "psh", "urg"),
    "header": ("header", "hdr"),
    "protocol_service": ("protocol", "proto", "service", "state"),
    "activity": ("active", "idle"),
}

PROTOCOL_VALUES = {
    "icmp": 1.0,
    "tcp": 2.0,
    "udp": 3.0,
    "arp": 4.0,
    "igmp": 5.0,
}

CATEGORICAL_NAME_PARTS = ("protocol", "proto", "service", "state")


@dataclass(frozen=True)
class SchemaRegistry:
    """Fixed schema metadata for one or more datasets."""

    domains: tuple[str, ...]
    columns: tuple[tuple[str, ...], ...]
    concept_ids: np.ndarray
    name_fingerprints: np.ndarray
    type_flags: np.ndarray

    def domain_index(self, name: str) -> int:
        return self.domains.index(name)

    @property
    def max_features(self) -> int:
        return int(self.concept_ids.shape[1])

    def transform(
        self, frame: pd.DataFrame, domain: str
    ) -> tuple[np.ndarray, np.ndarray]:
        index = self.domain_index(domain)
        return transform_frame(frame, self.columns[index], self.max_features)


def normalize_feature_name(name: str) -> str:
    """Normalize a column name before creating its deterministic identity."""

    return re.sub(r"[^a-z0-9]+", "_", name.lower())


def concept_id(name: str) -> int:
    """Map a feature name to one of ten broad network concepts."""

    lowered = name.lower()
    for index, concept in enumerate(CONCEPTS):
        if any(pattern in lowered for pattern in CONCEPT_PATTERNS[concept]):
            return index
    return -1


def name_fingerprint(name: str, width: int = 16) -> np.ndarray:
    """Return the paper's stable signed-bit SHA-256 feature identity."""

    normalized = normalize_feature_name(name)
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:width]
    return bits.astype(np.float32) * 2.0 - 1.0


def is_categorical(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in CATEGORICAL_NAME_PARTS)


def signed_log_transform(series: pd.Series) -> np.ndarray:
    """Convert a raw feature to the fixed range used in the paper."""

    numeric = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    lowered = series.astype(str).str.lower()
    for token, value in PROTOCOL_VALUES.items():
        selected = numeric.isna() & lowered.str.contains(token, regex=False, na=False)
        numeric.loc[selected] = value

    values = numeric.to_numpy(dtype=np.float64)
    values = np.sign(values) * np.log1p(np.minimum(np.abs(values), 1e30))
    return np.clip(values / 20.0, -1.0, 1.0).astype(np.float32)


def build_registry(
    schemas: Mapping[str, Sequence[str]], max_features: int = 128
) -> SchemaRegistry:
    """Build fixed concept, name, and type metadata for each dataset schema."""

    if not schemas:
        raise ValueError("At least one dataset schema is required")
    domains = tuple(schemas)
    columns = tuple(tuple(schemas[name][:max_features]) for name in domains)
    concept_ids = np.full((len(domains), max_features), -1, dtype=np.int64)
    fingerprints = np.zeros((len(domains), max_features, 16), dtype=np.float32)
    type_flags = np.zeros((len(domains), max_features, 2), dtype=np.float32)

    for domain_index, names in enumerate(columns):
        for feature_index, name in enumerate(names):
            concept_ids[domain_index, feature_index] = concept_id(name)
            fingerprints[domain_index, feature_index] = name_fingerprint(name)
            type_flags[domain_index, feature_index] = (
                (0.0, 1.0) if is_categorical(name) else (1.0, 0.0)
            )

    return SchemaRegistry(
        domains=domains,
        columns=columns,
        concept_ids=concept_ids,
        name_fingerprints=fingerprints,
        type_flags=type_flags,
    )


def transform_frame(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    max_features: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-width value and presence arrays for a flow table."""

    values = np.zeros((len(frame), max_features), dtype=np.float32)
    present = np.zeros_like(values)
    for index, name in enumerate(feature_names[:max_features]):
        if name not in frame:
            continue
        transformed = signed_log_transform(frame[name])
        observed = np.isfinite(transformed)
        values[:, index] = np.where(observed, transformed, 0.0)
        present[:, index] = observed.astype(np.float32)
    return values, present


def binary_labels(series: pd.Series) -> np.ndarray:
    """Map benign/normal labels to 0 and attack labels to 1."""

    if series.isna().any():
        raise ValueError("Labels must not contain missing values")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return (numeric.to_numpy(dtype=np.float64) != 0).astype(np.int64)
    normalized = series.astype(str).str.strip().str.lower()
    if normalized.eq("").any():
        raise ValueError("Labels must not contain empty values")
    benign = normalized.isin({"benign", "normal", "normal.", "0"})
    return (~benign).to_numpy(dtype=np.int64)


def deterministic_partitions(
    labels: np.ndarray,
    sampling_seed: int,
    *,
    development_rows: int = 120_000,
    confirmation_rows: int = 30_000,
    validation_fraction: float = 0.20,
    validation_seed: int = 2026,
) -> dict[str, np.ndarray]:
    """Reproduce the paper's priority sample and development split.

    The lowest random priorities form a bounded sample. Confirmation rows are
    separated before the development rows are stratified into train and
    validation partitions. Large readers can apply the same priority rule in
    chunks without keeping every flow record in memory.
    """

    labels = np.asarray(labels, dtype=np.int64)
    sample_rows = development_rows + confirmation_rows
    if len(labels) < sample_rows:
        raise ValueError(f"Need at least {sample_rows:,} rows, found {len(labels):,}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes")

    priorities = np.random.default_rng(sampling_seed).random(
        len(labels), dtype=np.float32
    )
    selected = np.argpartition(priorities, sample_rows - 1)[:sample_rows]
    order = np.lexsort((selected, priorities[selected]))
    selected = selected[order]
    development = selected[:development_rows]
    confirmation = selected[development_rows:]
    train, validation = train_test_split(
        development,
        test_size=validation_fraction,
        random_state=validation_seed,
        stratify=labels[development],
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "confirmation": np.sort(confirmation),
    }
