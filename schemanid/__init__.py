"""Public SchemaNID architecture and preprocessing API."""

from .model import SchemaNID
from .preprocessing import (
    SchemaRegistry,
    binary_labels,
    build_registry,
    deterministic_partitions,
    transform_frame,
)
from .training import TrainingConfig, train

__all__ = [
    "SchemaNID",
    "SchemaRegistry",
    "TrainingConfig",
    "binary_labels",
    "build_registry",
    "deterministic_partitions",
    "train",
    "transform_frame",
]
