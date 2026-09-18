"""The SchemaNID schema-aware attention architecture."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .preprocessing import CONCEPTS, SchemaRegistry


class SchemaNID(nn.Module):
    """Encode heterogeneous flow schemas with shared self-attention.

    Each token combines a transformed value, presence/type metadata, a broad
    network-concept embedding, and a deterministic feature-name fingerprint.
    The ablation flags expose the variants reported in the paper.
    """

    def __init__(
        self,
        registry: SchemaRegistry,
        model_dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.10,
        *,
        use_name_fingerprints: bool = True,
        use_concept_embeddings: bool = True,
        use_presence_type: bool = True,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")

        self.use_name_fingerprints = use_name_fingerprints
        self.use_concept_embeddings = use_concept_embeddings
        self.use_presence_type = use_presence_type

        self.value_projection = nn.Linear(4, model_dim)
        self.name_projection = nn.Linear(16, model_dim, bias=False)
        self.concept_embedding = nn.Embedding(len(CONCEPTS) + 1, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.pool_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

        self.register_buffer(
            "concept_ids",
            torch.as_tensor(registry.concept_ids + 1, dtype=torch.long),
        )
        self.register_buffer(
            "name_fingerprints",
            torch.as_tensor(registry.name_fingerprints, dtype=torch.float32),
        )
        self.register_buffer(
            "type_flags",
            torch.as_tensor(registry.type_flags, dtype=torch.float32),
        )

    def reset_parameters(self) -> None:
        """Reinitialize trainable weights for a seeded paper run."""

        self.value_projection.reset_parameters()
        self.name_projection.reset_parameters()
        self.concept_embedding.reset_parameters()

        first_layer = self.encoder.layers[0]
        first_layer.self_attn.out_proj.reset_parameters()
        first_layer.self_attn._reset_parameters()
        first_layer.linear1.reset_parameters()
        first_layer.linear2.reset_parameters()
        first_layer.norm1.reset_parameters()
        first_layer.norm2.reset_parameters()
        for layer in self.encoder.layers[1:]:
            layer.load_state_dict(first_layer.state_dict())

        self.pool_norm.reset_parameters()
        for layer in self.classifier:
            reset = getattr(layer, "reset_parameters", None)
            if reset is not None:
                reset()

    def forward(
        self,
        values: torch.Tensor,
        present: torch.Tensor,
        schema_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return binary logits and the pooled flow representation."""

        types = self.type_flags[schema_ids]
        if self.use_presence_type:
            metadata_presence = present
            metadata_types = types
        else:
            metadata_presence = torch.zeros_like(present)
            metadata_types = torch.zeros_like(types)

        base = torch.cat(
            (values.unsqueeze(-1), metadata_presence.unsqueeze(-1), metadata_types),
            dim=-1,
        )
        tokens = self.value_projection(base)
        if self.use_concept_embeddings:
            tokens = tokens + self.concept_embedding(self.concept_ids[schema_ids])
        if self.use_name_fingerprints:
            tokens = tokens + self.name_projection(
                self.name_fingerprints[schema_ids]
            )

        available = present > 0
        attention_available = available.clone()
        empty_rows = ~attention_available.any(dim=1)
        if empty_rows.any():
            attention_available[empty_rows, 0] = True

        encoded = self.encoder(
            tokens, src_key_padding_mask=~attention_available
        )
        weights = available.float().unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        representation = self.pool_norm(pooled)
        logits = self.classifier(representation).squeeze(-1)
        return {"logits": logits, "representation": representation}
