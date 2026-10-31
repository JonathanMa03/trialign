"""Model specifications; selection remains an experiment decision."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    query_prefix: str = ""
    notes: str = ""


CANDIDATE_MODELS = {
    "minilm": ModelSpec(
        "sentence-transformers/all-MiniLM-L6-v2", "general", notes="Fast general baseline"
    ),
    "mpnet": ModelSpec(
        "sentence-transformers/all-mpnet-base-v2", "general", notes="Stronger general baseline"
    ),
    "bge": ModelSpec(
        "BAAI/bge-base-en-v1.5",
        "retrieval",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "biolord": ModelSpec("FremyCompany/BioLORD-2023", "biomedical"),
    "sapbert": ModelSpec(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "biomedical_concept",
        notes="Entity-oriented comparator; do not assume sentence-level suitability",
    ),
}

