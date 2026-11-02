import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from typer.testing import CliRunner

import trialign.cli as cli
import trialign.models.dense as dense_module


def test_dense_cli_writes_outputs(monkeypatch, tmp_path):
    records = [
        SimpleNamespace(
            criterion_id="a",
            trial_id="trial1",
            comparison_text="age eighteen",
            source="Age at least eighteen",
        ),
        SimpleNamespace(
            criterion_id="b",
            trial_id="trial2",
            comparison_text="age twenty one",
            source="Age at least twenty one",
        ),
    ]

    class FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            assert texts == ["age eighteen", "age twenty one"]
            return np.array(
                [[1.0, 0.0], [0.8, 0.6]],
                dtype=np.float32,
            )

    monkeypatch.setattr(cli, "load_parquet_corpus", lambda path: records)
    monkeypatch.setattr(dense_module, "DenseEncoder", FakeEncoder)

    corpus = tmp_path / "provisional.parquet"
    manifest = corpus.with_suffix(".parquet.manifest.json")
    manifest.write_text(
        json.dumps({
            "development_only": True,
            "human_review_complete": False,
            "source_sha256": "synthetic-test",
        }),
        encoding="utf-8",
    )

    output = tmp_path / "results"
    result = CliRunner().invoke(
        cli.app,
        [
            "baseline-dense",
            "--corpus", str(corpus),
            "--output-dir", str(output),
            "--model-name", "fake-model",
            "--device", "cpu",
            "--top-k", "1",
        ],
    )

    assert result.exit_code == 0, result.output

    neighbors = pd.read_parquet(output / "neighbors.parquet")
    embeddings = np.load(output / "embeddings.npy")
    metadata = json.loads((output / "run.json").read_text())

    assert list(
        zip(neighbors["query_id"], neighbors["candidate_id"], strict=True)
    ) == [
        ("a", "b"),
        ("b", "a"),
    ]
    assert embeddings.shape == (2, 2)
    assert metadata["model"] == "fake-model"
    assert metadata["development_only"] is True
    assert metadata["human_review_complete"] is False