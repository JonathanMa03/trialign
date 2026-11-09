"""Lazy-loaded sentence embeddings for dense retrieval."""

import numpy as np


class DenseEncoder:
    def __init__(
        self,
        model_name,
        *,
        device="cpu",
        revision=None,
        batch_size=32,
        model_factory=None,
        max_seq_length=None,
        cache_folder=None,
        local_files_only=False,
    ):
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("Unsupported device.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self.name = model_name
        self.device = device
        self.revision = revision
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._model = None
        self.max_seq_length = max_seq_length
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only

    @property
    def effective_device(self):
        return str(self._model.device) if self._model is not None else self.device

    def _load_model(self):
        if self._model is None:
            factory = self._model_factory
            if factory is None:
                from sentence_transformers import SentenceTransformer

                factory = SentenceTransformer

            options = {"revision": self.revision}
            if self.cache_folder is not None:
                options["cache_folder"] = str(self.cache_folder)
            if self.local_files_only:
                options["local_files_only"] = True
            if self.device != "auto":
                options["device"] = self.device

            self._model = factory(self.name, **options)
            if self.max_seq_length is not None:
                native = self._model.max_seq_length
                if native and self.max_seq_length > native:
                    self._model = None
                    raise ValueError(
                        f"Requested token length exceeds this encoder's limit ({native})."
                    )
                self._model.max_seq_length = self.max_seq_length

        return self._model

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            raise ValueError("Cannot encode an empty collection.")

        model = self._load_model()
        vectors = np.asarray(
            model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("Model returned an unexpected embedding shape.")
        if not np.isfinite(vectors).all():
            raise ValueError("Model returned non-finite embeddings.")

        # Enforce normalization at our interface boundary.
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(lengths == 0):
            raise ValueError("Model returned a zero embedding.")

        return vectors / lengths
