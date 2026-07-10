"""Unit tests for embedder module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from reporag.embedding.code_embedder import CodeEmbedder


@pytest.fixture
def mock_tokenizer():
    mock = MagicMock()

    # Tokenizer output mock with a .to method
    class MockBatchEncoding(dict):
        def to(self, device):
            return self

    mock.return_value = MockBatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
    return mock


@pytest.fixture
def mock_model():
    mock = MagicMock()
    # Mock last_hidden_state where the CLS token (index 0) has a specific representation
    mock_outputs = MagicMock()
    # shape: (batch_size, sequence_length, hidden_size) = (1, 3, 768)
    mock_outputs.last_hidden_state = torch.ones((1, 3, 768)) * 2.0
    mock.return_value = mock_outputs
    mock.eval = MagicMock()
    mock.to = MagicMock(return_value=mock)
    return mock


@patch("reporag.embedding.code_embedder.AutoTokenizer.from_pretrained")
@patch("reporag.embedding.code_embedder.AutoModel.from_pretrained")
def test_code_embedder_initialization(
    mock_auto_model, mock_auto_tokenizer, mock_model, mock_tokenizer
):
    mock_auto_tokenizer.return_value = mock_tokenizer
    mock_auto_model.return_value = mock_model

    embedder = CodeEmbedder(model_name="test/model")

    assert embedder.model_name == "test/model"
    assert embedder.tokenizer == mock_tokenizer
    assert embedder.model == mock_model
    mock_model.eval.assert_called_once()
    assert embedder.device in [
        torch.device("cpu"),
        torch.device("cuda"),
        torch.device("mps"),
    ]


@patch("reporag.embedding.code_embedder.AutoTokenizer.from_pretrained")
@patch("reporag.embedding.code_embedder.AutoModel.from_pretrained")
def test_code_embedder_embed_batch(
    mock_auto_model, mock_auto_tokenizer, mock_model, mock_tokenizer
):
    mock_auto_tokenizer.return_value = mock_tokenizer
    mock_auto_model.return_value = mock_model

    embedder = CodeEmbedder()

    code_strings = ["def foo(): pass"]
    embeddings = embedder.embed_batch(code_strings)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (1, 768)
    assert embeddings.dtype == np.float32

    # Verify L2 normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    # Check if the cache was updated
    assert "def foo(): pass" in embedder._cache
    assert np.array_equal(embedder._cache["def foo(): pass"], embeddings[0])


@patch("reporag.embedding.code_embedder.AutoTokenizer.from_pretrained")
@patch("reporag.embedding.code_embedder.AutoModel.from_pretrained")
def test_code_embedder_caching(
    mock_auto_model, mock_auto_tokenizer, mock_model, mock_tokenizer
):
    mock_auto_tokenizer.return_value = mock_tokenizer
    mock_auto_model.return_value = mock_model

    embedder = CodeEmbedder()

    # First call will invoke the model
    code_strings = ["def bar(): pass"]
    _ = embedder.embed_batch(code_strings)
    call_count_1 = mock_model.call_count

    # Second call with the same string should hit the cache and not invoke the model
    _ = embedder.embed_batch(code_strings)
    call_count_2 = mock_model.call_count

    assert call_count_1 == call_count_2  # Model was not called again

    # Check cache limit
    embedder.cache_size = 1
    embedder.embed_batch(["def new(): pass"])
    assert len(embedder._cache) == 1
    assert "def new(): pass" in embedder._cache
    assert "def bar(): pass" not in embedder._cache


@patch("reporag.embedding.code_embedder.AutoTokenizer.from_pretrained")
@patch("reporag.embedding.code_embedder.AutoModel.from_pretrained")
def test_code_embedder_empty_batch(
    mock_auto_model, mock_auto_tokenizer, mock_model, mock_tokenizer
):
    mock_auto_tokenizer.return_value = mock_tokenizer
    mock_auto_model.return_value = mock_model

    embedder = CodeEmbedder()
    embeddings = embedder.embed_batch([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 768)
