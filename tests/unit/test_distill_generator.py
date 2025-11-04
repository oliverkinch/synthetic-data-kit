"""Unit tests for Distill generator."""

import pytest
from unittest.mock import MagicMock

from synthetic_data_kit.generators.distill_generator import DistillGenerator


@pytest.mark.unit
def test_distill_generator_initialization(patch_config):
    """Test Distill generator initialization."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = DistillGenerator(client=mock_client)

    # Check that the generator was initialized correctly
    assert generator.client == mock_client
    assert generator.config is not None
    assert generator.generation_config is not None


@pytest.mark.unit
def test_process_documents(patch_config):
    """Test processing documents to distill text."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.batch_completion.return_value = [
        "This is the distilled version of document 1.",
        "This is the distilled version of document 2."
    ]

    # Initialize generator
    generator = DistillGenerator(client=mock_client)

    # Process documents
    documents = [
        {"id": "doc1", "text": "This is a very long document 1 that needs to be distilled into a shorter version."},
        {"id": "doc2", "text": "This is a very long document 2 that needs to be distilled into a shorter version."}
    ]
    
    results = generator.process_documents(documents=documents, verbose=False)

    # Check that the results contain distilled text
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "This is the distilled version of document 1."
    assert results[0]["original_length"] == len(documents[0]["text"])
    assert results[0]["distilled_length"] == len("This is the distilled version of document 1.")
    assert "compression_ratio" in results[0]
    
    # Check that batch_completion was called
    assert mock_client.batch_completion.called


@pytest.mark.unit
def test_process_responses(patch_config):
    """Test processing raw LLM responses."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = DistillGenerator(client=mock_client)

    # Test documents and responses
    documents = [
        {"id": "doc1", "text": "Long document 1 with lots of text"},
        {"id": "doc2", "text": "Long document 2 with lots of text"}
    ]
    
    responses = [
        "Short distilled 1",
        "Short distilled 2"
    ]
    
    results = generator.process_responses(documents, responses, verbose=False)

    # Check result structure
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "Short distilled 1"
    assert results[0]["compression_ratio"] < 1.0  # Should be compressed
    assert results[1]["id"] == "doc2"
    assert results[1]["text"] == "Short distilled 2"


@pytest.mark.unit
def test_get_prompt_name(patch_config):
    """Test that correct prompt name is returned."""
    mock_client = MagicMock()
    generator = DistillGenerator(client=mock_client)
    
    assert generator._get_prompt_name() == "distill"

