"""Unit tests for QA generator."""

import pytest
from unittest.mock import MagicMock

from synthetic_data_kit.generators.qa_generator import QAGenerator


@pytest.mark.unit
def test_qa_generator_initialization(patch_config):
    """Test QA generator initialization."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Check that the generator was initialized correctly
    assert generator.client == mock_client
    assert generator.config is not None
    assert generator.generation_config is not None


@pytest.mark.unit
def test_process_documents(patch_config):
    """Test processing documents to generate QA pairs."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.batch_completion.return_value = [
        """Here are the questions and answers based on the provided text:
- Question: What is synthetic data? Answer: Synthetic data is artificially generated data.
- Question: Why use it? Answer: To protect privacy.""",
        """Here are the questions and answers based on the provided text:
- Question: What is machine learning? Answer: A method of data analysis.
- Question: How does it work? Answer: By learning from patterns."""
    ]

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Process documents
    documents = [
        {"id": "doc1", "text": "This is document 1 about synthetic data."},
        {"id": "doc2", "text": "This is document 2 about machine learning."}
    ]
    
    result = generator.process_documents(documents=documents, verbose=False)

    # Check that the result contains QA pairs
    assert "qa_pairs" in result
    assert len(result["qa_pairs"]) == 4  # 2 pairs from each document
    
    # Check that document IDs are attached
    assert all("id" in pair for pair in result["qa_pairs"])
    assert result["qa_pairs"][0]["id"] == "doc1"
    assert result["qa_pairs"][2]["id"] == "doc2"
    
    # Check that batch_completion was called
    assert mock_client.batch_completion.called


@pytest.mark.unit
def test_process_responses(patch_config):
    """Test processing raw LLM responses."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Test documents and responses (using the format that parse_qa_pairs expects)
    documents = [
        {"id": "doc1", "text": "Document 1"},
        {"id": "doc2", "text": "Document 2"}
    ]
    
    responses = [
        "- Question: What is Q1? Answer: This is A1",
        "- Question: What is Q2? Answer: This is A2"
    ]
    
    result = generator.process_responses(documents, responses, verbose=False)

    # Check result structure
    assert "qa_pairs" in result
    assert len(result["qa_pairs"]) == 2
    assert result["qa_pairs"][0]["id"] == "doc1"
    assert result["qa_pairs"][1]["id"] == "doc2"
