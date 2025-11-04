"""Unit tests for Knowledge List generator."""

import pytest
from unittest.mock import MagicMock

from synthetic_data_kit.generators.knowledge_list_generator import KnowledgeListGenerator


@pytest.mark.unit
def test_knowledge_list_generator_initialization(patch_config):
    """Test Knowledge List generator initialization."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = KnowledgeListGenerator(client=mock_client)

    # Check that the generator was initialized correctly
    assert generator.client == mock_client
    assert generator.config is not None
    assert generator.generation_config is not None


@pytest.mark.unit
def test_process_documents(patch_config):
    """Test processing documents to extract knowledge."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.batch_completion.return_value = [
        "- Key fact 1\n- Key fact 2\n- Important concept A",
        "- Key fact 3\n- Key fact 4\n- Important concept B"
    ]

    # Initialize generator
    generator = KnowledgeListGenerator(client=mock_client)

    # Process documents
    documents = [
        {"id": "doc1", "text": "This is a document with lots of information that needs to be extracted into key facts."},
        {"id": "doc2", "text": "This is another document with different information to extract."}
    ]
    
    results = generator.process_documents(documents=documents, verbose=False)

    # Check that the results contain extracted knowledge
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "- Key fact 1\n- Key fact 2\n- Important concept A"
    assert results[0]["original_length"] == len(documents[0]["text"])
    assert results[0]["knowledge_length"] == len("- Key fact 1\n- Key fact 2\n- Important concept A")
    assert "compression_ratio" in results[0]
    
    # Check that batch_completion was called
    assert mock_client.batch_completion.called


@pytest.mark.unit
def test_process_responses(patch_config):
    """Test processing raw LLM responses."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = KnowledgeListGenerator(client=mock_client)

    # Test documents and responses
    documents = [
        {"id": "doc1", "text": "Long document 1 with lots of detailed information"},
        {"id": "doc2", "text": "Long document 2 with lots of detailed information"}
    ]
    
    responses = [
        "- Fact 1\n- Fact 2",
        "- Fact 3\n- Fact 4"
    ]
    
    results = generator.process_responses(documents, responses, verbose=False)

    # Check result structure
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "- Fact 1\n- Fact 2"
    assert results[0]["compression_ratio"] < 1.0  # Should be compressed
    assert results[1]["id"] == "doc2"
    assert results[1]["text"] == "- Fact 3\n- Fact 4"


@pytest.mark.unit
def test_get_prompt_name(patch_config):
    """Test that correct prompt name is returned."""
    mock_client = MagicMock()
    generator = KnowledgeListGenerator(client=mock_client)
    
    assert generator._get_prompt_name() == "knowledge_list"

