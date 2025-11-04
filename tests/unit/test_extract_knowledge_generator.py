# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pytest
from unittest.mock import MagicMock, patch

from synthetic_data_kit.generators.extract_knowledge_generator import ExtractKnowledgeGenerator


# Patch config loading at module level
@pytest.fixture(autouse=True)
def patch_config():
    with patch('synthetic_data_kit.utils.config.load_config') as mock_load:
        mock_load.return_value = {
            'generation': {
                'batch_size': 32,
                'temperature': 0.7,
                'single_call_max_size': 8000
            },
            'prompts': {
                'extract_knowledge': {
                    'system': 'You are an expert at extracting knowledge.',
                    'user': 'Extract knowledge from:\n{text}'
                }
            }
        }
        yield mock_load


@pytest.mark.unit
def test_extract_knowledge_generator_initialization(patch_config):
    """Test that ExtractKnowledgeGenerator initializes correctly."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = ExtractKnowledgeGenerator(client=mock_client)

    # Check that generator has the expected attributes
    assert generator.client == mock_client
    assert generator.config is not None
    assert generator.generation_config is not None


@pytest.mark.unit
def test_process_documents(patch_config):
    """Test processing documents to extract knowledge."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.batch_completion.return_value = [
        "This is a rewritten passage that clearly explains the key concepts from document 1.",
        "This is a rewritten passage that clearly explains the key concepts from document 2."
    ]

    # Initialize generator
    generator = ExtractKnowledgeGenerator(client=mock_client)

    # Process documents
    documents = [
        {"id": "doc1", "text": "This is a document with complex information that needs to be rewritten clearly."},
        {"id": "doc2", "text": "This is another document with detailed knowledge to extract and rewrite."}
    ]
    
    results = generator.process_documents(documents=documents, verbose=False)

    # Check that the results contain extracted knowledge
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "This is a rewritten passage that clearly explains the key concepts from document 1."
    assert results[1]["id"] == "doc2"
    assert results[1]["text"] == "This is a rewritten passage that clearly explains the key concepts from document 2."
    
    # Check that batch_completion was called
    assert mock_client.batch_completion.called


@pytest.mark.unit
def test_process_responses(patch_config):
    """Test processing raw LLM responses."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = ExtractKnowledgeGenerator(client=mock_client)

    # Test documents and responses
    documents = [
        {"id": "doc1", "text": "Original text 1"},
        {"id": "doc2", "text": "Original text 2"}
    ]

    responses = [
        "Rewritten knowledge passage 1",
        "Rewritten knowledge passage 2"
    ]

    results = generator.process_responses(documents, responses, verbose=False)

    # Check result structure
    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["text"] == "Rewritten knowledge passage 1"
    assert results[1]["id"] == "doc2"
    assert results[1]["text"] == "Rewritten knowledge passage 2"


@pytest.mark.unit
def test_get_prompt_name(patch_config):
    """Test that the generator returns the correct prompt name."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = ExtractKnowledgeGenerator(client=mock_client)

    # Check prompt name
    assert generator._get_prompt_name() == "extract_knowledge"

