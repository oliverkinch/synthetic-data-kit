# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate the content: CoT/QA/Summary Datasets
import os
import json
from pathlib import Path
from typing import Optional

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator

from synthetic_data_kit.generators.distill_generator import DistillGenerator
from synthetic_data_kit.generators.knowledge_list_generator import KnowledgeListGenerator
from synthetic_data_kit.generators.extract_knowledge_generator import ExtractKnowledgeGenerator
from synthetic_data_kit.generators.wikipedia_rephrase_generator import WikipediaRephraseGenerator

from synthetic_data_kit.utils.lance_utils import load_lance_dataset

def read_json(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    return document_text


def process_file(
    file_path: str,
    output_dir: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: str = "qa",
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> str:
    """Process a file to generate content
    
    Args:
        file_path: Path to the text file to process
        output_dir: Directory to save generated content
        config_path: Path to configuration file
        api_base: VLLM API base URL
        model: Model to use
        content_type: Type of content to generate (qa, distill, knowledge-list, extract-knowledge, wikipedia-rephrase)
        num_pairs: Target number of QA pairs to generate
        threshold: Quality threshold for filtering (1-10)
    
    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    # The reason for having this directory logic for now is explained in context.py
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LLM client
    client = LLMClient(
        config_path=config_path,
        provider=provider,
        api_base=api_base,
        model_name=model
    )
    
    # Override chunking config if provided
    if chunk_size is not None:
        client.config.setdefault('generation', {})['chunk_size'] = chunk_size
    if chunk_overlap is not None:
        client.config.setdefault('generation', {})['overlap'] = chunk_overlap
    
    # Debug: Print which provider is being used
    print(f"L Using {client.provider} provider")
    
    # Generate base filename for output
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Generate content based on type
    if file_path.endswith(".lance"):
        dataset = load_lance_dataset(file_path)
        documents = dataset.to_table().to_pylist()
    else:
        documents = [{"text": read_json(file_path), "image": None}]

    if content_type == "qa":
        generator = QAGenerator(client=client, config_path=config_path)
        
        # Process document
        result = generator.process_documents(
            documents=documents,
            verbose=verbose,
        )

        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
        print(f"Saving result to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        return output_path

    elif content_type == "distill":
        generator = DistillGenerator(client=client, config_path=config_path)
        
        # Process documents
        result = generator.process_documents(
            documents=documents,
            verbose=verbose
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_distilled.json")
        print(f"Saving result to {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_path

    elif content_type == "knowledge-list":
        generator = KnowledgeListGenerator(client=client, config_path=config_path)
        
        # Process documents
        result = generator.process_documents(
            documents,
            verbose=verbose
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_knowledge.json")
        print(f"Saving result to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_path

    elif content_type == "extract-knowledge":
        generator = ExtractKnowledgeGenerator(client=client, config_path=config_path)
        
        # Process documents
        result = generator.process_documents(
            documents=documents,
            verbose=verbose
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_extracted_knowledge.json")
        print(f"Saving result to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_path

    elif content_type == "wikipedia-rephrase":
        generator = WikipediaRephraseGenerator(client=client, config_path=config_path)
        
        # Process documents
        result = generator.process_documents(
            documents=documents,
            verbose=verbose
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_wikipedia_rephrased.json")
        print(f"Saving result to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_path
    else:
        raise ValueError(f"Unknown content type: {content_type}")