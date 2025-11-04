# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

from typing import Dict, List, Any, Optional
from pathlib import Path

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.base import BaseGenerator
from synthetic_data_kit.utils.llm_processing import parse_qa_pairs
from synthetic_data_kit.utils.config import load_config, get_generation_config

class QAGenerator(BaseGenerator):
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the QA Generator with an LLM client and optional config"""
        super().__init__(client, config_path)
    
    def process_responses(self, 
                         documents: List[Dict[str, Any]], 
                         responses: List[str],
                         verbose: bool = False) -> Dict[str, Any]:
        """Process QA generation responses into structured results
        
        Args:
            documents: Original input documents
            responses: Raw LLM responses (QA pairs text)
            verbose: Whether to show detailed output
            
        Returns:
            Dict with 'qa_pairs' list containing parsed QA pairs
        """
        all_qa_pairs = []
        doc_qa_counts = {}  # Track QA pairs per document
        
        for doc_idx, response in enumerate(responses):
            doc_id = documents[doc_idx]["id"]
            qa_pairs = parse_qa_pairs(response)
            
            if qa_pairs:
                # Attach source document id to each QA pair
                for qa_pair in qa_pairs:
                    qa_pair["id"] = doc_id
                all_qa_pairs.extend(qa_pairs)
                
                # Track count
                doc_qa_counts[doc_id] = len(qa_pairs)
        
        if verbose:
            print(f"\n✅ Generated QA pairs per document:")
            for doc_id, count in doc_qa_counts.items():
                print(f"  {doc_id}: {count} QA pairs")
        
        print(f"✅ Total: {len(all_qa_pairs)} QA pairs generated from {len(documents)} documents")

        result = {
            "qa_pairs": all_qa_pairs
        }

        return result

    def _get_prompt_name(self) -> str:
        """Return the name of the prompt to use from config"""
        return "qa_generation"