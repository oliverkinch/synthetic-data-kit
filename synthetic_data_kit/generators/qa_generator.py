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
            Dict with document IDs as keys, each containing 'original_text' and 'qa_pairs'
        """
        result = {}
        total_qa_pairs = 0
        
        for doc_idx, response in enumerate(responses):
            doc_id = documents[doc_idx]["id"]
            original_text = documents[doc_idx]["text"]
            qa_pairs = parse_qa_pairs(text=response)
            
            # Create entry for this document
            result[doc_id] = {
                "original_text": original_text,
                "qa_pairs": qa_pairs
            }
            
            if qa_pairs:
                total_qa_pairs += len(qa_pairs)
                if verbose:
                    print(f"  {doc_id}: {len(qa_pairs)} QA pairs")
        
        if verbose:
            print(f"\n✅ Generated QA pairs per document:")
        
        print(f"✅ Total: {total_qa_pairs} QA pairs generated from {len(documents)} documents")

        return result

    def _get_prompt_name(self) -> str:
        """Return the name of the prompt to use from config"""
        return "qa_generation"