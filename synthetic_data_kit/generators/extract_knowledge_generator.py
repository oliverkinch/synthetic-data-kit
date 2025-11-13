# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Extract and rewrite knowledge from text into clear passages

from typing import Dict, List, Any, Optional
from pathlib import Path

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.base import BaseGenerator


class ExtractKnowledgeGenerator(BaseGenerator):
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the Extract Knowledge Generator with an LLM client and optional config"""
        super().__init__(client, config_path)
    
    def process_responses(self, 
                         documents: List[Dict[str, Any]], 
                         responses: List[str],
                         verbose: bool = False) -> List[Dict[str, Any]]:
        """Process extract knowledge responses into structured results
        
        Args:
            documents: Original input documents
            responses: Raw LLM responses (rewritten knowledge passages)
            verbose: Whether to show detailed output
            
        Returns:
            List of results with extracted/rewritten knowledge and metadata
        """
        results = []
        for i, (doc, knowledge) in enumerate(zip(documents, responses)):
            result = {
                "id": doc["id"],
                "text": knowledge,
                "original_text": doc["text"],
                "original_length": len(doc["text"]),
                "knowledge_length": len(knowledge),
                "compression_ratio": len(knowledge) / len(doc["text"])
            }
            results.append(result)
            
            if verbose:
                print(f"  Extracted knowledge for {doc['id']} ({len(knowledge)} chars, "
                      f"{result["compression_ratio"]:.2%} of original)")
        
        print(f"✅ Successfully extracted knowledge from {len(results)} documents")
        
        return results

    def _get_prompt_name(self) -> str:
        """Return the name of the prompt to use from config"""
        return "extract_knowledge"

