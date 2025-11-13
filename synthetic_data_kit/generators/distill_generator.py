# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Distill text into concise passages

from typing import Dict, List, Any, Optional
import os
from pathlib import Path

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.base import BaseGenerator
from synthetic_data_kit.utils.config import get_prompt


class DistillGenerator(BaseGenerator):
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the Distill Generator with an LLM client and optional config"""
        super().__init__(client, config_path)
    
    def process_responses(self, 
                         documents: List[Dict[str, Any]], 
                         responses: List[str],
                         verbose: bool = False) -> List[Dict[str, Any]]:
        """Process distill responses into structured results
        
        Args:
            documents: Original input documents
            responses: Raw LLM responses (distilled text)
            verbose: Whether to show detailed output
            
        Returns:
            List of results with distilled text and metadata
        """
        results = []
        for i, (doc, distilled) in enumerate(zip(documents, responses)):
            result = {
                "id": doc["id"],
                "text": distilled,
                "original_text": doc["text"],
                "original_length": len(doc["text"]),
                "distilled_length": len(distilled),
                "compression_ratio": len(distilled) / len(doc["text"])
            }
            results.append(result)
            
            if verbose:
                print(f"  Distilled text for {doc['id']} ({len(distilled)} chars, "
                      f"{result['compression_ratio']:.2%} of original)")
        
        print(f"✅ Successfully distilled {len(results)} documents")
        
        return results

    def _get_prompt_name(self) -> str:
        """Return the name of the prompt to use from config"""
        return "distill"
