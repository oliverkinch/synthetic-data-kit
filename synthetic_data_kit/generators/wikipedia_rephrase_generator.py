# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate Wikipedia-style paraphrases from text

from typing import Dict, List, Any, Optional
from pathlib import Path

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.base import BaseGenerator


class WikipediaRephraseGenerator(BaseGenerator):
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the Wikipedia Rephrase Generator with an LLM client and optional config"""
        super().__init__(client, config_path)
    
    def process_responses(self, 
                         documents: List[Dict[str, Any]], 
                         responses: List[str],
                         verbose: bool = False) -> List[Dict[str, Any]]:
        """Process Wikipedia rephrasing responses into structured results
        
        Args:
            documents: Original input documents
            responses: Raw LLM responses (rephrased text)
            verbose: Whether to show detailed output
            
        Returns:
            List of results with rephrased text
        """
        results = []
        for i, (doc, rephrased) in enumerate(zip(documents, responses)):
            result = {
                "id": doc["id"],
                "text": rephrased,
            }
            results.append(result)
            
            if verbose:
                print(f"  Rephrased text for {doc['id']} ({len(rephrased)} chars)")
        
        print(f"✅ Successfully rephrased {len(results)} documents")
        
        return results

    def _get_prompt_name(self) -> str:
        """Return the name of the prompt to use from config"""
        return "wikipedia_rephrase"

