# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Distill text into concise passages

from typing import Dict, List, Any, Optional
import os
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import load_config, get_generation_config, get_prompt


class DistillGenerator:
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the Distill Generator with an LLM client and optional config"""
        self.client = client
        
        # Load config
        self.config = load_config(config_path)
        
        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
    
    def distill_text(self, document_text: str) -> str:
        """Distill document text into a concise and clear passage"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        if verbose:
            print("Distilling document text...")
        
        # Get distill prompt from config
        prompt_template = get_prompt(self.config, "distill")
        prompt = prompt_template.format(text=document_text)
        
        # Generate distilled text using the LLM
        distilled_text = self.client.generate(prompt)
        
        if verbose:
            print(f"Distilled text generated ({len(distilled_text)} chars)")
        
        return distilled_text
    
    def process_documents(self, 
                          documents: List[Dict[str, Any]], 
                          verbose: bool = False) -> List[Dict[str, Any]]:
        """Process multiple documents and distill each one
        
        Args:
            documents: List of documents with 'text' field
            verbose: Whether to show progress
            
        Returns:
            List of results with original and distilled text
        """
        results = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Distilling {len(documents)} documents...", 
                total=len(documents)
            )
            
            for i, doc in enumerate(documents):
                doc_text = doc.get("text", "")
                
                if verbose:
                    print(f"\nProcessing document {i+1}/{len(documents)} ({len(doc_text)} chars)...")
                
                # Distill the text
                distilled = self.distill_text(doc_text)
                
                # Create result entry
                result = {
                    "original_text": doc_text,
                    "distilled_text": distilled,
                    "original_length": len(doc_text),
                    "distilled_length": len(distilled),
                    "compression_ratio": len(distilled) / len(doc_text) if len(doc_text) > 0 else 0
                }
                
                # Include metadata if available
                if "metadata" in doc:
                    result["metadata"] = doc["metadata"]
                
                results.append(result)
                
                if verbose:
                    print(f"  Generated distilled text ({len(distilled)} chars, "
                          f"{result['compression_ratio']:.2%} of original)")
                
                progress.advance(task)
        
        return results

