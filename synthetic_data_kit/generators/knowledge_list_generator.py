# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Extract knowledge from text into information-dense lists

from typing import Dict, List, Any, Optional
import os
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import load_config, get_generation_config, get_prompt


class KnowledgeListGenerator:
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the Knowledge List Generator with an LLM client and optional config"""
        self.client = client
        
        # Load config
        self.config = load_config(config_path)
        
        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
    
    def extract_knowledge(self, document_text: str) -> str:
        """Extract key knowledge and facts from document text"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        if verbose:
            print("Extracting knowledge from document...")
        
        # Get knowledge_list prompt from config
        prompt_template = get_prompt(self.config, "knowledge_list")
        prompt = prompt_template.format(text=document_text)
        
        # Generate knowledge list using the LLM
        messages = [
            {"role": "system", "content": prompt}
        ]
        
        knowledge_text = self.client.chat_completion(
            messages,
            temperature=self.generation_config.get("temperature", 0.7)
        )
        
        if verbose:
            print(f"Knowledge extracted ({len(knowledge_text)} chars)")
        
        return knowledge_text
    
    def process_documents(self, 
                          documents: List[Dict[str, Any]], 
                          verbose: bool = False) -> List[Dict[str, Any]]:
        """Process multiple documents and extract knowledge from each one
        
        Args:
            documents: List of documents with 'text' field and 'id' field
            verbose: Whether to show progress
            
        Returns:
            List of results with original and extracted knowledge, preserving id
        """
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'
            
        results = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Extracting knowledge from {len(documents)} documents...", 
                total=len(documents)
            )
            
            for i, doc in enumerate(documents):
                doc_text = doc["text"]
                
                if verbose:
                    print(f"\nProcessing document {i+1}/{len(documents)} ({len(doc_text)} chars)...")
                
                # Extract knowledge
                knowledge = self.extract_knowledge(document_text=doc_text)
                
                # Create result entry with text and id
                result = {
                    "id": doc["id"],
                    "text": knowledge,
                    "original_length": len(doc_text),
                    "knowledge_length": len(knowledge),
                    "compression_ratio": len(knowledge) / len(doc_text)
                }
                
                results.append(result)
                
                if verbose:
                    print(f"  Extracted knowledge for {doc['id']} ({len(knowledge)} chars, "
                          f"{result['compression_ratio']:.2%} of original)")
                
                progress.advance(task)
        
        if verbose:
            print(f"\n✅ Successfully extracted knowledge from {len(results)} documents")
        
        return results

