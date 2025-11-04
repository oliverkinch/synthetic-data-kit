# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Base class for all generators

from typing import Dict, List, Any, Optional
import os
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import load_config, get_generation_config, get_prompt


class BaseGenerator:
    """Base class for all generators that process documents using LLMs"""
    
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the generator with an LLM client and optional config"""
        self.client = client
        
        # Load config
        self.config = load_config(config_path)
        
        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
    
    def _get_prompt_name(self) -> str:
        """Get the name of the prompt to use from config. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_prompt_name()")
    
    def process_responses(self, 
                         documents: List[Dict[str, Any]], 
                         responses: List[str],
                         verbose: bool = False) -> Any:
        """Process raw LLM responses into final result format. Override in subclasses.
        
        Args:
            documents: Original input documents
            responses: Raw LLM responses (one per document)
            verbose: Whether to show detailed output
            
        Returns:
            Processed results in the format expected by this generator
        """
        raise NotImplementedError("Subclasses must implement process_responses()")
    
    def process_documents(self, 
                          documents: List[Dict[str, Any]], 
                          verbose: bool = False) -> Any:
        """Process multiple documents using batch processing
        
        This method handles the common workflow:
        1. Prepare messages for all documents
        2. Process in parallel batches
        3. Call process_responses() to format results
        
        Args:
            documents: List of documents with 'text' field and 'id' field
            verbose: Whether to show progress
            
        Returns:
            Processed results (format determined by subclass)
        """
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'
        
        # Get batch size and temperature from config
        batch_size = self.generation_config.get("batch_size", 32)
        temperature = self.generation_config.get("temperature", 0.7)
        
        # Get prompt from config
        prompt_name = self._get_prompt_name()
        prompt = get_prompt(config=self.config, prompt_name=prompt_name)
        
        # Prepare all message batches
        all_messages = []
        for doc in documents:
            doc_text = doc["text"]
            
            system_content = prompt['system']
            user_content = prompt['user'].format(text=doc_text)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
                
            all_messages.append(messages)
        
        print(f"Processing {len(documents)} documents...")
        
        # Process in batches using batch_completion
        all_responses = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Processing {len(documents)} documents...", 
                total=len(documents)
            )
            
            for batch_start in range(0, len(documents), batch_size):
                batch_end = min(batch_start + batch_size, len(documents))
                batch_messages = all_messages[batch_start:batch_end]
                current_batch_size = len(batch_messages)
                
                batch_num = batch_start // batch_size + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                if not verbose:
                    print(f"Processing batch {batch_num}/{total_batches}...", end="\r")
                else:
                    print(f"Processing batch {batch_num}/{total_batches} with {current_batch_size} documents")
                

                batch_responses = self.client.batch_completion(
                    batch_messages,
                    temperature=temperature,
                    batch_size=batch_size
                )
                
                all_responses.extend(batch_responses)
                
                if verbose:
                    print(f"  Received {len(batch_responses)} responses")
                
                # Update progress
                progress.advance(task, current_batch_size)
                    
        # Clear the progress line in non-verbose mode
        if not verbose:
            print(" " * 80, end="\r")
        
        # Call subclass-specific response processing
        return self.process_responses(documents=documents, responses=all_responses, verbose=verbose)
