# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

from typing import Dict, List, Any, Optional, Tuple
import json
import time
import os
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.text import split_into_chunks
from synthetic_data_kit.utils.llm_processing import parse_qa_pairs, parse_ratings, convert_to_conversation_format
from synthetic_data_kit.utils.config import load_config, get_generation_config, get_curate_config, get_prompt

class QAGenerator:
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client
        
        # Load config
        self.config = load_config(config_path)
        
        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)
    
    def generate_summary(self, 
                         document_text: str, 
                         rolling_summary: Optional[bool] = False) -> str:
        """Generate a summary of the document"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        if verbose:
            print("Generating document summary...")
        
        # Get summary prompt and params from config
        prompt = get_prompt(self.config, "summary")
        max_context_length = self.generation_config.get("max_context_length", 8000)
        summary_overlap = self.generation_config.get("summary_overlap", 0)

        if rolling_summary:
            summary_per_chunk = []
            #split text into long chunks for summarization
            chunks = split_into_chunks(document_text,
                                       chunk_size=max_context_length,
                                       overlap=summary_overlap)

            for chunk in chunks:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk}
                ]
                new_summary = self.client.chat_completion(
                    messages, 
                    temperature=0.1  # Use lower temperature for summaries
                )
                summary_per_chunk.append(new_summary)

            summary = " .".join(summary_per_chunk)
            # Summarize again to reduce overall length and redundancy
            summary = self.generate_summary(summary,
                                            rolling_summary=False)
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": document_text[0:max_context_length]}
            ]
            
            summary = self.client.chat_completion(
                messages, 
                temperature=0.1  # Use lower temperature for summaries
            )
        
        if verbose:
            print(f"Summary generated ({len(summary)} chars)")
        return summary
    
    def generate_qa_pairs(self, document_text: str) -> List[Dict[str, str]]:
        """Generate QA pairs from the document using batched processing"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Get generation config
        chunk_size = self.generation_config.get("chunk_size", 4000)
        temperature = self.generation_config.get("temperature", 0.7)
        overlap = self.generation_config.get("overlap", 200)
        batch_size = self.generation_config.get("batch_size", 32)
        
        # Split text into chunks
        chunks = split_into_chunks(
            document_text, 
            chunk_size=chunk_size, 
            overlap=overlap
        )
        
        if verbose:
            print(f"Generating QA pairs...")
            print(f"Document split into {len(chunks)} chunks")
            print(f"Using batch size of {batch_size}")
        
        all_qa_pairs = []
        
        # Get QA generation prompt template
        qa_prompt_template = get_prompt(self.config, "qa_generation")
        
        # Prepare all message batches
        all_messages = []
        for i, chunk in enumerate(chunks):
            # Format the prompt with text
            qa_prompt = qa_prompt_template.format(
                text=chunk
            )
            
            messages = [
                {"role": "system", "content": qa_prompt}
            ]
            all_messages.append(messages)
        
        print(f"Processing {len(chunks)} chunks to generate QA pairs...")
        
        # Set up progress tracking based on verbose mode
        if verbose:
            from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
            
            progress_columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ]
            
            progress_ctx = Progress(*progress_columns)
            generate_task = progress_ctx.add_task(f"Generating QA pairs", total=len(chunks))
            progress_ctx.start()
        else:
            progress_ctx = None
            generate_task = None
        
        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_messages = all_messages[batch_start:batch_end]
            current_batch_size = len(batch_messages)
            
            batch_num = batch_start//batch_size + 1
            total_batches = (len(chunks) + batch_size - 1)//batch_size
            
            # Simple progress indicator for non-verbose mode
            if not verbose:
                print(f"Processing batch {batch_num}/{total_batches}...", end="\r")
            else:
                print(f"Processing batch {batch_num}/{total_batches} with {current_batch_size} chunks")
            
            try:
                # Process the batch
                batch_responses = self.client.batch_completion(
                    batch_messages,
                    temperature=temperature,
                    batch_size=batch_size
                )
                
                # Process each response in the batch
                for j, response in enumerate(batch_responses):
                    chunk_index = batch_start + j
                    chunk_pairs = parse_qa_pairs(response)

                    if chunk_pairs:
                        all_qa_pairs.extend(chunk_pairs)
                    # Only add pairs up to the target limit
                    if verbose:
                        print(f"  Generated {len(chunk_pairs)} pairs from chunk {chunk_index+1} (total: {len(all_qa_pairs)})")
                
                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)
                
            except Exception as e:
                if verbose:
                    print(f"  Error processing batch {batch_num}: {str(e)}")
                
                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)
        
        # Stop progress bar if in verbose mode
        if progress_ctx:
            progress_ctx.stop()
        
        # Clear the progress line in non-verbose mode
        if not verbose:
            print(" " * 80, end="\r")
            print("Batch processing complete.")
        
        # Always print summary information, even in non-verbose mode
        print(f"Generated {len(all_qa_pairs)} QA pairs total")
        return all_qa_pairs
    
    def process_documents(self,
                        documents: List[Dict[str, Any]],
                        num_pairs: int = 25,
                        verbose: bool = False,
                        rolling_summary: Optional[bool] = False) -> Dict[str, Any]:
        """Process a list of documents to generate QA pairs without rating
        
        Each document is processed independently and QA pairs are generated for each one.
        num_pairs specifies how many QA pairs to generate PER DOCUMENT.
        """
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'

        all_qa_pairs = []
        
        if verbose:
            print(f"Processing {len(documents)} documents independently")
            print(f"Target: {num_pairs} QA pairs per document")
            print(f"Expected total: {num_pairs * len(documents)} QA pairs")
        
        # Process each document independently
        for i, doc in enumerate(documents, 1):
            doc_text = doc["text"]
            
            if verbose:
                print(f"\nProcessing document {i}/{len(documents)} ({len(doc_text)} chars)...")
            
            # Generate QA pairs for this document (num_pairs per document)
            qa_pairs = self.generate_qa_pairs(document_text=doc_text)
            
            # Attach source document id to each QA pair
            for qa_pair in qa_pairs:
                qa_pair["id"] = doc["id"]
            
            if verbose:
                print(f"  Generated {len(qa_pairs)} QA pairs for document {i}")
            
            all_qa_pairs.extend(qa_pairs)

        result = {
            "qa_pairs": all_qa_pairs
        }
        
        if verbose:
            print(f"\nTotal: {len(all_qa_pairs)} QA pairs generated from {len(documents)} documents")

        return result