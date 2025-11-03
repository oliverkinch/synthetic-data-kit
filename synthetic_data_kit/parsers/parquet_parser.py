# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Parquet parser logic

import os
from typing import Dict, Any, List

class ParquetParser:
    """Parser for Parquet files"""
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a Parquet file into plain text
        
        Args:
            file_path: Path to the Parquet file
            
        Returns:
            List of dictionaries with text content extracted from the parquet file
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for Parquet parsing. Install it with: pip install pandas pyarrow"
            )
        
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Check if there's a 'text' column - raise error if not present
        if 'text' not in df.columns:
            available_columns = ", ".join(df.columns)
            raise ValueError(
                f"Parquet file must contain a 'text' column. "
                f"Available columns: {available_columns}"
            )
        
        # Extract text from each row and return as separate entries
        result = []
        for _, row in df.iterrows():
            text_content = str(row['text']) if pd.notna(row['text']) else ""
            if text_content.strip():  # Only include non-empty text
                result.append({"text": text_content})
        
        # Return empty text if no valid rows found
        return result if result else [{"text": ""}]
    
    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a file
        
        Args:
            content: Extracted text content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

