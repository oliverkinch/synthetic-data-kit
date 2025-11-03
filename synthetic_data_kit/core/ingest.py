# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest parquet files

import os
from pathlib import Path
from typing import Optional, Dict, Any

from synthetic_data_kit.utils.config import get_path_config


def determine_parser(file_path: str, config: Dict[str, Any], multimodal: bool = False):
    """Determine the appropriate parser for a parquet file"""
    from synthetic_data_kit.parsers.parquet_parser import ParquetParser

    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext != ".parquet":
        raise ValueError(
            f"Only .parquet files are supported. Got: {ext}\n"
            f"Please provide a parquet file with a 'text' column."
        )
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    return ParquetParser()


def process_file(
    file_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    multimodal: bool = False,
) -> str:
    """Process a parquet file and convert it to lance format

    Args:
        file_path: Path to the parquet file
        output_dir: Directory to save parsed text (if None, uses config)
        output_name: Custom filename for output (if None, uses original name)
        config: Configuration dictionary (if None, uses default)
        multimodal: Whether to use the multimodal parser (not used for parquet)

    Returns:
        Path to the output lance file
    """
    from synthetic_data_kit.utils.lance_utils import create_lance_dataset
    import pyarrow as pa
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Get parser (will validate it's a parquet file)
    parser = determine_parser(file_path, config, multimodal)

    # Parse the parquet file (extracts text and id columns)
    content = parser.parse(file_path)

    # Generate output filename if not provided
    if not output_name:
        base_name = os.path.basename(file_path)
        output_name = os.path.splitext(base_name)[0]

    output_name += ".lance"
    output_path = os.path.join(output_dir, output_name)

    # Create schema with text and id fields
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("id", pa.string())
    ])

    create_lance_dataset(content, output_path, schema=schema)

    return output_path
