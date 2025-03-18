#!/usr/bin/env python3
"""
Standalone PDB cleaning script.
Takes an input directory, cleans all PDB files, and saves them to an output directory.
"""

import os
import sys
import logging
from typing import Dict, Any
import concurrent.futures
from functools import partial

# Assuming mdcath.utils.logging_utils is available
from mdcath.utils.logging_utils import log_info, log_error

# Assuming pdbUtils is available
from pdbUtils import pdbUtils

# Fixed configuration with all cleaning options enabled
CLEANING_CONFIG = {
    "replace_chain_0_with_A": True,
    "fix_atom_numbering": True,
    "remove_hydrogens": True,
}

def clean_pdb_file(input_pdb: str, output_pdb: str, config: Dict[str, Any]) -> bool:
    """
    Clean a PDB file using pdbUtils.

    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path to output PDB file
        config: Cleaning configuration parameters

    Returns:
        True if cleaning was successful, False otherwise
    """
    try:
        # Convert the input PDB file to a DataFrame using pdbUtils
        pdb_df = pdbUtils.pdb2df(input_pdb)
        check_for_res_atom_dupes(pdb_df)
        # Get number of atoms before cleaning
        initial_atoms = len(pdb_df)

        # Replace chain identifier "0" with "A"
        if config["replace_chain_0_with_A"] and len(pdb_df.columns) > 4:
            chain_col = pdb_df.columns[4]
            pdb_df[chain_col] = pdb_df[chain_col].apply(lambda x: 'A' if str(x).strip() == '0' else x)

        # Fix atom numbering
        if config["fix_atom_numbering"] and "atom_num" in pdb_df.columns:
            pdb_df["atom_num"] = range(1, len(pdb_df) + 1)

        # Remove hydrogens
        if config["remove_hydrogens"] and "element" in pdb_df.columns:
            pdb_df = pdb_df[pdb_df["element"] != "H"]

        # Get number of atoms after cleaning
        final_atoms = len(pdb_df)

        # Write the DataFrame back to a PDB file
        pdbUtils.df2pdb(pdb_df, output_pdb)

        # Log changes
        if initial_atoms != final_atoms:
            log_info(f"Cleaned {os.path.basename(input_pdb)}: {initial_atoms} atoms -> {final_atoms} atoms")
        else:
            log_info(f"Cleaned {os.path.basename(input_pdb)} without atom count changes")

        return True
    except Exception as e:
        log_error(f"Failed to clean PDB {input_pdb}: {e}")
        return False


def check_for_res_atom_dupes(pdb_df):

    for chain_id, chain_df in pdb_df.groupby("CHAIN_ID"):
        for res_id, res_df in chain_df.groupby("RES_ID"):
            atom_names_list = res_df["ATOM_NAME"].to_list()
            atom_names_set = set(atom_names_list)
            
def process_file(output_dir: str, config: Dict[str, Any], pdb_file: str) -> bool:
    """
    Process a single PDB file: clean it and save to output directory.

    Args:
        output_dir: Directory to save cleaned PDB files
        config: Cleaning configuration parameters
        pdb_file: Path to the PDB file to process

    Returns:
        True if cleaning was successful, False otherwise
    """
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_fixed.pdb")
    success = clean_pdb_file(pdb_file, output_path, config)
    if success:
        log_info(f"Cleaned {os.path.basename(pdb_file)} -> {os.path.basename(output_path)}")
    else:
        log_error(f"Failed to clean {os.path.basename(pdb_file)}")
    return success

def clean_all_pdbs(input_dir: str, output_dir: str, config: Dict[str, Any], num_cores: int = 4) -> Dict[str, Any]:
    """
    Clean all PDB files in the input directory and save to output directory.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory to save cleaned PDB files
        config: Cleaning configuration parameters
        num_cores: Number of processor cores to use

    Returns:
        Dictionary with cleaning statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all PDB files in the input directory
    pdb_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdb")]
    log_info(f"Found {len(pdb_files)} PDB files to clean")

    # Process files using multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        process_func = partial(process_file, output_dir, config)
        results = list(executor.map(process_func, pdb_files))

    # Calculate statistics
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    log_info(f"Cleaning completed: {successful} successful, {failed} failed")

    return {
        "total_files": len(pdb_files),
        "successful": successful,
        "failed": failed
    }

if __name__ == "__main__":
    # Check for correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python pdb_cleaner.py input_dir output_dir")
        sys.exit(1)

    # Get input and output directories from command-line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Run the cleaning process
    clean_all_pdbs(input_dir, output_dir, CLEANING_CONFIG)