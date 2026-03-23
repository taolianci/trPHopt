import subprocess
import shutil
import sys
import os
import numpy as np
from pathlib import Path

# Set the root directory to the script's location
IUPRED_ROOT = Path(__file__).parent.resolve()

def run_command(cmd, cwd):
    """Run a shell command."""
    print(f"Running: {cmd}")
    # Use shell=True for complex commands with redirection, or handle redirection in Python
    # Here we will use shell=True to exactly match the request's command structure
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd),
        text=True,
        capture_output=True 
    )
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Stderr: {result.stderr}")
        sys.exit(1)

def parse_iupred_output(file_path, col_indices):
    """
    Parse IUPred2A output file.
    Skip lines until '# POS' is found, then read subsequent lines.
    Extract specified columns (0-based index).
    """
    data = []
    start_reading = False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("# POS"):
                start_reading = True
                continue
            
            if start_reading:
                parts = line.split()
                try:
                    row_data = [float(parts[i]) for i in col_indices]
                    data.append(row_data)
                except (ValueError, IndexError):
                    continue
                    
    return np.array(data)

def main():
    if len(sys.argv) > 1:
        input_fasta = Path(sys.argv[1])
    else:
        # Default to A7RDD3.fasta if not provided
        input_fasta = IUPRED_ROOT / "A7RDD3.fasta"

    if not input_fasta.exists():
        print(f"Input file not found: {input_fasta}")
        sys.exit(1)

    # 1. Force suffix .fasta to .seq
    seq_file = input_fasta.with_suffix(".seq")
    print(f"Copying {input_fasta} to {seq_file}")
    shutil.copy(input_fasta, seq_file)

    # Define output txt files
    long_output = IUPRED_ROOT / "iupred2a_result.long.txt"
    short_output = IUPRED_ROOT / "iupred2a_result.short.txt"
    
    # 2. Construct and run commands
    # Command 1: python iupred2a.py -a A7RDD3.seq long > iupred2a_result.long.txt
    cmd_long = f'{sys.executable} iupred2a.py -a "{seq_file.name}" long > "{long_output.name}"'
    run_command(cmd_long, IUPRED_ROOT)

    # Command 2: python iupred2a.py A7RDD3.seq short > iupred2a_result.short.txt
    cmd_short = f'{sys.executable} iupred2a.py "{seq_file.name}" short > "{short_output.name}"'
    run_command(cmd_short, IUPRED_ROOT)

    # 3. Parse outputs
    print(f"Parsing {long_output.name}...")
    # Long file: Collect 3rd (idx 2) and 4th (idx 3) columns [IUPred Score, ANCHOR Score]
    data_long = parse_iupred_output(long_output, [2, 3])
    
    print(f"Parsing {short_output.name}...")
    # Short file: Collect 3rd (idx 2) column [IUPred Short Score]
    data_short = parse_iupred_output(short_output, [2])

    if len(data_long) == 0 or len(data_short) == 0:
        print("Error: No data parsed from output files.")
        sys.exit(1)

    if len(data_long) != len(data_short):
        print(f"Error: Length mismatch. Long: {len(data_long)}, Short: {len(data_short)}")
        # Handle minimal length if robust
        min_len = min(len(data_long), len(data_short))
        data_long = data_long[:min_len]
        data_short = data_short[:min_len]

    # 4. Combine and swap columns
    # We have:
    # data_long[:, 0] -> Long Score
    # data_long[:, 1] -> ANCHOR Score
    # data_short[:, 0] -> Short Score
    
    # Requirement: "First file collect fourth column with second output file collect third column perform column exchange"
    # Original order if just appended: [Long, Anchor, Short]
    # Swapping Anchor (from 1st file) and Short (from 2nd file): [Long, Short, Anchor]
    
    col_long_score = data_long[:, 0]
    col_anchor_score = data_long[:, 1]
    col_short_score = data_short[:, 0]
    
    # Combined: [Long Score, Short Score, ANCHOR Score]
    final_data = np.column_stack((col_long_score, col_short_score, col_anchor_score))
    
    print(f"Final data shape: {final_data.shape}")
    
    # Save to .npy
    output_npy = IUPRED_ROOT / f"iupred_{input_fasta.stem}.npy"
    np.save(output_npy, final_data)
    print(f"Saved result to {output_npy}")

if __name__ == "__main__":
    main()
