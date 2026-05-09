import sys
import os
import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

def main():
    if len(sys.argv) < 2:
        print("Usage: python ESM-C_extraction.py <fasta_file>")
        sys.exit(1)

    fasta_file = sys.argv[1]
    
    # Get the stem of the filename (e.g., A7RDD3 from A7RDD3.fasta)
    base_name = os.path.splitext(os.path.basename(fasta_file))[0]
    output_file = f"ESM-C_{base_name}.npy"

    # Read FASTA file
    try:
        with open(fasta_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) < 2:
            print("Error: FASTA file must contain at least a header line and a sequence line.")
            sys.exit(1)
            
        header = lines[0]
        sequence = lines[1] # User specified strict 2nd line as sequence
        
        if not header.startswith(">"):
            print(f"Warning: First line '{header}' does not start with '>'")

    except Exception as e:
        print(f"Error reading file {fasta_file}: {e}")
        sys.exit(1)

    print(f"Processing sequence for {header} (Length: {len(sequence)})")

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESMC model on {device}...")
    
    try:
        client = ESMC.from_pretrained("esmc_300m").to(device)
        
        protein = ESMProtein(sequence=sequence)
        
        # Encode
        protein_tensor = client.encode(protein)
        
        # Extract logits/embeddings
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        
        # Get embeddings
        embeddings = logits_output.embeddings
        
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
            
        # Save
        np.save(output_file, embeddings)
        print(f"Embeddings saved to {output_file}")
        print(f"Shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"Error running model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
