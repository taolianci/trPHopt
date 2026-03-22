import torch
import numpy as np
import json
import argparse
from pathlib import Path
from model import TwoStagePredictor
from utils import encode_sequence, ph_to_coarse_class

def get_args():
    parser = argparse.ArgumentParser(description="Protein pH Prediction Single Inference")
    parser.add_argument('model_path', type=str, help='Path to model checkpoint or directory')
    parser.add_argument('esmc_path', type=str, help='Path to ESMC features (.npy)')
    parser.add_argument('iupred_path', type=str, help='Path to IUPred features (.npy)')
    parser.add_argument('trrosetta_path', type=str, help='Path to trRosetta features (.npz)')
    parser.add_argument('fasta_path', type=str, help='Path to FASTA file containing sequence')
    return parser.parse_args()

def read_fasta(fasta_path):
    """Read the first sequence from the FASTA file"""
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines or not lines[0].startswith('>'):
        raise ValueError(f"Invalid or empty FASTA file: {fasta_path}. First line must start with >")
        
    protein_id = lines[0][1:] # Get content after > as ID
    
    # Simple multi-line FASTA processing: read until next > or end of file
    sequence_parts = []
    for line in lines[1:]:
        if line.startswith('>'):
            break
        sequence_parts.append(line)
        
    sequence = "".join(sequence_parts)
    return protein_id, sequence

def load_model(checkpoint_path_str, device):
    """Load model and configuration"""
    checkpoint_path = Path(checkpoint_path_str)
    
    # Smart path handling: if directory, look for best_model.pt; if file without suffix, try adding .pt
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "best_model.pt"
    elif not checkpoint_path.suffix and not checkpoint_path.exists():
        checkpoint_path = checkpoint_path.with_suffix('.pt')
        
    config_path = checkpoint_path.parent / "config.json"
    print(f"Loading model config: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Get ESMC dimension (default 960)
    esmc_dim = 960 
    # If feature directory is accessible, we could get it dynamically, otherwise use default
    
    model = TwoStagePredictor(
        phychem_dim=config.get('phychem_dim', 18),
        esmc_dim=esmc_dim,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_len=config['max_len'],
        n_coarse_classes=config.get('n_coarse_classes', 3),
        fusion_type=config.get('fusion_type', 'esmc_dominant')
    )
    
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        print(f"Model loaded: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
    model.to(device)
    model.eval()
    return model, config

def process_single_protein(protein_data, max_len=256, esmc_dim=960):
    """Process single protein data, convert to model input format"""
    seq = protein_data['sequence']
    seq_len = len(seq)
    
    print(f"\n Processing protein: {protein_data.get('id', 'Unknown')}")
    print(f"   Sequence Length: {seq_len}")

    # 1. Physicochemical features (Sequence -> 15 dim)
    phychem = encode_sequence(seq) # (L, 15)
    
    # 2. IUPred features (Load -> 3 dim)
    iupred_path = Path(protein_data['iupred_path'])
    if iupred_path.exists():
        try:
            iupred = np.load(iupred_path) # Expect (L, 3)
            # Simple length check
            if iupred.shape[0] != seq_len:
                print(f"IUPred length mismatch ({iupred.shape[0]} vs {seq_len}), trying to adjust...")
                # Simple truncation or padding strategy, skipping complex handling, assuming alignment
                if iupred.shape[0] > seq_len:
                    iupred = iupred[:seq_len]
                else:
                    pad = np.zeros((seq_len - iupred.shape[0], 3))
                    iupred = np.concatenate([iupred, pad], axis=0)
        except Exception as e:
            print(f"Failed to load IUPred: {e}, using zero features")
            iupred = np.zeros((seq_len, 3))
    else:
        print(f"IUPred file not found: {iupred_path}, using zero features")
        iupred = np.zeros((seq_len, 3))
        
    phychem_features = np.concatenate([phychem, iupred], axis=-1) # (L, 18)

    # 3. ESMC features (Load -> 960 dim)
    esmc_path = Path(protein_data['esmc_path'])
    if esmc_path.exists():
        try:
            esmc_features = np.load(esmc_path) # Expect (L, 960)
            if esmc_features.shape[0] != seq_len:
                 # Same processing as above
                 if esmc_features.shape[0] > seq_len:
                    esmc_features = esmc_features[:seq_len]
                 else:
                    pad = np.zeros((seq_len - esmc_features.shape[0], esmc_dim))
                    esmc_features = np.concatenate([esmc_features, pad], axis=0)
        except Exception as e:
            print(f"Failed to load ESMC: {e}, using zero features")
            esmc_features = np.zeros((seq_len, esmc_dim))
    else:
        print(f"ESMC file not found: {esmc_path}, using zero features")
        esmc_features = np.zeros((seq_len, esmc_dim))

    # 4. trRosetta structural features (Load -> L, L, 7)
    tr_path = Path(protein_data['trrosetta_path'])
    if tr_path.exists():
        try:
            data = np.load(tr_path)
            geom_features = data if isinstance(data, np.ndarray) else data['features']
            
            # Check if dimensions meet requirements (L, L, 7)
            if geom_features.shape[-1] != 7:
                 print(f"trRosetta dimensions incorrect: {geom_features.shape}, expected (L, L, 7), using zero features")
                 geom_features = np.zeros((seq_len, seq_len, 7), dtype=np.float32)

            # Use logic in utils.py to handle size
            curr_L = geom_features.shape[0]
            if curr_L != seq_len:
                new_geom = np.zeros((seq_len, seq_len, 7), dtype=np.float32)
                min_L = min(curr_L, seq_len)
                new_geom[:min_L, :min_L, :] = geom_features[:min_L, :min_L, :]
                geom_features = new_geom
        except Exception as e:
             print(f"Failed to load trRosetta: {e}, using zero features")
             geom_features = np.zeros((seq_len, seq_len, 7))
    else:
        print(f"trRosetta file not found: {tr_path}, using zero features")
        geom_features = np.zeros((seq_len, seq_len, 7))

    # 5. Padding & Masking
    mask = np.ones(seq_len, dtype=np.float32)
    
    if seq_len > max_len:
        # Truncate
        phychem_features = phychem_features[:max_len]
        esmc_features = esmc_features[:max_len]
        geom_features = geom_features[:max_len, :max_len, :]
        mask = np.ones(max_len, dtype=np.float32)
    elif seq_len < max_len:
        # Pad
        pad_len = max_len - seq_len
        phychem_features = np.pad(phychem_features, ((0, pad_len), (0, 0)), mode='constant')
        esmc_features = np.pad(esmc_features, ((0, pad_len), (0, 0)), mode='constant')
        geom_features = np.pad(geom_features, ((0, pad_len), (0, pad_len), (0, 0)), mode='constant')
        mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])

    # Convert to Tensor and add Batch dimension
    return {
        'phychem_features': torch.FloatTensor(phychem_features).unsqueeze(0),
        'esmc_features': torch.FloatTensor(esmc_features).unsqueeze(0),
        'geom_features': torch.FloatTensor(geom_features).unsqueeze(0),
        'mask': torch.FloatTensor(mask).unsqueeze(0)
    }

def print_prediction(outputs, protein_id):
    """Format and output prediction results"""
    # Get prediction values
    pred_ph_calib = outputs['calibrated'].item()
    pred_ph_basic = outputs['regression'].item()
    
    # Get classification probabilities
    coarse_probs = torch.softmax(outputs['coarse'], dim=-1).squeeze().tolist()
    classes = ['Acidic (pH<5)', 'Neutral (5<=pH<9)', 'Alkaline (pH>=9)']
    pred_class_idx = outputs['coarse'].argmax(dim=-1).item()
    
    print(f"\n Prediction Result [{protein_id}]:")
    print(f"{'='*40}")
    print(f"Predicted pH (Calibrated): {pred_ph_calib:.4f}")
    print(f"   Predicted pH (Basic):      {pred_ph_basic:.4f}")
    print(f"{'-'*40}")
    print(f"Predicted Class: {classes[pred_class_idx]}")
    print(f"   Class Probability Distribution:")
    for i, (cls, prob) in enumerate(zip(classes, coarse_probs)):
        marker = " selected" if i == pred_class_idx else ""
        print(f"   - {cls:<18}: {prob:.4f}{marker}")
    print(f"{'='*40}\n")

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using Device: {device}")

    try:
        # Parse FASTA
        protein_id, sequence = read_fasta(args.fasta_path)
        print(f"Reading FASTA: {args.fasta_path}")
        print(f"   ID: {protein_id}")
        print(f"   Seq Len: {len(sequence)}")
        
        # Build data dictionary
        protein_data = {
            'id': protein_id,
            'sequence': sequence,
            'esmc_path': args.esmc_path,
            'iupred_path': args.iupred_path,
            'trrosetta_path': args.trrosetta_path
        }

        # 1. Load model
        model, config = load_model(args.model_path, device)
        
        # 2. Process data
        inputs = process_single_protein(
            protein_data, 
            max_len=config['max_len'], 
            esmc_dim=960 # default
        )
        
        # 3. Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 4. Inference
        with torch.no_grad():
            outputs = model(
                inputs['phychem_features'],
                inputs['esmc_features'],
                inputs['geom_features'],
                inputs['mask']
            )
            
        # 5. Output results
        print_prediction(outputs, protein_data['id'])
        
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
