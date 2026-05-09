import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# ============================================================
# Physicochemical Properties Dictionary (Complete)
# ============================================================

HYDROPHOBICITY = {
    'A': 0.700, 'R': 0.000, 'N': 0.111, 'D': 0.111, 'C': 0.778,
    'Q': 0.111, 'E': 0.111, 'G': 0.456, 'H': 0.144, 'I': 1.000,
    'L': 0.922, 'K': 0.067, 'M': 0.711, 'F': 0.811, 'P': 0.322,
    'S': 0.411, 'T': 0.422, 'W': 0.400, 'Y': 0.356, 'V': 0.967,
    'X': 0.500, 'U': 0.500, 'B': 0.111, 'Z': 0.111, 'O': 0.500
}

MOLECULAR_WEIGHT = {
    'A': 0.115, 'R': 0.449, 'N': 0.230, 'D': 0.216, 'C': 0.189,
    'Q': 0.264, 'E': 0.251, 'G': 0.057, 'H': 0.312, 'I': 0.230,
    'L': 0.230, 'K': 0.264, 'M': 0.264, 'F': 0.345, 'P': 0.172,
    'S': 0.149, 'T': 0.184, 'W': 0.471, 'Y': 0.378, 'V': 0.195,
    'X': 0.250, 'U': 0.300, 'B': 0.223, 'Z': 0.258, 'O': 0.400
}

ISOELECTRIC_POINT = {
    'A': 0.439, 'R': 0.855, 'N': 0.385, 'D': 0.192, 'C': 0.380,
    'Q': 0.401, 'E': 0.224, 'G': 0.434, 'H': 0.571, 'I': 0.434,
    'L': 0.439, 'K': 0.720, 'M': 0.408, 'F': 0.397, 'P': 0.461,
    'S': 0.410, 'T': 0.404, 'W': 0.424, 'Y': 0.397, 'V': 0.439,
    'X': 0.450, 'U': 0.400, 'B': 0.289, 'Z': 0.313, 'O': 0.450
}

CHARGE = {
    'A': 0.0, 'R': 1.0, 'N': 0.0, 'D': -1.0, 'C': 0.0,
    'Q': 0.0, 'E': -1.0, 'G': 0.0, 'H': 0.1, 'I': 0.0,
    'L': 0.0, 'K': 1.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
    'X': 0.0, 'U': 0.0, 'B': -0.5, 'Z': -0.5, 'O': 0.0
}

POLARITY = {
    'A': 0.000, 'R': 0.691, 'N': 0.477, 'D': 0.538, 'C': 0.128,
    'Q': 0.492, 'E': 0.538, 'G': 0.000, 'H': 0.312, 'I': 0.000,
    'L': 0.000, 'K': 0.691, 'M': 0.092, 'F': 0.031, 'P': 0.262,
    'S': 0.323, 'T': 0.323, 'W': 0.062, 'Y': 0.154, 'V': 0.000,
    'X': 0.250, 'U': 0.100, 'B': 0.508, 'Z': 0.515, 'O': 0.250
}

VOLUME = {
    'A': 0.167, 'R': 0.596, 'N': 0.319, 'D': 0.299, 'C': 0.278,
    'Q': 0.406, 'E': 0.387, 'G': 0.111, 'H': 0.411, 'I': 0.424,
    'L': 0.424, 'K': 0.500, 'M': 0.432, 'F': 0.535, 'P': 0.298,
    'S': 0.208, 'T': 0.285, 'W': 0.681, 'Y': 0.574, 'V': 0.354,
    'X': 0.380, 'U': 0.350, 'B': 0.309, 'Z': 0.396, 'O': 0.500
}

SURFACE_AREA = {
    'A': 0.245, 'R': 0.700, 'N': 0.384, 'D': 0.371, 'C': 0.306,
    'Q': 0.480, 'E': 0.459, 'G': 0.192, 'H': 0.457, 'I': 0.449,
    'L': 0.457, 'K': 0.559, 'M': 0.478, 'F': 0.557, 'P': 0.339,
    'S': 0.282, 'T': 0.351, 'W': 0.686, 'Y': 0.596, 'V': 0.382,
    'X': 0.420, 'U': 0.380, 'B': 0.378, 'Z': 0.470, 'O': 0.500
}

H_BOND_DONOR = {
    'A': 0.167, 'R': 0.833, 'N': 0.500, 'D': 0.167, 'C': 0.167,
    'Q': 0.500, 'E': 0.167, 'G': 0.167, 'H': 0.333, 'I': 0.167,
    'L': 0.167, 'K': 0.500, 'M': 0.167, 'F': 0.167, 'P': 0.000,
    'S': 0.333, 'T': 0.333, 'W': 0.333, 'Y': 0.333, 'V': 0.167,
    'X': 0.250, 'U': 0.167, 'B': 0.333, 'Z': 0.333, 'O': 0.250
}

H_BOND_ACCEPTOR = {
    'A': 0.200, 'R': 0.400, 'N': 0.600, 'D': 0.800, 'C': 0.200,
    'Q': 0.600, 'E': 0.800, 'G': 0.200, 'H': 0.400, 'I': 0.200,
    'L': 0.200, 'K': 0.400, 'M': 0.200, 'F': 0.200, 'P': 0.200,
    'S': 0.400, 'T': 0.400, 'W': 0.200, 'Y': 0.400, 'V': 0.200,
    'X': 0.350, 'U': 0.200, 'B': 0.700, 'Z': 0.700, 'O': 0.350
}

ALPHA_HELIX = {
    'A': 0.830, 'R': 0.621, 'N': 0.443, 'D': 0.540, 'C': 0.475,
    'Q': 0.653, 'E': 0.876, 'G': 0.302, 'H': 0.540, 'I': 0.621,
    'L': 0.751, 'K': 0.686, 'M': 0.816, 'F': 0.653, 'P': 0.302,
    'S': 0.475, 'T': 0.475, 'W': 0.605, 'Y': 0.443, 'V': 0.589,
    'X': 0.550, 'U': 0.500, 'B': 0.492, 'Z': 0.765, 'O': 0.550
}

BETA_SHEET = {
    'A': 0.500, 'R': 0.578, 'N': 0.422, 'D': 0.344, 'C': 0.688,
    'Q': 0.641, 'E': 0.328, 'G': 0.469, 'H': 0.563, 'I': 0.969,
    'L': 0.781, 'K': 0.469, 'M': 0.641, 'F': 0.828, 'P': 0.344,
    'S': 0.469, 'T': 0.672, 'W': 0.797, 'Y': 0.891, 'V': 1.000,
    'X': 0.600, 'U': 0.600, 'B': 0.383, 'Z': 0.485, 'O': 0.600
}

TURN = {
    'A': 0.389, 'R': 0.611, 'N': 0.889, 'D': 0.889, 'C': 0.667,
    'Q': 0.611, 'E': 0.333, 'G': 1.000, 'H': 0.556, 'I': 0.278,
    'L': 0.333, 'K': 0.611, 'M': 0.389, 'F': 0.389, 'P': 0.889,
    'S': 0.778, 'T': 0.611, 'W': 0.611, 'Y': 0.667, 'V': 0.278,
    'X': 0.550, 'U': 0.500, 'B': 0.889, 'Z': 0.472, 'O': 0.550
}

HYDROPHOBIC_MOMENT = {
    'A': 0.310, 'R': 0.000, 'N': 0.060, 'D': 0.030, 'C': 0.350,
    'Q': 0.000, 'E': 0.000, 'G': 0.000, 'H': 0.130, 'I': 0.990,
    'L': 0.940, 'K': 0.060, 'M': 0.640, 'F': 1.000, 'P': 0.360,
    'S': 0.120, 'T': 0.210, 'W': 0.810, 'Y': 0.680, 'V': 0.760,
    'X': 0.400, 'U': 0.400, 'B': 0.045, 'Z': 0.000, 'O': 0.400
}

PKA_SIDECHAIN = {
    'A': 0.500, 'R': 0.857, 'N': 0.500, 'D': 0.271, 'C': 0.571,
    'Q': 0.500, 'E': 0.300, 'G': 0.500, 'H': 0.429, 'I': 0.500,
    'L': 0.500, 'K': 0.743, 'M': 0.500, 'F': 0.500, 'P': 0.500,
    'S': 0.500, 'T': 0.500, 'W': 0.500, 'Y': 0.729, 'V': 0.500,
    'X': 0.500, 'U': 0.500, 'B': 0.386, 'Z': 0.400, 'O': 0.500
}

FLEXIBILITY = {
    'A': 0.360, 'R': 0.530, 'N': 0.460, 'D': 0.510, 'C': 0.350,
    'Q': 0.490, 'E': 0.500, 'G': 0.540, 'H': 0.320, 'I': 0.460,
    'L': 0.370, 'K': 0.470, 'M': 0.300, 'F': 0.310, 'P': 0.510,
    'S': 0.510, 'T': 0.440, 'W': 0.310, 'Y': 0.420, 'V': 0.390,
    'X': 0.420, 'U': 0.380, 'B': 0.485, 'Z': 0.495, 'O': 0.420
}

ALL_FEATURES = [
    HYDROPHOBICITY, MOLECULAR_WEIGHT, ISOELECTRIC_POINT, CHARGE,
    POLARITY, VOLUME, SURFACE_AREA, H_BOND_DONOR, H_BOND_ACCEPTOR,
    ALPHA_HELIX, BETA_SHEET, TURN, HYDROPHOBIC_MOMENT, PKA_SIDECHAIN,
    FLEXIBILITY
]

FEATURE_NAMES = [
    'Hydrophobicity', 'MolecularWeight', 'IsoelectricPoint', 'Charge',
    'Polarity', 'Volume', 'SurfaceArea', 'HBondDonor', 'HBondAcceptor',
    'AlphaHelix', 'BetaSheet', 'Turn', 'HydrophobicMoment', 'pKaSidechain',
    'Flexibility', 'IUPred_Short', 'IUPred_Long', 'IUPred_Anchor'
]

PHYCHEM_DIM = 18
TRROSETTA_DIM = 7

# ============================================================
# Helper Functions
# ============================================================

def encode_sequence(seq: str) -> np.ndarray:
    """Encode amino acid sequence into physicochemical feature matrix"""
    seq = seq.upper()
    features = np.zeros((len(seq), 15), dtype=np.float32)
    for i, aa in enumerate(seq):
        for j, feat_dict in enumerate(ALL_FEATURES):
            features[i, j] = feat_dict.get(aa, 0.5)
    return features

def parse_fasta_with_labels(fasta_path: str) -> dict:
    """Parse FASTA file with labels"""
    data = {}
    current_id = None
    current_ph = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    data[current_id] = (''.join(current_seq), current_ph)

                parts = line[1:].split('|')
                current_id = parts[0].strip()
                try:
                    current_ph = float(parts[3].strip())
                except (IndexError, ValueError):
                    current_ph = 7.0
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            data[current_id] = (''.join(current_seq), current_ph)

    return data

def ph_to_coarse_class(ph: float) -> int:
    """Convert pH value to 3-class coarse labels"""
    if ph < 5:
        return 0  # Acidic
    elif ph < 9:
        return 1  # Neutral
    else:
        return 2  # Alkaline

def get_esmc_dim(esmc_dir: str) -> int:
    path = Path(esmc_dir)
    files = list(path.glob("*.npy"))
    if not files: return 0
    return np.load(files[0]).shape[-1]

def check_data_integrity(fasta_path: str, esmc_dir: str, iupred_dir: str, trrosetta_dir: str,
                        dataset_name: str = "", enable_fallback: bool = False):
    """
    Check data integrity (added trRosetta check)

    Args:
        enable_fallback: If True, proteins with missing features will be included (loaded from fallback)
    """
    print(f"\n{'='*60}")
    print(f" Data Integrity Check: {dataset_name}")
    print('='*60)

    # 1. Parse FASTA
    fasta_data = parse_fasta_with_labels(fasta_path)
    fasta_ids = set(fasta_data.keys())
    print(f" Proteins in FASTA: {len(fasta_ids)}")

    # 2. Check Feature Files
    def get_file_ids(dir_path, ext):
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        return {f.stem for f in path.glob(f"*{ext}")}

    esmc_ids = get_file_ids(esmc_dir, ".npy")
    iupred_ids = get_file_ids(iupred_dir, ".npy")
    # trRosetta is .npz file
    trrosetta_ids = get_file_ids(trrosetta_dir, ".npz")

    print(f" ESMC Features: {len(esmc_ids)}")
    print(f" IUPred Features: {len(iupred_ids)}")
    print(f" trRosetta Features: {len(trrosetta_ids)}")

    # 3. Check pH Values
    valid_ph_ids = {pid for pid, (_, ph) in fasta_data.items() if ph is not None and 0 <= ph <= 14}

    # 4.  Select Strategy based on enable_fallback
    if enable_fallback:
        # If fallback enabled, valid as long as pH exists (missing features loaded from fallback)
        valid_ids = valid_ph_ids
        print(f"\n Total Valid Proteins: {len(valid_ids)} (Fallback Mode Enabled)")

        # Count Missing
        missing_esmc = valid_ids - esmc_ids
        missing_iupred = valid_ids - iupred_ids
        missing_tr = valid_ids - trrosetta_ids

        total_missing = len(missing_esmc | missing_iupred | missing_tr)
        if total_missing > 0:
            print(f"  Detected {total_missing} proteins with missing features, will load from fallback directory (test set):")
            if missing_esmc:
                print(f"   Missing ESMC: {len(missing_esmc)} (e.g.: {list(missing_esmc)[:3]})")
            if missing_iupred:
                print(f"   Missing IUPred: {len(missing_iupred)} (e.g.: {list(missing_iupred)[:3]})")
            if missing_tr:
                print(f"   Missing trRosetta: {len(missing_tr)} (e.g.: {list(missing_tr)[:3]})")
        else:
            print(f" All protein features are complete, no fallback needed")
    else:
        # Fallback disabled, all features must exist
        valid_ids = valid_ph_ids & esmc_ids & iupred_ids & trrosetta_ids
        print(f"\n Total Valid Proteins: {len(valid_ids)}")

        missing_any = fasta_ids - valid_ids
        if missing_any:
            print(f" {len(missing_any)} proteins excluded (missing features or invalid pH)")
            print(f"   e.g.: {list(missing_any)[:3]}")

    stats = {
        'total_fasta': len(fasta_ids),
        'valid_count': len(valid_ids)
    }
    return valid_ids, stats

# ============================================================
# Dataset Class ( Supports cross-dataset feature sharing)
# ============================================================

class ProteinDataset(Dataset):
    """
    Integrated Dataset: Physicochemical + ESMC + IUPred + trRosetta(7 channels)
     Supports fallback mechanism: if feature file not found in primary dir, automatically load from fallback dir
    """
    def __init__(
        self,
        fasta_path: str,
        esmc_dir: str,
        iupred_dir: str,
        trrosetta_dir: str,
        valid_ids: set = None,
        max_len: int = 1024,
        #  New: fallback directory (optional)
        esmc_fallback_dir: str = None,
        iupred_fallback_dir: str = None,
        trrosetta_fallback_dir: str = None
    ):
        self.max_len = max_len
        self.esmc_dir = Path(esmc_dir)
        self.iupred_dir = Path(iupred_dir)
        self.trrosetta_dir = Path(trrosetta_dir)

        #  Save fallback directories
        self.esmc_fallback_dir = Path(esmc_fallback_dir) if esmc_fallback_dir else None
        self.iupred_fallback_dir = Path(iupred_fallback_dir) if iupred_fallback_dir else None
        self.trrosetta_fallback_dir = Path(trrosetta_fallback_dir) if trrosetta_fallback_dir else None

        fasta_data = parse_fasta_with_labels(fasta_path)

        if valid_ids is not None:
            self.data = [(pid, seq, ph) for pid, (seq, ph) in fasta_data.items() if pid in valid_ids]
        else:
            self.data = []

        self.esmc_dim = get_esmc_dim(esmc_dir)

        #  Count fallback usage
        self.fallback_stats = {'esmc': 0, 'iupred': 0, 'trrosetta': 0}
        #  Flag: print warning only once for each feature
        self.fallback_warned = {'esmc': False, 'iupred': False, 'trrosetta': False}

    def _load_with_fallback(self, primary_path, fallback_dir, protein_id, ext, feature_name):
        """
        Try to load file from primary directory, if failed, load from fallback directory

        Args:
            primary_path: Primary directory file path
            fallback_dir: Fallback directory (Path object or None)
            protein_id: Protein ID
            ext: File extension (e.g., '.npy', '.npz')
            feature_name: Feature name (for statistics)

        Returns:
            Loaded data, None if failed
        """
        # 1. Try loading from primary directory
        try:
            return np.load(primary_path)
        except Exception:
            pass

        # 2. If primary failed and fallback exists, try loading from fallback
        if fallback_dir is not None:
            fallback_path = fallback_dir / f"{protein_id}{ext}"
            try:
                data = np.load(fallback_path)
                #  Silent stats (no print to avoid multi-process spam)
                self.fallback_stats[feature_name] += 1
                return data
            except Exception:
                pass

        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        protein_id, seq, ph = self.data[idx]
        seq_len = len(seq)

        # 1. Physicochemical + IUPred Feature
        phychem = encode_sequence(seq)

        #  Load IUPred using fallback mechanism
        iupred = self._load_with_fallback(
            self.iupred_dir / f"{protein_id}.npy",
            self.iupred_fallback_dir,
            protein_id,
            '.npy',
            'iupred'
        )

        if iupred is not None:
            iupred = iupred.astype(np.float32)
            if iupred.shape[0] != seq_len:
                if iupred.shape[0] > seq_len:
                    iupred = iupred[:seq_len]
                else:
                    iupred = np.pad(iupred, ((0, seq_len-iupred.shape[0]), (0,0)), mode='constant')
        else:
            iupred = np.zeros((seq_len, 3), dtype=np.float32)

        phychem_features = np.concatenate([phychem, iupred], axis=-1)

        # 2.  Load ESMC using fallback mechanism
        esmc_features = self._load_with_fallback(
            self.esmc_dir / f"{protein_id}.npy",
            self.esmc_fallback_dir,
            protein_id,
            '.npy',
            'esmc'
        )

        if esmc_features is not None:
            esmc_features = esmc_features.astype(np.float32)
            if esmc_features.shape[0] != seq_len:
                if esmc_features.shape[0] > seq_len:
                    esmc_features = esmc_features[:seq_len]
                else:
                    esmc_features = np.pad(esmc_features, ((0, seq_len-esmc_features.shape[0]), (0,0)), mode='constant')
        else:
            # If load failed completely, use zero matrix
            esmc_features = np.zeros((seq_len, self.esmc_dim), dtype=np.float32)

        # 3.  Load trRosetta geometric features using fallback mechanism (L, L, 7)
        tr_data = self._load_with_fallback(
            self.trrosetta_dir / f"{protein_id}.npz",
            self.trrosetta_fallback_dir,
            protein_id,
            '.npz',
            'trrosetta'
        )

        if tr_data is not None:
            try:
                geom_features = tr_data['features'].astype(np.float32)
            except:
                geom_features = np.zeros((seq_len, seq_len, 7), dtype=np.float32)
        else:
            geom_features = np.zeros((seq_len, seq_len, 7), dtype=np.float32)

        # Ensure size matches (handle slight length diff from MSA)
        curr_L = geom_features.shape[0]
        if curr_L != seq_len:
            target_L = seq_len
            new_geom = np.zeros((target_L, target_L, 7), dtype=np.float32)
            min_L = min(curr_L, target_L)
            new_geom[:min_L, :min_L, :] = geom_features[:min_L, :min_L, :]
            geom_features = new_geom

        # 4. Unified Padding to max_len
        mask = np.ones(seq_len, dtype=np.float32)
        actual_len = seq_len

        if seq_len > self.max_len:
            # Truncate
            phychem_features = phychem_features[:self.max_len]
            esmc_features = esmc_features[:self.max_len]
            geom_features = geom_features[:self.max_len, :self.max_len, :]
            mask = np.ones(self.max_len, dtype=np.float32)
            actual_len = self.max_len
        elif seq_len < self.max_len:
            # Pad
            pad_len = self.max_len - seq_len

            # 1D Padding
            phychem_features = np.pad(phychem_features, ((0, pad_len), (0, 0)), mode='constant')
            esmc_features = np.pad(esmc_features, ((0, pad_len), (0, 0)), mode='constant')

            # 2D Padding (pad first two dims, channel not padded)
            geom_features = np.pad(geom_features, ((0, pad_len), (0, pad_len), (0, 0)), mode='constant')

            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])

        return {
            'phychem_features': torch.FloatTensor(phychem_features),
            'esmc_features': torch.FloatTensor(esmc_features),
            'geom_features': torch.FloatTensor(geom_features),
            'mask': torch.FloatTensor(mask),
            'ph_value': torch.FloatTensor([ph]),
            'coarse_label': torch.LongTensor([ph_to_coarse_class(ph)]),
            'seq_len': actual_len
        }

    def print_fallback_stats(self):
        """Print fallback usage stats"""
        if any(self.fallback_stats.values()):
            print(f"\n Fallback Feature Loading Stats:")
            for feat, count in self.fallback_stats.items():
                if count > 0:
                    print(f"  {feat}: {count} samples loaded from fallback directory")

def collate_fn(batch):
    return {
        'phychem_features': torch.stack([x['phychem_features'] for x in batch]),
        'esmc_features': torch.stack([x['esmc_features'] for x in batch]),
        'geom_features': torch.stack([x['geom_features'] for x in batch]),
        'mask': torch.stack([x['mask'] for x in batch]),
        'ph_value': torch.stack([x['ph_value'] for x in batch]),
        'coarse_label': torch.cat([x['coarse_label'] for x in batch]),
        'seq_len': [x['seq_len'] for x in batch]
    }
