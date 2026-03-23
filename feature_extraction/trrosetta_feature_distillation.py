#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trRosetta Feature Distillation Tool
Compress probability distributions (112 channels) into deterministic geometric values (7 channels)

Compress from 1TB scale to 35GB scale while preserving core structural information
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json


# ============================================================================
# trRosetta Bin Definition (based on official implementation)
# ============================================================================

# Distance bins (37): 2-20Å, step 0.5Å
DIST_BINS = np.array([
    2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
    7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5,
    12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5,
    17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0
], dtype=np.float32)

# Omega bins (25): -180° to +180°, step 15°
OMEGA_BINS = np.linspace(-180, 165, 25, dtype=np.float32)

# Theta bins (25): -180° to +180°, step 15°
THETA_BINS = np.linspace(-180, 165, 25, dtype=np.float32)

# Phi bins (13): 0° to 180°, step 15°
PHI_BINS = np.linspace(0, 180, 13, dtype=np.float32)


def calculate_expected_distance(dist_probs: np.ndarray) -> np.ndarray:
    """
    Calculate expected distance

    Args:
        dist_probs: (L, L, 37) distance probability distribution

    Returns:
        expected_dist: (L, L) expected distance value (Unit: Å)
    """
    # Ensure probability normalization
    dist_probs = dist_probs / (dist_probs.sum(axis=-1, keepdims=True) + 1e-8)

    # Calculate expected value: E[d] = Σ P(d_i) * d_i
    expected_dist = np.sum(dist_probs * DIST_BINS[None, None, :], axis=-1)

    return expected_dist.astype(np.float16)


def calculate_expected_angle(angle_probs: np.ndarray, bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate expected angle and convert to sin/cos encoding

    Args:
        angle_probs: (L, L, n_bins) angle probability distribution
        bins: (n_bins,) angle bin center values (unit: degree)

    Returns:
        sin_values: (L, L) sin(angle)
        cos_values: (L, L) cos(angle)
    """
    # Ensure probability normalization
    angle_probs = angle_probs / (angle_probs.sum(axis=-1, keepdims=True) + 1e-8)

    # Method 1: Directly use probability-weighted sin/cos vectors
    # This is mathematically more correct than calculating average angle then converting to sin/cos
    # (Avoids ambiguity of angle averaging)
    bins_rad = np.deg2rad(bins)
    sin_bins = np.sin(bins_rad)
    cos_bins = np.cos(bins_rad)

    # E[sin(θ)] = Σ P(θ_i) * sin(θ_i)
    sin_values = np.sum(angle_probs * sin_bins[None, None, :], axis=-1)
    # E[cos(θ)] = Σ P(θ_i) * cos(θ_i)
    cos_values = np.sum(angle_probs * cos_bins[None, None, :], axis=-1)

    # Normalize to unit circle (optional but recommended)
    magnitude = np.sqrt(sin_values**2 + cos_values**2) + 1e-8
    sin_values = sin_values / magnitude
    cos_values = cos_values / magnitude

    return sin_values.astype(np.float16), cos_values.astype(np.float16)


def distill_trrosetta_features(npz_file: Path) -> np.ndarray:
    """
    Distill trRosetta features

    Args:
        npz_file: trRosetta output .npz file

    Returns:
        distilled: (L, L, 7) compressed features
            [:, :, 0]: Distance (Å)
            [:, :, 1]: sin(omega)
            [:, :, 2]: cos(omega)
            [:, :, 3]: sin(theta)
            [:, :, 4]: cos(theta)
            [:, :, 5]: sin(phi)
            [:, :, 6]: cos(phi)
    """
    # Load original data
    data = np.load(npz_file)

    # Check required keys
    required_keys = ['dist', 'omega', 'theta', 'phi']
    for key in required_keys:
        if key not in data.files:
            raise KeyError(f"Missing required key: {key}")

    dist_probs = data['dist']
    omega_probs = data['omega']
    theta_probs = data['theta']
    phi_probs = data['phi']

    L = dist_probs.shape[0]

    # 1. Calculate expected distance
    expected_dist = calculate_expected_distance(dist_probs)  # (L, L)

    # 2. Calculate expected angle sin/cos
    sin_omega, cos_omega = calculate_expected_angle(omega_probs, OMEGA_BINS)  # (L, L)
    sin_theta, cos_theta = calculate_expected_angle(theta_probs, THETA_BINS)  # (L, L)
    sin_phi, cos_phi = calculate_expected_angle(phi_probs, PHI_BINS)  # (L, L)

    # 3. Stack into 7-channel features
    distilled = np.stack([
        expected_dist,
        sin_omega, cos_omega,
        sin_theta, cos_theta,
        sin_phi, cos_phi
    ], axis=-1)  # (L, L, 7)

    return distilled


def batch_distill(input_dir: Path, output_dir: Path, verbose: bool = True):
    """
    Batch distill trRosetta features

    Args:
        input_dir: Input directory containing .npz files
        output_dir: Output directory
        verbose: Whether to show progress
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = list(input_dir.glob("*.npz"))

    if verbose:
        print(f"Found {len(npz_files)} .npz files")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")

    stats = {
        'total': len(npz_files),
        'success': 0,
        'failed': 0,
        'total_original_size_mb': 0,
        'total_compressed_size_mb': 0
    }

    for i, npz_file in enumerate(npz_files, 1):
        try:
            if verbose and i % 100 == 0:
                print(f"Processing: {i}/{len(npz_files)}")

            # Get original file size
            original_size = npz_file.stat().st_size / (1024**2)  # MB

            # Distill
            distilled = distill_trrosetta_features(npz_file)

            # Save (using compression)
            output_file = output_dir / npz_file.name
            np.savez_compressed(output_file, features=distilled)

            # Get compressed file size
            compressed_size = output_file.stat().st_size / (1024**2)  # MB

            stats['success'] += 1
            stats['total_original_size_mb'] += original_size
            stats['total_compressed_size_mb'] += compressed_size

        except Exception as e:
            if verbose:
                print(f"❌ Processing failed: {npz_file.name} - {e}")
            stats['failed'] += 1

    # Show statistics
    if verbose:
        print(f"\n{'='*60}")
        print(f"Distillation completed!")
        print(f"{'='*60}")
        print(f"Success: {stats['success']}/{stats['total']}")
        print(f"Failed: {stats['failed']}/{stats['total']}")
        print(f"Total original size: {stats['total_original_size_mb']:.1f} MB "
              f"({stats['total_original_size_mb']/1024:.2f} GB)")
        print(f"Total compressed size: {stats['total_compressed_size_mb']:.1f} MB "
              f"({stats['total_compressed_size_mb']/1024:.2f} GB)")
        compression_ratio = (1 - stats['total_compressed_size_mb'] /
                            (stats['total_original_size_mb'] + 1e-8)) * 100
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"{'='*60}\n")

    # Save statistics
    with open(output_dir / 'distillation_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


def verify_distillation(original_file: Path, distilled_file: Path):
    """
    Verify the reasonableness of distillation results

    Compare original probability distribution and distilled deterministic values
    """
    # Load original data
    original = np.load(original_file)
    dist_probs = original['dist']

    # Load distilled data
    distilled_data = np.load(distilled_file)
    distilled = distilled_data['features']

    L = dist_probs.shape[0]

    print(f"\nVerifying file: {original_file.name}")
    print(f"Sequence length: {L}")
    print(f"\nOriginal data shape:")
    for key in original.files:
        print(f"  {key}: {original[key].shape}")
    print(f"\nDistilled data shape: {distilled.shape}")

    # Randomly select some residue pairs for comparison
    np.random.seed(42)
    sample_pairs = [(np.random.randint(0, L), np.random.randint(0, L)) for _ in range(3)]

    print(f"\nExample comparison (randomly selected 3 residue pairs):")
    print("="*60)

    for i, j in sample_pairs:
        print(f"\nResidue pair ({i}, {j}):")

        # Original distance distribution
        dist_prob = dist_probs[i, j]
        max_prob_idx = np.argmax(dist_prob)
        most_likely_dist = DIST_BINS[max_prob_idx]

        # Distilled distance
        expected_dist = distilled[i, j, 0]

        print(f"  Distance:")
        print(f"    Most likely distance (argmax): {most_likely_dist:.2f} Å")
        print(f"    Expected distance (distilled): {expected_dist:.2f} Å")

        # Angle example
        omega_prob = original['omega'][i, j]
        max_omega_idx = np.argmax(omega_prob)
        most_likely_omega = OMEGA_BINS[max_omega_idx]

        sin_omega = distilled[i, j, 1]
        cos_omega = distilled[i, j, 2]
        recovered_omega = np.rad2deg(np.arctan2(sin_omega, cos_omega))

        print(f"  Omega angle:")
        print(f"    Most likely angle (argmax): {most_likely_omega:.1f}°")
        print(f"    Distilled angle (arctan2): {recovered_omega:.1f}°")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='trRosetta Feature Distillation Tool')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory (containing trRosetta .npz files)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory (save distilled features)')
    parser.add_argument('--verify', type=str, default=None,
                       help='Verification mode: provide single .npz file path for verification')

    args = parser.parse_args()

    if args.verify:
        # Verification mode
        original_file = Path(args.verify)
        output_dir = Path(args.output)
        distilled_file = output_dir / original_file.name

        if not distilled_file.exists():
            print(f"Distilling this file first...")
            distilled = distill_trrosetta_features(original_file)
            np.savez_compressed(distilled_file, features=distilled)

        verify_distillation(original_file, distilled_file)
    else:
        # Batch distillation mode
        batch_distill(args.input, args.output, verbose=True)
