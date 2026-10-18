"""
Generate diagnostic figures for splice boundary investigation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "02_preprocessing_qc" / "phase2_step3_splice_boundary_investigation.csv"
FIG_DIR = PROJECT_ROOT / "results" / "02_preprocessing_qc" / "figures"

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    if len(df) == 0:
        print("No D2-relevant splice events to plot")
        return

    print(f"Creating diagnostic figures for {len(df)} D2-relevant splice events")

    # Figure 1: Distribution of absolute discontinuities
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['abs_disc'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute Reflectance Discontinuity', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Splice Discontinuity Magnitudes\n(D2-relevant splice boundaries)', fontsize=13)
    ax.axvline(df['abs_disc'].median(), color='red', linestyle='--', label=f'Median: {df["abs_disc"].median():.5f}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'splice_discontinuity_distribution.png', dpi=150)
    print("  Created: splice_discontinuity_distribution.png")

    # Figure 2: Discontinuity vs local variation
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = df.dropna(subset=['abs_disc', 'disc_vs_variation'])
    scatter = ax.scatter(valid['local_variation'], valid['abs_disc'],
                        c=valid['disc_vs_variation'], cmap='viridis', s=80, edgecolor='black')
    ax.set_xlabel('Local Spectral Variation (±25 nm std)', fontsize=12)
    ax.set_ylabel('Absolute Discontinuity', fontsize=12)
    ax.set_title('Splice Discontinuity vs Local Spectral Variation', fontsize=13)
    ax.plot([0, valid['local_variation'].max()], [0, valid['local_variation'].max()],
           'r--', alpha=0.5, label='1:1 line')
    ax.legend()
    plt.colorbar(scatter, label='Discontinuity/Variation Ratio')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'splice_disc_vs_variation.png', dpi=150)
    print("  Created: splice_disc_vs_variation.png")

    # Figure 3: Splice wavelengths distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['splice_wl_nm'].dropna(), bins=15, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Splice Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of D2-Relevant Splice Wavelengths', fontsize=13)
    ax.axvline(320, color='green', linestyle=':', alpha=0.7, label='D2 boundaries')
    ax.axvline(2550, color='green', linestyle=':', alpha=0.7)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'splice_wavelength_distribution.png', dpi=150)
    print("  Created: splice_wavelength_distribution.png")

    # Figure 4: Mineral distribution of D2-relevant splices
    fig, ax = plt.subplots(figsize=(9, 5))
    mineral_counts = df['mineral'].value_counts()
    mineral_counts.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mineral', fontsize=12)
    ax.set_ylabel('Number of Splice Events', fontsize=12)
    ax.set_title('Mineral Distribution of D2-Relevant Splice Events', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'splice_mineral_distribution.png', dpi=150)
    print("  Created: splice_mineral_distribution.png")

    print("\nAll diagnostic figures created successfully")

if __name__ == "__main__":
    main()
