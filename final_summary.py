"""
Final summary of Siemens MRI distortion analysis.

This script provides a complete summary of the distortion mapping results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_corrected_results():
    """Load the corrected analysis results."""
    results_path = Path("results")
    
    print("📊 Siemens MRI Distortion Analysis - Final Summary")
    print("=" * 70)
    
    # Load corrected data
    fieldmap_hz = np.load(results_path / "corrected_b0_fieldmap_hz.npy")
    distortion_mm = np.load(results_path / "corrected_distortion_map_mm.npy")
    metadata = np.load(results_path / "metadata.npy", allow_pickle=True).item()
    
    print(f"✅ Corrected data loaded successfully!")
    print(f"   📏 Data shape: {fieldmap_hz.shape}")
    print(f"   ⏱️  TE1: {metadata['te1_ms']:.1f} ms")
    print(f"   ⏱️  TE2: {metadata['te2_ms']:.1f} ms")
    print(f"   ⏱️  ΔTE: {metadata['delta_te_ms']:.1f} ms")
    print(f"   ⏱️  Echo spacing: {metadata['echo_spacing_ms']:.1f} ms")
    print(f"   📏 Voxel size: {metadata['voxel_size_mm']}")
    print(f"   🧲 Field strength: 1.5T (from DICOM)")
    
    return fieldmap_hz, distortion_mm, metadata


def comprehensive_analysis(fieldmap_hz, distortion_mm, metadata):
    """Perform comprehensive distortion analysis."""
    print("\n📈 Comprehensive Distortion Analysis")
    print("=" * 50)
    
    # Create mask for analysis
    mask = distortion_mm != 0
    
    if not np.any(mask):
        print("   ⚠️  No distortion data found")
        return None
    
    distortion_values = distortion_mm[mask]
    fieldmap_values = fieldmap_hz[mask]
    
    # Basic statistics
    print(f"📊 Basic Statistics:")
    print(f"   B0 field map range: {np.min(fieldmap_values):.1f} to {np.max(fieldmap_values):.1f} Hz")
    print(f"   B0 field map mean: {np.mean(fieldmap_values):.1f} Hz")
    print(f"   B0 field map std: {np.std(fieldmap_values):.1f} Hz")
    print(f"   ")
    print(f"   Distortion range: {np.min(distortion_values):.2f} to {np.max(distortion_values):.2f} mm")
    print(f"   Mean absolute distortion: {np.mean(np.abs(distortion_values)):.2f} mm")
    print(f"   Std absolute distortion: {np.std(np.abs(distortion_values)):.2f} mm")
    print(f"   Max absolute distortion: {np.max(np.abs(distortion_values)):.2f} mm")
    
    # Percentile analysis
    print(f"\n📊 Percentile Analysis:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(np.abs(distortion_values), p)
        print(f"   {p}th percentile: {value:.2f} mm")
    
    # Severity classification
    print(f"\n🎯 Distortion Severity Classification:")
    severe = np.sum(np.abs(distortion_values) > 5.0)
    moderate = np.sum((np.abs(distortion_values) > 2.0) & (np.abs(distortion_values) <= 5.0))
    mild = np.sum((np.abs(distortion_values) > 1.0) & (np.abs(distortion_values) <= 2.0))
    minimal = np.sum((np.abs(distortion_values) > 0.0) & (np.abs(distortion_values) <= 1.0))
    total = len(distortion_values)
    
    print(f"   Severe (>5mm): {severe} voxels ({severe/total*100:.1f}%)")
    print(f"   Moderate (2-5mm): {moderate} voxels ({moderate/total*100:.1f}%)")
    print(f"   Mild (1-2mm): {mild} voxels ({mild/total*100:.1f}%)")
    print(f"   Minimal (≤1mm): {minimal} voxels ({minimal/total*100:.1f}%)")
    
    # Clinical significance
    print(f"\n🏥 Clinical Significance:")
    if np.max(np.abs(distortion_values)) > 3.0:
        print(f"   ⚠️  Maximum distortion ({np.max(np.abs(distortion_values)):.2f} mm) exceeds typical clinical threshold (3mm)")
    else:
        print(f"   ✅ Maximum distortion ({np.max(np.abs(distortion_values)):.2f} mm) within acceptable clinical range")
    
    if np.mean(np.abs(distortion_values)) > 2.0:
        print(f"   ⚠️  Mean distortion ({np.mean(np.abs(distortion_values)):.2f} mm) is high")
    else:
        print(f"   ✅ Mean distortion ({np.mean(np.abs(distortion_values)):.2f} mm) is acceptable")
    
    # Volume analysis
    voxel_volume_mm3 = np.prod(metadata['voxel_size_mm'])
    total_volume_cm3 = total * voxel_volume_mm3 / 1000
    severe_volume_cm3 = severe * voxel_volume_mm3 / 1000
    
    print(f"\n📏 Volume Analysis:")
    print(f"   Total analyzed volume: {total_volume_cm3:.1f} cm³")
    print(f"   Severely distorted volume: {severe_volume_cm3:.1f} cm³ ({severe_volume_cm3/total_volume_cm3*100:.1f}%)")
    
    return {
        'fieldmap_range': (np.min(fieldmap_values), np.max(fieldmap_values)),
        'fieldmap_mean': np.mean(fieldmap_values),
        'fieldmap_std': np.std(fieldmap_values),
        'distortion_range': (np.min(distortion_values), np.max(distortion_values)),
        'distortion_mean': np.mean(np.abs(distortion_values)),
        'distortion_std': np.std(np.abs(distortion_values)),
        'distortion_max': np.max(np.abs(distortion_values)),
        'percentiles': {p: np.percentile(np.abs(distortion_values), p) for p in percentiles},
        'severity_counts': {'severe': severe, 'moderate': moderate, 'mild': mild, 'minimal': minimal},
        'severity_percentages': {'severe': severe/total*100, 'moderate': moderate/total*100, 
                                'mild': mild/total*100, 'minimal': minimal/total*100},
        'total_voxels': total,
        'total_volume_cm3': total_volume_cm3,
        'severe_volume_cm3': severe_volume_cm3
    }


def create_final_visualization(fieldmap_hz, distortion_mm, metadata, stats):
    """Create final comprehensive visualization."""
    print("\n📊 Creating Final Visualization")
    print("=" * 50)
    
    # Select multiple slices for visualization
    n_slices = fieldmap_hz.shape[2]
    slice_indices = [n_slices//4, n_slices//2, 3*n_slices//4]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i, slice_idx in enumerate(slice_indices):
        # B0 field map
        im1 = axes[i, 0].imshow(fieldmap_hz[:, :, slice_idx], cmap='RdBu_r')
        axes[i, 0].set_title(f'B0 Field Map (Hz)\nSlice {slice_idx}')
        axes[i, 0].axis('off')
        if i == 0:
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Distortion map
        im2 = axes[i, 1].imshow(distortion_mm[:, :, slice_idx], cmap='RdBu_r')
        axes[i, 1].set_title(f'Distortion Map (mm)\nSlice {slice_idx}')
        axes[i, 1].axis('off')
        if i == 0:
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Absolute distortion
        im3 = axes[i, 2].imshow(np.abs(distortion_mm[:, :, slice_idx]), cmap='hot')
        axes[i, 2].set_title(f'Absolute Distortion (mm)\nSlice {slice_idx}')
        axes[i, 2].axis('off')
        if i == 0:
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # Distortion severity
        severity_map = np.zeros_like(distortion_mm[:, :, slice_idx])
        slice_distortion = np.abs(distortion_mm[:, :, slice_idx])
        severity_map[slice_distortion > 5.0] = 4  # Severe
        severity_map[(slice_distortion > 2.0) & (slice_distortion <= 5.0)] = 3  # Moderate
        severity_map[(slice_distortion > 1.0) & (slice_distortion <= 2.0)] = 2  # Mild
        severity_map[(slice_distortion > 0.0) & (slice_distortion <= 1.0)] = 1  # Minimal
        
        im4 = axes[i, 3].imshow(severity_map, cmap='RdYlGn', vmin=0, vmax=4)
        axes[i, 3].set_title(f'Distortion Severity\nSlice {slice_idx}')
        axes[i, 3].axis('off')
        if i == 0:
            cbar = plt.colorbar(im4, ax=axes[i, 3], fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 1, 2, 3, 4])
            cbar.set_ticklabels(['None', 'Minimal', 'Mild', 'Moderate', 'Severe'])
    
    plt.tight_layout()
    plt.savefig('results/final_distortion_summary.png', dpi=300, bbox_inches='tight')
    print("   ✅ Final visualization saved to 'results/final_distortion_summary.png'")


def generate_report(fieldmap_hz, distortion_mm, metadata, stats):
    """Generate a comprehensive text report."""
    print("\n📝 Generating Comprehensive Report")
    print("=" * 50)
    
    report = f"""
Siemens MRI Geometrical Distortion Analysis Report
==================================================

Scan Parameters:
- Scanner: Siemens 1.5T
- TE1: {metadata['te1_ms']:.1f} ms
- TE2: {metadata['te2_ms']:.1f} ms
- ΔTE: {metadata['delta_te_ms']:.1f} ms
- Echo spacing: {metadata['echo_spacing_ms']:.1f} ms
- Voxel size: {metadata['voxel_size_mm'][0]:.2f} × {metadata['voxel_size_mm'][1]:.2f} × {metadata['voxel_size_mm'][2]:.1f} mm
- Matrix size: {metadata['data_shape'][0]} × {metadata['data_shape'][1]} × {metadata['data_shape'][2]}

B0 Field Map Analysis:
- Range: {stats['fieldmap_range'][0]:.1f} to {stats['fieldmap_range'][1]:.1f} Hz
- Mean: {stats['fieldmap_mean']:.1f} Hz
- Standard deviation: {stats['fieldmap_std']:.1f} Hz

Geometrical Distortion Analysis:
- Range: {stats['distortion_range'][0]:.2f} to {stats['distortion_range'][1]:.2f} mm
- Mean absolute distortion: {stats['distortion_mean']:.2f} mm
- Standard deviation: {stats['distortion_std']:.2f} mm
- Maximum absolute distortion: {stats['distortion_max']:.2f} mm

Percentile Analysis:
- 50th percentile: {stats['percentiles'][50]:.2f} mm
- 75th percentile: {stats['percentiles'][75]:.2f} mm
- 90th percentile: {stats['percentiles'][90]:.2f} mm
- 95th percentile: {stats['percentiles'][95]:.2f} mm
- 99th percentile: {stats['percentiles'][99]:.2f} mm

Distortion Severity Classification:
- Severe (>5mm): {stats['severity_counts']['severe']} voxels ({stats['severity_percentages']['severe']:.1f}%)
- Moderate (2-5mm): {stats['severity_counts']['moderate']} voxels ({stats['severity_percentages']['moderate']:.1f}%)
- Mild (1-2mm): {stats['severity_counts']['mild']} voxels ({stats['severity_percentages']['mild']:.1f}%)
- Minimal (≤1mm): {stats['severity_counts']['minimal']} voxels ({stats['severity_percentages']['minimal']:.1f}%)

Volume Analysis:
- Total analyzed volume: {stats['total_volume_cm3']:.1f} cm³
- Severely distorted volume: {stats['severe_volume_cm3']:.1f} cm³ ({stats['severe_volume_cm3']/stats['total_volume_cm3']*100:.1f}%)

Clinical Assessment:
- Maximum distortion: {stats['distortion_max']:.2f} mm
- Mean distortion: {stats['distortion_mean']:.2f} mm
- 95% of voxels have distortion < {stats['percentiles'][95]:.2f} mm

Recommendations:
- {'⚠️  Consider distortion correction for clinical applications' if stats['distortion_max'] > 3.0 else '✅ Distortion levels are acceptable for clinical use'}
- {'⚠️  High mean distortion may affect quantitative measurements' if stats['distortion_mean'] > 2.0 else '✅ Mean distortion is within acceptable range'}

Files Generated:
- corrected_b0_fieldmap_hz.npy: B0 field map in Hz
- corrected_distortion_map_mm.npy: Geometrical distortion map in mm
- final_distortion_summary.png: Comprehensive visualization
- This report: final_analysis_report.txt

Analysis completed using custom B0 field map processing utilities.
"""
    
    # Save report
    with open('results/final_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("   ✅ Report saved to 'results/final_analysis_report.txt'")
    print("\n" + "="*70)
    print(report)
    print("="*70)


def main():
    """Main function for final summary."""
    try:
        # Load corrected results
        fieldmap_hz, distortion_mm, metadata = load_corrected_results()
        
        # Perform comprehensive analysis
        stats = comprehensive_analysis(fieldmap_hz, distortion_mm, metadata)
        
        if stats:
            # Create final visualization
            create_final_visualization(fieldmap_hz, distortion_mm, metadata, stats)
            
            # Generate report
            generate_report(fieldmap_hz, distortion_mm, metadata, stats)
            
            print("\n" + "=" * 70)
            print("🎉 FINAL ANALYSIS COMPLETE!")
            print("=" * 70)
            print("📁 All results saved to 'results/' folder:")
            print("   • corrected_b0_fieldmap_hz.npy - B0 field map data")
            print("   • corrected_distortion_map_mm.npy - Distortion map in mm")
            print("   • final_distortion_summary.png - Multi-slice visualization")
            print("   • final_analysis_report.txt - Comprehensive report")
            
            print(f"\n🎯 Key Results:")
            print(f"   • Maximum distortion: {stats['distortion_max']:.2f} mm")
            print(f"   • Mean distortion: {stats['distortion_mean']:.2f} mm")
            print(f"   • 95% of voxels have < {stats['percentiles'][95]:.2f} mm distortion")
            print(f"   • {stats['severity_percentages']['severe']:.1f}% of volume has severe distortion")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


