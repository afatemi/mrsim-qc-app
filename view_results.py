"""
View and analyze the Siemens MRI distortion results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results():
    """Load the analysis results."""
    results_path = Path("results")
    
    print("📊 Loading Siemens MRI Distortion Results")
    print("=" * 50)
    
    # Load data
    fieldmap_hz = np.load(results_path / "b0_fieldmap_hz.npy")
    distortion_mm = np.load(results_path / "distortion_map_mm.npy")
    metadata = np.load(results_path / "metadata.npy", allow_pickle=True).item()
    
    print(f"✅ Data loaded successfully!")
    print(f"   📏 Data shape: {fieldmap_hz.shape}")
    print(f"   ⏱️  TE1: {metadata['te1_ms']:.1f} ms")
    print(f"   ⏱️  TE2: {metadata['te2_ms']:.1f} ms")
    print(f"   ⏱️  ΔTE: {metadata['delta_te_ms']:.1f} ms")
    print(f"   ⏱️  Echo spacing: {metadata['echo_spacing_ms']:.1f} ms")
    print(f"   📏 Voxel size: {metadata['voxel_size_mm']}")
    
    return fieldmap_hz, distortion_mm, metadata


def analyze_distortion(distortion_mm, metadata):
    """Analyze distortion statistics."""
    print("\n📈 Distortion Analysis")
    print("=" * 30)
    
    # Create a simple mask (non-zero values)
    mask = distortion_mm != 0
    
    if np.any(mask):
        distortion_values = distortion_mm[mask]
        
        print(f"📊 Distortion Statistics:")
        print(f"   Max displacement: {np.max(np.abs(distortion_values)):.2f} mm")
        print(f"   Mean displacement: {np.mean(np.abs(distortion_values)):.2f} mm")
        print(f"   Std displacement: {np.std(distortion_values):.2f} mm")
        print(f"   95th percentile: {np.percentile(np.abs(distortion_values), 95):.2f} mm")
        print(f"   99th percentile: {np.percentile(np.abs(distortion_values), 99):.2f} mm")
        
        # Calculate distortion severity
        severe_distortion = np.sum(np.abs(distortion_values) > 5.0)  # > 5mm
        moderate_distortion = np.sum((np.abs(distortion_values) > 2.0) & (np.abs(distortion_values) <= 5.0))
        mild_distortion = np.sum((np.abs(distortion_values) > 1.0) & (np.abs(distortion_values) <= 2.0))
        minimal_distortion = np.sum(np.abs(distortion_values) <= 1.0)
        
        total_voxels = len(distortion_values)
        
        print(f"\n🎯 Distortion Severity Classification:")
        print(f"   Severe (>5mm): {severe_distortion} voxels ({severe_distortion/total_voxels*100:.1f}%)")
        print(f"   Moderate (2-5mm): {moderate_distortion} voxels ({moderate_distortion/total_voxels*100:.1f}%)")
        print(f"   Mild (1-2mm): {mild_distortion} voxels ({mild_distortion/total_voxels*100:.1f}%)")
        print(f"   Minimal (≤1mm): {minimal_distortion} voxels ({minimal_distortion/total_voxels*100:.1f}%)")
        
        return {
            'max': np.max(np.abs(distortion_values)),
            'mean': np.mean(np.abs(distortion_values)),
            'std': np.std(distortion_values),
            'p95': np.percentile(np.abs(distortion_values), 95),
            'p99': np.percentile(np.abs(distortion_values), 99),
            'severe': severe_distortion,
            'moderate': moderate_distortion,
            'mild': mild_distortion,
            'minimal': minimal_distortion
        }
    else:
        print("   ⚠️  No distortion data found")
        return None


def create_detailed_visualization(fieldmap_hz, distortion_mm, metadata):
    """Create detailed visualization of results."""
    print("\n📊 Creating detailed visualization...")
    
    # Select middle slice
    slice_idx = fieldmap_hz.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # B0 field map
    im1 = axes[0, 0].imshow(fieldmap_hz[:, :, slice_idx], cmap='RdBu_r')
    axes[0, 0].set_title(f'B0 Field Map (Hz)\nSlice {slice_idx}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Distortion map
    im2 = axes[0, 1].imshow(distortion_mm[:, :, slice_idx], cmap='RdBu_r')
    axes[0, 1].set_title(f'Distortion Map (mm)\nSlice {slice_idx}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Absolute distortion
    im3 = axes[0, 2].imshow(np.abs(distortion_mm[:, :, slice_idx]), cmap='hot')
    axes[0, 2].set_title(f'Absolute Distortion (mm)\nSlice {slice_idx}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Histogram of distortion
    mask = distortion_mm != 0
    if np.any(mask):
        distortion_values = distortion_mm[mask]
        axes[1, 0].hist(distortion_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_title('Distortion Distribution')
        axes[1, 0].set_xlabel('Displacement (mm)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics lines
        mean_val = np.mean(np.abs(distortion_values))
        std_val = np.std(distortion_values)
        axes[1, 0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}mm')
        axes[1, 0].axvline(mean_val + std_val, color='orange', linestyle='--', label=f'Mean+1σ: {mean_val+std_val:.1f}mm')
        axes[1, 0].legend()
    
    # Distortion severity map
    severity_map = np.zeros_like(distortion_mm)
    severity_map[np.abs(distortion_mm) > 5.0] = 4  # Severe
    severity_map[(np.abs(distortion_mm) > 2.0) & (np.abs(distortion_mm) <= 5.0)] = 3  # Moderate
    severity_map[(np.abs(distortion_mm) > 1.0) & (np.abs(distortion_mm) <= 2.0)] = 2  # Mild
    severity_map[(np.abs(distortion_mm) > 0.0) & (np.abs(distortion_mm) <= 1.0)] = 1  # Minimal
    
    im4 = axes[1, 1].imshow(severity_map[:, :, slice_idx], cmap='RdYlGn', vmin=0, vmax=4)
    axes[1, 1].set_title(f'Distortion Severity\nSlice {slice_idx}')
    axes[1, 1].axis('off')
    cbar = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['None', 'Minimal', 'Mild', 'Moderate', 'Severe'])
    
    # Summary statistics
    axes[1, 2].axis('off')
    if np.any(mask):
        stats = analyze_distortion(distortion_mm, metadata)
        if stats:
            stats_text = f"""
Siemens MRI Distortion Analysis

Scan Parameters:
• TE1: {metadata['te1_ms']:.1f} ms
• TE2: {metadata['te2_ms']:.1f} ms
• ΔTE: {metadata['delta_te_ms']:.1f} ms
• Echo spacing: {metadata['echo_spacing_ms']:.1f} ms
• Voxel size: {metadata['voxel_size_mm'][0]:.1f}×{metadata['voxel_size_mm'][1]:.1f}×{metadata['voxel_size_mm'][2]:.1f} mm

Distortion Statistics:
• Max: {stats['max']:.2f} mm
• Mean: {stats['mean']:.2f} mm
• Std: {stats['std']:.2f} mm
• 95th percentile: {stats['p95']:.2f} mm
• 99th percentile: {stats['p99']:.2f} mm

Severity Classification:
• Severe (>5mm): {stats['severe']/np.sum(mask)*100:.1f}%
• Moderate (2-5mm): {stats['moderate']/np.sum(mask)*100:.1f}%
• Mild (1-2mm): {stats['mild']/np.sum(mask)*100:.1f}%
• Minimal (≤1mm): {stats['minimal']/np.sum(mask)*100:.1f}%
            """
            axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('results/detailed_distortion_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Detailed visualization saved to 'results/detailed_distortion_analysis.png'")
    
    return fig


def main():
    """Main function to view and analyze results."""
    try:
        # Load results
        fieldmap_hz, distortion_mm, metadata = load_results()
        
        # Analyze distortion
        stats = analyze_distortion(distortion_mm, metadata)
        
        # Create detailed visualization
        create_detailed_visualization(fieldmap_hz, distortion_mm, metadata)
        
        print("\n" + "=" * 60)
        print("🎉 RESULTS ANALYSIS COMPLETE!")
        print("=" * 60)
        print("📁 Files generated:")
        print("   • results/detailed_distortion_analysis.png - Detailed visualization")
        print("   • results/siemens_distortion_analysis.png - Original analysis")
        print("   • results/b0_fieldmap_hz.npy - B0 field map data")
        print("   • results/distortion_map_mm.npy - Distortion map data")
        print("   • results/metadata.npy - Scan parameters")
        
        if stats:
            print(f"\n🎯 Key Findings:")
            print(f"   • Maximum distortion: {stats['max']:.2f} mm")
            print(f"   • Average distortion: {stats['mean']:.2f} mm")
            print(f"   • 95% of voxels have < {stats['p95']:.2f} mm distortion")
            
            if stats['severe'] > 0:
                print(f"   ⚠️  {stats['severe']/np.sum(distortion_mm != 0)*100:.1f}% of voxels have severe distortion (>5mm)")
            else:
                print(f"   ✅ No severe distortion detected")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


