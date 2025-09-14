"""
Diagnose Siemens MRI data to understand the scale and units.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom


def examine_dicom_metadata():
    """Examine DICOM metadata to understand data scaling."""
    data_path = Path("/Users/alifatemi/Desktop/MRI phantom images")
    
    print("🔍 Examining DICOM Metadata")
    print("=" * 50)
    
    # Check magnitude data
    mag_files = list((data_path / "Mag TE10").glob("*.dcm"))
    if mag_files:
        mag_dicom = pydicom.dcmread(mag_files[0])
        print("📊 Magnitude Image Metadata:")
        print(f"   Pixel array shape: {mag_dicom.pixel_array.shape}")
        print(f"   Pixel array dtype: {mag_dicom.pixel_array.dtype}")
        print(f"   Pixel array range: {mag_dicom.pixel_array.min()} to {mag_dicom.pixel_array.max()}")
        
        # Check for scaling attributes
        scaling_attrs = ['RescaleSlope', 'RescaleIntercept', 'PixelIntensityRelationship', 
                        'PixelIntensityRelationshipSign', 'WindowCenter', 'WindowWidth']
        for attr in scaling_attrs:
            if hasattr(mag_dicom, attr):
                print(f"   {attr}: {getattr(mag_dicom, attr)}")
    
    # Check phase difference data
    phase_files = list((data_path / "Phase differences").glob("*.dcm"))
    if phase_files:
        phase_dicom = pydicom.dcmread(phase_files[0])
        print("\n📊 Phase Difference Metadata:")
        print(f"   Pixel array shape: {phase_dicom.pixel_array.shape}")
        print(f"   Pixel array dtype: {phase_dicom.pixel_array.dtype}")
        print(f"   Pixel array range: {phase_dicom.pixel_array.min()} to {phase_dicom.pixel_array.max()}")
        
        # Check for scaling attributes
        for attr in scaling_attrs:
            if hasattr(phase_dicom, attr):
                print(f"   {attr}: {getattr(phase_dicom, attr)}")
        
        # Check for phase-specific attributes
        phase_attrs = ['EchoTime', 'RepetitionTime', 'FlipAngle', 'MagneticFieldStrength']
        for attr in phase_attrs:
            if hasattr(phase_dicom, attr):
                print(f"   {attr}: {getattr(phase_dicom, attr)}")


def analyze_phase_scale():
    """Analyze the phase difference data to understand its scale."""
    data_path = Path("/Users/alifatemi/Desktop/MRI phantom images")
    
    print("\n🔬 Analyzing Phase Difference Scale")
    print("=" * 50)
    
    # Load a few phase difference slices
    phase_files = sorted(list((data_path / "Phase differences").glob("*.dcm")))
    
    if phase_files:
        # Load first few slices
        phase_data = []
        for i in range(min(5, len(phase_files))):
            dicom = pydicom.dcmread(phase_files[i])
            phase_data.append(dicom.pixel_array)
        
        phase_data = np.array(phase_data)
        
        print(f"📊 Phase difference data analysis:")
        print(f"   Shape: {phase_data.shape}")
        print(f"   Data type: {phase_data.dtype}")
        print(f"   Range: {phase_data.min()} to {phase_data.max()}")
        print(f"   Mean: {phase_data.mean():.2f}")
        print(f"   Std: {phase_data.std():.2f}")
        
        # Check if data is in radians, degrees, or other units
        print(f"\n🔍 Unit Analysis:")
        
        # If data is in degrees (0-360 range)
        if phase_data.max() > 100 and phase_data.min() >= 0:
            print(f"   📐 Data appears to be in DEGREES (0-360 range)")
            print(f"   🔄 Converting to radians: multiply by π/180")
            phase_rad = phase_data * np.pi / 180
            print(f"   📊 Radian range: {phase_rad.min():.3f} to {phase_rad.max():.3f}")
            return phase_rad, "degrees"
        
        # If data is in radians (-π to π range)
        elif phase_data.max() <= np.pi and phase_data.min() >= -np.pi:
            print(f"   📐 Data appears to be in RADIANS (-π to π range)")
            return phase_data, "radians"
        
        # If data is in scaled units (e.g., 0-4095 for 12-bit)
        elif phase_data.max() > 1000:
            print(f"   📐 Data appears to be in SCALED UNITS (0-{phase_data.max():.0f} range)")
            print(f"   🔄 Converting to radians: scale to -π to π")
            phase_rad = (phase_data - phase_data.max()/2) * 2 * np.pi / phase_data.max()
            print(f"   📊 Radian range: {phase_rad.min():.3f} to {phase_rad.max():.3f}")
            return phase_rad, "scaled"
        
        else:
            print(f"   ❓ Unknown scale - manual inspection needed")
            return phase_data, "unknown"


def recalculate_fieldmap_with_correct_scale():
    """Recalculate field map with correct phase scale."""
    print("\n🔄 Recalculating Field Map with Correct Scale")
    print("=" * 50)
    
    # Analyze phase scale
    phase_data, scale_type = analyze_phase_scale()
    
    if scale_type == "unknown":
        print("   ⚠️  Cannot determine phase scale automatically")
        return None
    
    # Load full phase difference volume
    data_path = Path("/Users/alifatemi/Desktop/MRI phantom images")
    phase_files = sorted(list((data_path / "Phase differences").glob("*.dcm")))
    
    print(f"📁 Loading {len(phase_files)} phase difference slices...")
    
    # Load all slices
    phase_volume = []
    for phase_file in phase_files:
        dicom = pydicom.dcmread(phase_file)
        phase_volume.append(dicom.pixel_array)
    
    phase_volume = np.array(phase_volume).transpose(1, 2, 0)  # (rows, cols, slices)
    
    print(f"   ✅ Phase volume shape: {phase_volume.shape}")
    print(f"   📊 Original range: {phase_volume.min()} to {phase_volume.max()}")
    
    # Convert to radians based on scale type
    if scale_type == "degrees":
        phase_rad = phase_volume * np.pi / 180
        print(f"   🔄 Converted from degrees to radians")
    elif scale_type == "scaled":
        phase_rad = (phase_volume - phase_volume.max()/2) * 2 * np.pi / phase_volume.max()
        print(f"   🔄 Converted from scaled units to radians")
    else:  # radians
        phase_rad = phase_volume
        print(f"   ✅ Data already in radians")
    
    print(f"   📊 Radian range: {phase_rad.min():.3f} to {phase_rad.max():.3f}")
    
    # Calculate field map with correct scale
    delta_te = 0.005  # 5ms
    fieldmap_hz = phase_rad / (2 * np.pi * delta_te)
    
    print(f"   ✅ B0 field map calculated")
    print(f"   📊 Field map range: {fieldmap_hz.min():.1f} to {fieldmap_hz.max():.1f} Hz")
    
    # Calculate distortion in mm
    echo_spacing = 0.0008  # 0.8ms
    voxel_size_y = 2.34375  # mm
    distortion_mm = fieldmap_hz * echo_spacing * voxel_size_y
    
    print(f"   📏 Distortion range: {distortion_mm.min():.2f} to {distortion_mm.max():.2f} mm")
    print(f"   📊 Mean absolute distortion: {np.mean(np.abs(distortion_mm)):.2f} mm")
    print(f"   📈 Max absolute distortion: {np.max(np.abs(distortion_mm)):.2f} mm")
    
    # Save corrected results
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)
    
    np.save(results_path / "corrected_b0_fieldmap_hz.npy", fieldmap_hz)
    np.save(results_path / "corrected_distortion_map_mm.npy", distortion_mm)
    
    print(f"   💾 Corrected results saved to {results_path}")
    
    return fieldmap_hz, distortion_mm


def create_corrected_visualization(fieldmap_hz, distortion_mm):
    """Create visualization with corrected data."""
    print("\n📊 Creating Corrected Visualization")
    print("=" * 50)
    
    # Select middle slice
    slice_idx = fieldmap_hz.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # B0 field map
    im1 = axes[0, 0].imshow(fieldmap_hz[:, :, slice_idx], cmap='RdBu_r')
    axes[0, 0].set_title(f'Corrected B0 Field Map (Hz)\nSlice {slice_idx}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Distortion map
    im2 = axes[0, 1].imshow(distortion_mm[:, :, slice_idx], cmap='RdBu_r')
    axes[0, 1].set_title(f'Corrected Distortion Map (mm)\nSlice {slice_idx}')
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
        axes[1, 0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}mm')
        axes[1, 0].axvline(mean_val + std_val, color='orange', linestyle='--', label=f'Mean+1σ: {mean_val+std_val:.2f}mm')
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
        distortion_values = distortion_mm[mask]
        stats_text = f"""
Corrected Siemens MRI Distortion Analysis

Distortion Statistics:
• Max: {np.max(np.abs(distortion_values)):.2f} mm
• Mean: {np.mean(np.abs(distortion_values)):.2f} mm
• Std: {np.std(distortion_values):.2f} mm
• 95th percentile: {np.percentile(np.abs(distortion_values), 95):.2f} mm

Severity Classification:
• Severe (>5mm): {np.sum(np.abs(distortion_values) > 5.0)/len(distortion_values)*100:.1f}%
• Moderate (2-5mm): {np.sum((np.abs(distortion_values) > 2.0) & (np.abs(distortion_values) <= 5.0))/len(distortion_values)*100:.1f}%
• Mild (1-2mm): {np.sum((np.abs(distortion_values) > 1.0) & (np.abs(distortion_values) <= 2.0))/len(distortion_values)*100:.1f}%
• Minimal (≤1mm): {np.sum(np.abs(distortion_values) <= 1.0)/len(distortion_values)*100:.1f}%

Scan Parameters:
• TE1: 10.0 ms
• TE2: 15.0 ms
• ΔTE: 5.0 ms
• Echo spacing: 0.8 ms
• Voxel size: 2.34×2.34×5.0 mm
        """
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('results/corrected_distortion_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Corrected visualization saved to 'results/corrected_distortion_analysis.png'")


def main():
    """Main diagnostic function."""
    print("🔍 Siemens MRI Data Diagnostic")
    print("=" * 60)
    
    try:
        # Examine DICOM metadata
        examine_dicom_metadata()
        
        # Analyze phase scale
        phase_data, scale_type = analyze_phase_scale()
        
        # Recalculate with correct scale
        corrected_results = recalculate_fieldmap_with_correct_scale()
        
        if corrected_results:
            fieldmap_hz, distortion_mm = corrected_results
            
            # Create corrected visualization
            create_corrected_visualization(fieldmap_hz, distortion_mm)
            
            print("\n" + "=" * 60)
            print("🎉 DIAGNOSTIC COMPLETE!")
            print("=" * 60)
            print("📁 Files generated:")
            print("   • results/corrected_b0_fieldmap_hz.npy - Corrected B0 field map")
            print("   • results/corrected_distortion_map_mm.npy - Corrected distortion map")
            print("   • results/corrected_distortion_analysis.png - Corrected visualization")
            
            # Final statistics
            mask = distortion_mm != 0
            if np.any(mask):
                distortion_values = distortion_mm[mask]
                print(f"\n🎯 Corrected Results:")
                print(f"   • Maximum distortion: {np.max(np.abs(distortion_values)):.2f} mm")
                print(f"   • Average distortion: {np.mean(np.abs(distortion_values)):.2f} mm")
                print(f"   • 95% of voxels have < {np.percentile(np.abs(distortion_values), 95):.2f} mm distortion")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


