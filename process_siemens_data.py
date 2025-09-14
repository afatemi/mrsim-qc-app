"""
Process Siemens MRI data to generate geometrical distortion map in mm.

This script processes:
- Magnitude images at TE=10ms and TE=15ms
- Phase difference data
- Generates B0 field map and distortion map in mm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Import our b0map utilities
from mri_utils import (
    estimate_b0_fieldmap,
    assess_fieldmap_quality,
    smooth_fieldmap,
    correct_geometric_distortion,
    assess_distortion_severity
)


class SiemensMRIDataProcessor:
    """Process Siemens MRI data for distortion mapping."""
    
    def __init__(self, data_path):
        """
        Initialize processor with data path.
        
        Parameters:
        -----------
        data_path : str
            Path to the folder containing MRI data
        """
        self.data_path = Path(data_path)
        self.te1 = 0.010  # 10ms in seconds
        self.te2 = 0.015  # 15ms in seconds
        self.delta_te = self.te2 - self.te1  # 5ms
        
        # Initialize data containers
        self.mag_te10 = None
        self.mag_te15 = None
        self.phase_diff = None
        self.fieldmap_hz = None
        self.distortion_map_mm = None
        self.voxel_size = None
        self.echo_spacing = None
        
    def load_dicom_series(self, folder_name):
        """
        Load DICOM series from folder.
        
        Parameters:
        -----------
        folder_name : str
            Name of the folder containing DICOM files
            
        Returns:
        --------
        volume : np.ndarray
            3D volume of the DICOM series
        """
        folder_path = self.data_path / folder_name
        dicom_files = sorted(list(folder_path.glob("*.dcm")))
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        print(f"📁 Loading {len(dicom_files)} DICOM files from {folder_name}...")
        
        # Load first DICOM to get dimensions and metadata
        first_dicom = pydicom.dcmread(dicom_files[0])
        rows = first_dicom.Rows
        cols = first_dicom.Columns
        slices = len(dicom_files)
        
        # Initialize volume
        volume = np.zeros((rows, cols, slices), dtype=np.float32)
        
        # Load all slices
        for i, dicom_file in enumerate(dicom_files):
            dicom_data = pydicom.dcmread(dicom_file)
            volume[:, :, i] = dicom_data.pixel_array.astype(np.float32)
        
        # Get voxel size from DICOM metadata
        if hasattr(first_dicom, 'PixelSpacing'):
            self.voxel_size = (
                float(first_dicom.PixelSpacing[0]),  # x
                float(first_dicom.PixelSpacing[1]),  # y
                float(first_dicom.SliceThickness) if hasattr(first_dicom, 'SliceThickness') else 1.0  # z
            )
        else:
            self.voxel_size = (1.0, 1.0, 1.0)  # Default 1mm isotropic
        
        # Get echo spacing if available
        if hasattr(first_dicom, 'EchoSpacing'):
            self.echo_spacing = float(first_dicom.EchoSpacing) / 1000.0  # Convert to seconds
        else:
            # Estimate echo spacing (typical for Siemens)
            self.echo_spacing = 0.0008  # 0.8ms default
        
        print(f"   ✅ Volume shape: {volume.shape}")
        print(f"   📏 Voxel size: {self.voxel_size[0]:.2f} x {self.voxel_size[1]:.2f} x {self.voxel_size[2]:.2f} mm")
        print(f"   ⏱️  Echo spacing: {self.echo_spacing*1000:.1f} ms")
        
        return volume
    
    def load_all_data(self):
        """Load all MRI data (magnitude and phase)."""
        print("🚀 Loading Siemens MRI Data")
        print("=" * 50)
        
        # Load magnitude images
        self.mag_te10 = self.load_dicom_series("Mag TE10")
        self.mag_te15 = self.load_dicom_series("Mag TE 15")
        
        # Load phase difference
        self.phase_diff = self.load_dicom_series("Phase differences")
        
        print(f"✅ All data loaded successfully!")
        print(f"   📊 Data shapes: {self.mag_te10.shape}")
        
    def create_brain_mask(self, threshold_percentile=10):
        """
        Create brain mask from magnitude images.
        
        Parameters:
        -----------
        threshold_percentile : float
            Percentile threshold for mask creation
            
        Returns:
        --------
        mask : np.ndarray
            Boolean brain mask
        """
        print(f"🧠 Creating brain mask (threshold: {threshold_percentile}th percentile)...")
        
        # Use average of both magnitude images
        avg_magnitude = (self.mag_te10 + self.mag_te15) / 2
        
        # Calculate threshold
        threshold = np.percentile(avg_magnitude, threshold_percentile)
        
        # Create mask
        mask = avg_magnitude > threshold
        
        # Clean up mask with morphological operations
        from scipy.ndimage import binary_opening, binary_closing
        mask = binary_opening(mask, structure=np.ones((3,3,3)))
        mask = binary_closing(mask, structure=np.ones((3,3,3)))
        
        brain_volume = np.sum(mask) * np.prod(self.voxel_size) / 1000  # cm³
        print(f"   ✅ Brain mask created")
        print(f"   📏 Brain volume: {brain_volume:.1f} cm³")
        print(f"   🎯 Mask coverage: {np.sum(mask)/mask.size*100:.1f}%")
        
        return mask
    
    def estimate_fieldmap_from_phase_diff(self, mask=None):
        """
        Estimate B0 field map from phase difference data.
        
        Parameters:
        -----------
        mask : np.ndarray, optional
            Brain mask to limit processing
            
        Returns:
        --------
        fieldmap_hz : np.ndarray
            B0 field map in Hz
        """
        print("🔬 Estimating B0 field map from phase difference...")
        
        if mask is None:
            mask = self.create_brain_mask()
        
        # Convert phase difference to field map
        # Phase difference = 2π * ΔB0 * ΔTE
        # Therefore: ΔB0 = phase_diff / (2π * ΔTE)
        self.fieldmap_hz = self.phase_diff / (2 * np.pi * self.delta_te)
        
        # Apply mask
        self.fieldmap_hz = self.fieldmap_hz * mask
        
        # Assess quality
        quality_metrics = assess_fieldmap_quality(self.fieldmap_hz, mask)
        
        print(f"   ✅ B0 field map estimated")
        print(f"   📊 Field map range: {np.min(self.fieldmap_hz[mask]):.1f} to {np.max(self.fieldmap_hz[mask]):.1f} Hz")
        print(f"   📈 Mean: {quality_metrics['mean']:.2f} Hz")
        print(f"   📊 Std: {quality_metrics['std']:.2f} Hz")
        print(f"   🎯 SNR: {quality_metrics['snr']:.2f}")
        
        return self.fieldmap_hz
    
    def estimate_fieldmap_from_magnitude(self, mask=None):
        """
        Estimate B0 field map from magnitude images (alternative method).
        
        Parameters:
        -----------
        mask : np.ndarray, optional
            Brain mask to limit processing
            
        Returns:
        --------
        fieldmap_hz : np.ndarray
            B0 field map in Hz
        """
        print("🔬 Estimating B0 field map from magnitude images...")
        
        if mask is None:
            mask = self.create_brain_mask()
        
        # Create complex images (assuming phase is zero for magnitude-only data)
        echo1 = self.mag_te10.astype(complex)
        echo2 = self.mag_te15.astype(complex)
        
        # Estimate field map using our utility
        self.fieldmap_hz = estimate_b0_fieldmap(echo1, echo2, self.te1, self.te2, mask)
        
        # Assess quality
        quality_metrics = assess_fieldmap_quality(self.fieldmap_hz, mask)
        
        print(f"   ✅ B0 field map estimated from magnitude")
        print(f"   📊 Field map range: {np.min(self.fieldmap_hz[mask]):.1f} to {np.max(self.fieldmap_hz[mask]):.1f} Hz")
        print(f"   📈 Mean: {quality_metrics['mean']:.2f} Hz")
        print(f"   📊 Std: {quality_metrics['std']:.2f} Hz")
        print(f"   🎯 SNR: {quality_metrics['snr']:.2f}")
        
        return self.fieldmap_hz
    
    def calculate_distortion_map_mm(self, mask=None, phase_encode_direction='y'):
        """
        Calculate geometrical distortion map in mm.
        
        Parameters:
        -----------
        mask : np.ndarray, optional
            Brain mask to limit processing
        phase_encode_direction : str
            Phase encoding direction ('x', 'y', or 'z')
            
        Returns:
        --------
        distortion_map_mm : np.ndarray
            Distortion map in mm
        """
        print("📏 Calculating geometrical distortion map in mm...")
        
        if mask is None:
            mask = self.create_brain_mask()
        
        if self.fieldmap_hz is None:
            print("   ⚠️  No field map available. Estimating from phase difference...")
            self.estimate_fieldmap_from_phase_diff(mask)
        
        # Calculate displacement in pixels
        displacement_pixels = self.fieldmap_hz * self.echo_spacing
        
        # Convert to mm
        pe_axis = {'x': 0, 'y': 1, 'z': 2}[phase_encode_direction]
        self.distortion_map_mm = displacement_pixels * self.voxel_size[pe_axis]
        
        # Apply mask
        self.distortion_map_mm = self.distortion_map_mm * mask
        
        # Assess distortion severity
        severity_metrics = assess_distortion_severity(
            self.fieldmap_hz, self.echo_spacing, self.voxel_size, phase_encode_direction
        )
        
        print(f"   ✅ Distortion map calculated")
        print(f"   📏 Max displacement: {np.max(np.abs(self.distortion_map_mm[mask])):.2f} mm")
        print(f"   📊 Mean displacement: {np.mean(np.abs(self.distortion_map_mm[mask])):.2f} mm")
        print(f"   📈 Std displacement: {np.std(self.distortion_map_mm[mask]):.2f} mm")
        print(f"   🎯 Distortion volume fraction: {severity_metrics['distortion_volume_fraction']:.3f}")
        
        return self.distortion_map_mm
    
    def smooth_fieldmap(self, sigma=1.0, mask=None):
        """Smooth the B0 field map."""
        if self.fieldmap_hz is None:
            raise ValueError("No field map available. Run estimate_fieldmap first.")
        
        if mask is None:
            mask = self.create_brain_mask()
        
        print(f"🔧 Smoothing field map (σ={sigma})...")
        self.fieldmap_hz = smooth_fieldmap(self.fieldmap_hz, sigma=sigma, mask=mask)
        print("   ✅ Field map smoothed")
        
        return self.fieldmap_hz
    
    def save_results(self, output_dir="results"):
        """Save all results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"💾 Saving results to {output_path}...")
        
        # Save field map
        if self.fieldmap_hz is not None:
            np.save(output_path / "b0_fieldmap_hz.npy", self.fieldmap_hz)
            print("   ✅ B0 field map saved")
        
        # Save distortion map
        if self.distortion_map_mm is not None:
            np.save(output_path / "distortion_map_mm.npy", self.distortion_map_mm)
            print("   ✅ Distortion map saved")
        
        # Save metadata
        metadata = {
            'te1_ms': self.te1 * 1000,
            'te2_ms': self.te2 * 1000,
            'delta_te_ms': self.delta_te * 1000,
            'echo_spacing_ms': self.echo_spacing * 1000,
            'voxel_size_mm': self.voxel_size,
            'data_shape': self.mag_te10.shape
        }
        
        np.save(output_path / "metadata.npy", metadata)
        print("   ✅ Metadata saved")
        
        print(f"   📁 All results saved to {output_path}")
    
    def create_visualization(self, slice_idx=None, output_dir="results"):
        """Create comprehensive visualization."""
        if slice_idx is None:
            slice_idx = self.mag_te10.shape[2] // 2  # Middle slice
        
        print(f"📊 Creating visualization (slice {slice_idx})...")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Magnitude images
        axes[0, 0].imshow(self.mag_te10[:, :, slice_idx], cmap='gray')
        axes[0, 0].set_title(f'Magnitude TE=10ms (Slice {slice_idx})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.mag_te15[:, :, slice_idx], cmap='gray')
        axes[0, 1].set_title(f'Magnitude TE=15ms (Slice {slice_idx})')
        axes[0, 1].axis('off')
        
        # Phase difference
        axes[0, 2].imshow(self.phase_diff[:, :, slice_idx], cmap='RdBu_r')
        axes[0, 2].set_title(f'Phase Difference (Slice {slice_idx})')
        axes[0, 2].axis('off')
        
        # B0 field map
        if self.fieldmap_hz is not None:
            im = axes[0, 3].imshow(self.fieldmap_hz[:, :, slice_idx], cmap='RdBu_r')
            axes[0, 3].set_title(f'B0 Field Map (Hz) (Slice {slice_idx})')
            axes[0, 3].axis('off')
            plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Distortion map
        if self.distortion_map_mm is not None:
            im = axes[1, 0].imshow(self.distortion_map_mm[:, :, slice_idx], cmap='RdBu_r')
            axes[1, 0].set_title(f'Distortion Map (mm) (Slice {slice_idx})')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Magnitude difference
        mag_diff = self.mag_te15 - self.mag_te10
        axes[1, 1].imshow(mag_diff[:, :, slice_idx], cmap='RdBu_r')
        axes[1, 1].set_title(f'Magnitude Difference (Slice {slice_idx})')
        axes[1, 1].axis('off')
        
        # Histogram of distortion
        if self.distortion_map_mm is not None:
            mask = self.create_brain_mask()
            distortion_values = self.distortion_map_mm[mask]
            axes[1, 2].hist(distortion_values, bins=50, alpha=0.7, color='blue')
            axes[1, 2].set_title('Distortion Distribution (mm)')
            axes[1, 2].set_xlabel('Displacement (mm)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 3].axis('off')
        if self.distortion_map_mm is not None:
            mask = self.create_brain_mask()
            distortion_values = self.distortion_map_mm[mask]
            stats_text = f"""
Distortion Statistics:
Max: {np.max(np.abs(distortion_values)):.2f} mm
Mean: {np.mean(np.abs(distortion_values)):.2f} mm
Std: {np.std(distortion_values):.2f} mm
95th percentile: {np.percentile(np.abs(distortion_values), 95):.2f} mm

Voxel size: {self.voxel_size[0]:.1f}×{self.voxel_size[1]:.1f}×{self.voxel_size[2]:.1f} mm
Echo spacing: {self.echo_spacing*1000:.1f} ms
ΔTE: {self.delta_te*1000:.1f} ms
            """
            axes[1, 3].text(0.1, 0.9, stats_text, transform=axes[1, 3].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / 'siemens_distortion_analysis.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualization saved to {output_path / 'siemens_distortion_analysis.png'}")
        
        return fig


def main():
    """Main processing function."""
    print("🏥 Siemens MRI Distortion Analysis")
    print("=" * 60)
    
    # Initialize processor
    data_path = "/Users/alifatemi/Desktop/MRI phantom images"
    processor = SiemensMRIDataProcessor(data_path)
    
    try:
        # Step 1: Load all data
        processor.load_all_data()
        
        # Step 2: Create brain mask
        mask = processor.create_brain_mask()
        
        # Step 3: Estimate B0 field map from phase difference
        print("\n" + "="*60)
        print("STEP 1: B0 Field Map Estimation")
        print("="*60)
        fieldmap = processor.estimate_fieldmap_from_phase_diff(mask)
        
        # Step 4: Smooth field map (optional)
        print("\n" + "="*60)
        print("STEP 2: Field Map Smoothing")
        print("="*60)
        processor.smooth_fieldmap(sigma=1.0, mask=mask)
        
        # Step 5: Calculate distortion map in mm
        print("\n" + "="*60)
        print("STEP 3: Distortion Map Calculation")
        print("="*60)
        distortion_map = processor.calculate_distortion_map_mm(mask, phase_encode_direction='y')
        
        # Step 6: Save results
        print("\n" + "="*60)
        print("STEP 4: Save Results")
        print("="*60)
        processor.save_results()
        
        # Step 7: Create visualization
        print("\n" + "="*60)
        print("STEP 5: Visualization")
        print("="*60)
        processor.create_visualization()
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print("✅ B0 field map estimated and saved")
        print("✅ Geometrical distortion map calculated in mm")
        print("✅ Results saved to 'results/' folder")
        print("✅ Visualization created")
        
        print(f"\n📊 Final Results:")
        print(f"   📏 Max distortion: {np.max(np.abs(distortion_map[mask])):.2f} mm")
        print(f"   📊 Mean distortion: {np.mean(np.abs(distortion_map[mask])):.2f} mm")
        print(f"   📈 Std distortion: {np.std(distortion_map[mask]):.2f} mm")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


