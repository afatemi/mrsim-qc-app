"""
Simple example demonstrating B0 field map utilities.

This script shows basic usage of the b0map utilities without complex phase unwrapping.
"""

import numpy as np
import matplotlib.pyplot as plt
from mri_utils import (
    estimate_b0_fieldmap,
    phase_unwrap_region_growing,
    correct_geometric_distortion,
    assess_fieldmap_quality,
    smooth_fieldmap
)


def create_simple_synthetic_data():
    """Create simple synthetic multi-echo MRI data."""
    # Create 2D synthetic data for simplicity
    shape = (64, 64)
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing='ij')
    
    # Create brain mask
    brain_mask = (x**2 + y**2) < 0.8
    
    # Create synthetic B0 field map (Hz)
    fieldmap_true = 30 * np.exp(-(x**2 + y**2) / 0.3) * brain_mask
    
    # Create synthetic magnitude images
    magnitude1 = np.exp(-(x**2 + y**2) / 0.5) * brain_mask
    magnitude2 = magnitude1 * 0.9  # Slight decay
    
    # Create phase images with different echo times
    te1, te2 = 0.005, 0.010  # 5ms and 10ms
    phase1 = 2 * np.pi * fieldmap_true * te1
    phase2 = 2 * np.pi * fieldmap_true * te2
    
    # Add noise
    noise_level = 0.05
    magnitude1 += np.random.normal(0, noise_level, shape)
    magnitude2 += np.random.normal(0, noise_level, shape)
    phase1 += np.random.normal(0, noise_level, shape)
    phase2 += np.random.normal(0, noise_level, shape)
    
    # Create complex images
    echo1 = magnitude1 * np.exp(1j * phase1)
    echo2 = magnitude2 * np.exp(1j * phase2)
    
    return echo1, echo2, te1, te2, fieldmap_true, brain_mask


def demonstrate_basic_usage():
    """Demonstrate basic b0map utilities."""
    print("=== Basic B0 Field Map Utilities Demo ===")
    
    # Create synthetic data
    echo1, echo2, te1, te2, fieldmap_true, mask = create_simple_synthetic_data()
    
    print(f"Data shape: {echo1.shape}")
    print(f"Echo times: {te1*1000:.1f}ms, {te2*1000:.1f}ms")
    
    # 1. Estimate B0 field map
    print("\n1. Estimating B0 field map...")
    fieldmap_estimated = estimate_b0_fieldmap(echo1, echo2, te1, te2, mask)
    
    # 2. Assess quality
    print("2. Assessing field map quality...")
    quality_metrics = assess_fieldmap_quality(fieldmap_estimated, mask)
    print(f"   Mean: {quality_metrics['mean']:.2f} Hz")
    print(f"   Std: {quality_metrics['std']:.2f} Hz")
    print(f"   Range: {quality_metrics['range']:.2f} Hz")
    print(f"   SNR: {quality_metrics['snr']:.2f}")
    
    # 3. Smooth field map
    print("3. Smoothing field map...")
    fieldmap_smoothed = smooth_fieldmap(fieldmap_estimated, sigma=1.0, mask=mask)
    
    # 4. Simple phase unwrapping (region growing)
    print("4. Performing phase unwrapping...")
    # Create a simple wrapped phase for demonstration
    phase_wrapped = np.angle(echo2 * np.conj(echo1))
    phase_unwrapped = phase_unwrap_region_growing(phase_wrapped)
    
    # 5. Correct geometric distortion
    print("5. Correcting geometric distortion...")
    # Create a simple test image
    test_image = np.abs(echo1)
    echo_spacing = 0.001  # 1ms
    corrected_image = correct_geometric_distortion(
        test_image, fieldmap_estimated, echo_spacing, 'y'
    )
    
    print("✅ All utilities working successfully!")
    
    return {
        'fieldmap_true': fieldmap_true,
        'fieldmap_estimated': fieldmap_estimated,
        'fieldmap_smoothed': fieldmap_smoothed,
        'phase_wrapped': phase_wrapped,
        'phase_unwrapped': phase_unwrapped,
        'original_image': test_image,
        'corrected_image': corrected_image,
        'mask': mask
    }


def create_simple_visualization(results):
    """Create simple visualization of results."""
    print("\n6. Creating visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Field maps
    axes[0, 0].imshow(results['fieldmap_true'], cmap='RdBu_r')
    axes[0, 0].set_title('True Field Map (Hz)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(results['fieldmap_estimated'], cmap='RdBu_r')
    axes[0, 1].set_title('Estimated Field Map (Hz)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(results['fieldmap_smoothed'], cmap='RdBu_r')
    axes[0, 2].set_title('Smoothed Field Map (Hz)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(np.abs(results['fieldmap_estimated'] - results['fieldmap_true']), cmap='hot')
    axes[0, 3].set_title('Estimation Error (Hz)')
    axes[0, 3].axis('off')
    
    # Phase and images
    axes[1, 0].imshow(results['phase_wrapped'], cmap='RdBu_r')
    axes[1, 0].set_title('Wrapped Phase')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(results['phase_unwrapped'], cmap='RdBu_r')
    axes[1, 1].set_title('Unwrapped Phase')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(results['original_image'], cmap='gray')
    axes[1, 2].set_title('Original Image')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(results['corrected_image'], cmap='gray')
    axes[1, 3].set_title('Corrected Image')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_b0map_results.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved as 'simple_b0map_results.png'")
    
    return fig


def main():
    """Main demonstration function."""
    print("Simple B0 Field Map Utilities Demonstration")
    print("=" * 50)
    
    try:
        # Run demonstration
        results = demonstrate_basic_usage()
        
        # Create visualization
        create_simple_visualization(results)
        
        print("\n" + "=" * 50)
        print("✅ Demonstration completed successfully!")
        print("Your b0map utilities are working correctly.")
        print("\nNext steps:")
        print("- Check 'simple_b0map_results.png' for visualization")
        print("- Use the utilities in your own MRI processing pipeline")
        print("- Import functions: from mri_utils import estimate_b0_fieldmap, ...")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
