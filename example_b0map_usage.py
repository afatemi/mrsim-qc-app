"""
Example usage of B0 field map utilities for MRI geometric distortion correction.

This script demonstrates how to use the b0map utilities to:
1. Estimate B0 field maps from multi-echo data
2. Perform phase unwrapping
3. Correct geometric distortions
4. Assess field map quality
"""

import numpy as np
import matplotlib.pyplot as plt
from mri_utils import (
    estimate_b0_fieldmap,
    phase_unwrap_laplacian,
    phase_unwrap_quality_guided,
    correct_epi_distortion,
    assess_fieldmap_quality,
    smooth_fieldmap,
    detect_phase_jumps,
    assess_distortion_severity
)


def create_synthetic_data():
    """Create synthetic multi-echo MRI data for demonstration."""
    # Create synthetic brain-like phantom
    shape = (64, 64, 32)
    x, y, z = np.meshgrid(np.linspace(-1, 1, shape[0]),
                         np.linspace(-1, 1, shape[1]),
                         np.linspace(-1, 1, shape[2]), indexing='ij')
    
    # Create brain mask
    brain_mask = (x**2 + y**2 + z**2) < 0.8
    
    # Create synthetic B0 field map (Hz)
    fieldmap_true = 50 * np.exp(-(x**2 + y**2 + z**2) / 0.3) * brain_mask
    
    # Create synthetic magnitude images
    magnitude1 = np.exp(-(x**2 + y**2 + z**2) / 0.5) * brain_mask
    magnitude2 = magnitude1 * 0.9  # Slight decay
    
    # Create phase images with different echo times
    te1, te2 = 0.005, 0.010  # 5ms and 10ms
    phase1 = 2 * np.pi * fieldmap_true * te1
    phase2 = 2 * np.pi * fieldmap_true * te2
    
    # Add noise
    noise_level = 0.1
    magnitude1 += np.random.normal(0, noise_level, shape)
    magnitude2 += np.random.normal(0, noise_level, shape)
    phase1 += np.random.normal(0, noise_level, shape)
    phase2 += np.random.normal(0, noise_level, shape)
    
    # Create complex images
    echo1 = magnitude1 * np.exp(1j * phase1)
    echo2 = magnitude2 * np.exp(1j * phase2)
    
    return echo1, echo2, te1, te2, fieldmap_true, brain_mask


def demonstrate_fieldmap_estimation():
    """Demonstrate B0 field map estimation."""
    print("=== B0 Field Map Estimation ===")
    
    # Create synthetic data
    echo1, echo2, te1, te2, fieldmap_true, mask = create_synthetic_data()
    
    # Estimate field map
    fieldmap_estimated = estimate_b0_fieldmap(echo1, echo2, te1, te2, mask)
    
    # Assess quality
    quality_metrics = assess_fieldmap_quality(fieldmap_estimated, mask)
    print(f"Field map quality metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Smooth field map
    fieldmap_smoothed = smooth_fieldmap(fieldmap_estimated, sigma=1.0, mask=mask)
    
    return fieldmap_estimated, fieldmap_smoothed, fieldmap_true, mask


def demonstrate_phase_unwrapping():
    """Demonstrate phase unwrapping algorithms."""
    print("\n=== Phase Unwrapping ===")
    
    # Create synthetic wrapped phase
    shape = (64, 64)
    x, y = np.meshgrid(np.linspace(-1, 1, shape[0]), np.linspace(-1, 1, shape[1]), indexing='ij')
    
    # Create true unwrapped phase
    phase_true = 10 * np.pi * (x**2 + y**2)
    
    # Wrap phase
    phase_wrapped = np.arctan2(np.sin(phase_true), np.cos(phase_true))
    
    # Create magnitude for quality-guided unwrapping
    magnitude = np.exp(-(x**2 + y**2) / 0.3)
    
    # Apply different unwrapping methods
    phase_laplacian = phase_unwrap_laplacian(phase_wrapped)
    phase_quality_guided = phase_unwrap_quality_guided(phase_wrapped, magnitude)
    
    # Detect phase jumps
    jump_map = detect_phase_jumps(phase_wrapped)
    
    print(f"Phase unwrapping completed:")
    print(f"  Laplacian method - max difference from true: {np.max(np.abs(phase_laplacian - phase_true)):.3f}")
    print(f"  Quality-guided method - max difference from true: {np.max(np.abs(phase_quality_guided - phase_true)):.3f}")
    print(f"  Phase jumps detected: {np.sum(jump_map)} pixels")
    
    return phase_wrapped, phase_laplacian, phase_quality_guided, phase_true


def demonstrate_distortion_correction():
    """Demonstrate geometric distortion correction."""
    print("\n=== Distortion Correction ===")
    
    # Create synthetic distorted image
    shape = (64, 64, 32)
    x, y, z = np.meshgrid(np.linspace(-1, 1, shape[0]),
                         np.linspace(-1, 1, shape[1]),
                         np.linspace(-1, 1, shape[2]), indexing='ij')
    
    # Create original image
    original_image = np.exp(-(x**2 + y**2 + z**2) / 0.5)
    
    # Create field map
    fieldmap = 30 * np.exp(-(x**2 + y**2 + z**2) / 0.4)
    
    # Simulate distortion
    echo_spacing = 0.001  # 1ms
    distorted_image = correct_epi_distortion(original_image, fieldmap, echo_spacing, 'y', 'forward_warping')
    
    # Correct distortion
    corrected_image = correct_epi_distortion(distorted_image, fieldmap, echo_spacing, 'y', 'backward_warping')
    
    # Assess distortion severity
    voxel_size = (2.0, 2.0, 2.0)  # 2mm isotropic
    severity_metrics = assess_distortion_severity(fieldmap, echo_spacing, voxel_size, 'y')
    
    print(f"Distortion severity metrics:")
    for metric, value in severity_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Calculate correction accuracy
    correction_error = np.mean(np.abs(corrected_image - original_image))
    print(f"  Correction error (mean absolute difference): {correction_error:.3f}")
    
    return original_image, distorted_image, corrected_image, fieldmap


def create_visualization():
    """Create visualization of the results."""
    print("\n=== Creating Visualizations ===")
    
    # Get results from demonstrations
    fieldmap_est, fieldmap_smooth, fieldmap_true, mask = demonstrate_fieldmap_estimation()
    phase_wrapped, phase_lap, phase_qg, phase_true = demonstrate_phase_unwrapping()
    orig_img, dist_img, corr_img, fieldmap = demonstrate_distortion_correction()
    
    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Field map results
    slice_idx = 16
    axes[0, 0].imshow(fieldmap_true[:, :, slice_idx], cmap='RdBu_r')
    axes[0, 0].set_title('True Field Map')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fieldmap_est[:, :, slice_idx], cmap='RdBu_r')
    axes[0, 1].set_title('Estimated Field Map')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(fieldmap_smooth[:, :, slice_idx], cmap='RdBu_r')
    axes[0, 2].set_title('Smoothed Field Map')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(np.abs(fieldmap_est - fieldmap_true)[:, :, slice_idx], cmap='hot')
    axes[0, 3].set_title('Estimation Error')
    axes[0, 3].axis('off')
    
    # Phase unwrapping results
    axes[1, 0].imshow(phase_wrapped, cmap='RdBu_r')
    axes[1, 0].set_title('Wrapped Phase')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(phase_lap, cmap='RdBu_r')
    axes[1, 1].set_title('Laplacian Unwrapped')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(phase_qg, cmap='RdBu_r')
    axes[1, 2].set_title('Quality-Guided Unwrapped')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(np.abs(phase_qg - phase_true), cmap='hot')
    axes[1, 3].set_title('Unwrapping Error')
    axes[1, 3].axis('off')
    
    # Distortion correction results
    axes[2, 0].imshow(orig_img[:, :, slice_idx], cmap='gray')
    axes[2, 0].set_title('Original Image')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(dist_img[:, :, slice_idx], cmap='gray')
    axes[2, 1].set_title('Distorted Image')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(corr_img[:, :, slice_idx], cmap='gray')
    axes[2, 2].set_title('Corrected Image')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(np.abs(corr_img - orig_img)[:, :, slice_idx], cmap='hot')
    axes[2, 3].set_title('Correction Error')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('b0map_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'b0map_results.png'")
    
    return fig


def main():
    """Main demonstration function."""
    print("B0 Field Map Utilities Demonstration")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_fieldmap_estimation()
        demonstrate_phase_unwrapping()
        demonstrate_distortion_correction()
        
        # Create visualization
        create_visualization()
        
        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")
        print("All b0map utilities are working correctly.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
