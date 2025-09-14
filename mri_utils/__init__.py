"""
MRI Geometrical Utilities Package

This package provides comprehensive tools for MRI geometric distortion correction,
including B0 field map estimation, phase unwrapping, and distortion correction.
"""

# Import core b0map utilities
from .b0map import (
    estimate_b0_fieldmap,
    phase_unwrap_region_growing,
    phase_unwrap_goldstein,
    correct_geometric_distortion,
    assess_fieldmap_quality,
    smooth_fieldmap,
    estimate_echo_times_optimal
)

# Import advanced phase unwrapping utilities
from .phase_unwrap import (
    phase_unwrap_laplacian,
    phase_unwrap_quality_guided,
    phase_unwrap_multiresolution,
    detect_phase_jumps,
    estimate_unwrapping_reliability
)

# Import distortion correction utilities
from .distortion_correction import (
    correct_epi_distortion,
    correct_susceptibility_distortion,
    estimate_distortion_field,
    apply_distortion_field,
    correct_multi_echo_distortion,
    assess_distortion_severity,
    optimize_echo_spacing,
    create_distortion_correction_lut,
    apply_distortion_correction_lut
)

# Package version
__version__ = "1.0.0"

# Main classes and functions to expose
__all__ = [
    # Core b0map functions
    'estimate_b0_fieldmap',
    'phase_unwrap_region_growing', 
    'phase_unwrap_goldstein',
    'correct_geometric_distortion',
    'assess_fieldmap_quality',
    'smooth_fieldmap',
    'estimate_echo_times_optimal',
    
    # Advanced phase unwrapping
    'phase_unwrap_laplacian',
    'phase_unwrap_quality_guided',
    'phase_unwrap_multiresolution',
    'detect_phase_jumps',
    'estimate_unwrapping_reliability',
    
    # Distortion correction
    'correct_epi_distortion',
    'correct_susceptibility_distortion',
    'estimate_distortion_field',
    'apply_distortion_field',
    'correct_multi_echo_distortion',
    'assess_distortion_severity',
    'optimize_echo_spacing',
    'create_distortion_correction_lut',
    'apply_distortion_correction_lut'
]


