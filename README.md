# MRI Geometrical Utilities

A comprehensive Python package for MRI geometric distortion correction, including B0 field map estimation, phase unwrapping, and distortion correction algorithms.

## Features

### Core B0 Field Map Utilities (`b0map.py`)
- **Field Map Estimation**: Estimate B0 field maps from multi-echo MRI data
- **Phase Unwrapping**: Multiple algorithms including region growing and Goldstein branch cut
- **Quality Assessment**: Comprehensive field map quality metrics
- **Smoothing**: Gaussian smoothing for noise reduction
- **Echo Time Optimization**: Optimal echo time estimation for field mapping

### Advanced Phase Unwrapping (`phase_unwrap.py`)
- **Laplacian Method**: Poisson equation-based unwrapping
- **Quality-Guided Unwrapping**: Magnitude-guided phase unwrapping
- **Multi-Resolution Unwrapping**: Coarse-to-fine unwrapping strategy
- **Phase Jump Detection**: Automatic detection of phase discontinuities
- **Reliability Estimation**: Unwrapping reliability assessment

### Distortion Correction (`distortion_correction.py`)
- **EPI Distortion Correction**: Correction for echo-planar imaging distortions
- **Susceptibility Correction**: Correction for susceptibility-induced distortions
- **Multiple Warping Methods**: Forward, backward, and interpolation-based warping
- **Multi-Echo Correction**: Simultaneous correction of multiple echo images
- **Distortion Assessment**: Comprehensive distortion severity metrics
- **Lookup Table Optimization**: Efficient LUT-based correction

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The package is ready to use - no additional installation needed
```

## Quick Start

```python
import numpy as np
from mri_utils import (
    estimate_b0_fieldmap,
    phase_unwrap_laplacian,
    correct_epi_distortion,
    assess_fieldmap_quality
)

# Estimate B0 field map from multi-echo data
fieldmap = estimate_b0_fieldmap(echo1, echo2, te1, te2, mask)

# Unwrap phase using Laplacian method
unwrapped_phase = phase_unwrap_laplacian(wrapped_phase, mask)

# Correct EPI distortion
corrected_image = correct_epi_distortion(
    distorted_image, fieldmap, echo_spacing, 'y'
)

# Assess field map quality
quality_metrics = assess_fieldmap_quality(fieldmap, mask)
```

## Example Usage

Run the comprehensive example script to see all utilities in action:

```bash
python example_b0map_usage.py
```

This will demonstrate:
- B0 field map estimation from synthetic multi-echo data
- Phase unwrapping using different algorithms
- Geometric distortion correction
- Quality assessment and visualization

## API Reference

### Core Functions

#### Field Map Estimation
- `estimate_b0_fieldmap(echo1, echo2, te1, te2, mask=None)`: Estimate B0 field map from two echoes
- `assess_fieldmap_quality(fieldmap, mask=None)`: Assess field map quality metrics
- `smooth_fieldmap(fieldmap, sigma=1.0, mask=None)`: Smooth field map with Gaussian filter
- `estimate_echo_times_optimal(echo1, echo2, te1, te2, target_phase_diff=π/2)`: Optimize echo times

#### Phase Unwrapping
- `phase_unwrap_region_growing(phase, mask=None, seed=None)`: Region growing unwrapping
- `phase_unwrap_goldstein(phase, mask=None)`: Goldstein branch cut method
- `phase_unwrap_laplacian(phase, mask=None)`: Laplacian-based unwrapping
- `phase_unwrap_quality_guided(phase, magnitude, mask=None)`: Quality-guided unwrapping
- `phase_unwrap_multiresolution(phase, mask=None, levels=3)`: Multi-resolution unwrapping

#### Distortion Correction
- `correct_epi_distortion(image, fieldmap, echo_spacing, phase_encode_direction='y', method='interpolation')`: EPI correction
- `correct_susceptibility_distortion(image, fieldmap, echo_time, bandwidth_per_pixel, phase_encode_direction='y')`: Susceptibility correction
- `apply_distortion_field(image, distortion_field, method='backward')`: Apply distortion field
- `assess_distortion_severity(fieldmap, echo_spacing, voxel_size, phase_encode_direction='y')`: Assess distortion severity

## Dependencies

- **numpy** (>=1.21.0): Core numerical computing
- **scipy** (>=1.7.0): Scientific computing and sparse operations
- **matplotlib** (>=3.5.0): Visualization (optional)
- **scikit-image** (>=0.19.0): Image processing (optional)

## File Structure

```
mri_utils/
├── __init__.py              # Package initialization and exports
├── b0map.py                 # Core B0 field map utilities
├── phase_unwrap.py          # Advanced phase unwrapping algorithms
└── distortion_correction.py # Geometric distortion correction
```

## Contributing

This package is designed for research and clinical applications in MRI. Contributions are welcome for:
- Additional phase unwrapping algorithms
- Improved distortion correction methods
- Performance optimizations
- Documentation improvements

## License

This project is open source. Please check the license file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{mri_geometrical_utils,
  title={MRI Geometrical Utilities: B0 Field Map Processing and Distortion Correction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MRIgeometrical}
}
```


