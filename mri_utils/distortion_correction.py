"""
Geometric distortion correction utilities for MRI.

This module provides comprehensive tools for correcting geometric distortions
caused by B0 field inhomogeneities in MRI images.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.optimize import minimize
from typing import Tuple, Optional, Union, List
import warnings


def correct_epi_distortion(image: np.ndarray,
                          fieldmap: np.ndarray,
                          echo_spacing: float,
                          phase_encode_direction: str = 'y',
                          method: str = 'interpolation') -> np.ndarray:
    """
    Correct EPI geometric distortion using B0 field map.
    
    Parameters:
    -----------
    image : np.ndarray
        Distorted EPI image
    fieldmap : np.ndarray
        B0 field map in Hz
    echo_spacing : float
        Echo spacing in seconds
    phase_encode_direction : str
        Phase encoding direction ('x', 'y', or 'z')
    method : str
        Correction method ('interpolation', 'forward_warping', 'backward_warping')
        
    Returns:
    --------
    corrected_image : np.ndarray
        Geometrically corrected image
    """
    if method == 'interpolation':
        return _correct_distortion_interpolation(image, fieldmap, echo_spacing, phase_encode_direction)
    elif method == 'forward_warping':
        return _correct_distortion_forward_warping(image, fieldmap, echo_spacing, phase_encode_direction)
    elif method == 'backward_warping':
        return _correct_distortion_backward_warping(image, fieldmap, echo_spacing, phase_encode_direction)
    else:
        raise ValueError("Method must be 'interpolation', 'forward_warping', or 'backward_warping'")


def _correct_distortion_interpolation(image: np.ndarray,
                                    fieldmap: np.ndarray,
                                    echo_spacing: float,
                                    phase_encode_direction: str) -> np.ndarray:
    """Correct distortion using interpolation method."""
    # Calculate displacement field
    displacement = fieldmap * echo_spacing
    
    # Create coordinate grids
    shape = image.shape
    coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    
    # Apply displacement in phase encode direction
    if phase_encode_direction == 'x':
        coords[0] += displacement
    elif phase_encode_direction == 'y':
        coords[1] += displacement
    elif phase_encode_direction == 'z':
        coords[2] += displacement
    
    # Create interpolation points
    points = np.column_stack([c.ravel() for c in coords])
    values = image.ravel()
    
    # Create target grid
    target_coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    target_points = np.column_stack([c.ravel() for c in target_coords])
    
    # Interpolate
    corrected = griddata(points, values, target_points, method='linear', fill_value=0)
    corrected = corrected.reshape(shape)
    
    return corrected


def _correct_distortion_forward_warping(image: np.ndarray,
                                      fieldmap: np.ndarray,
                                      echo_spacing: float,
                                      phase_encode_direction: str) -> np.ndarray:
    """Correct distortion using forward warping."""
    # Calculate displacement field
    displacement = fieldmap * echo_spacing
    
    # Create coordinate grids
    shape = image.shape
    coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    
    # Apply displacement
    if phase_encode_direction == 'x':
        coords[0] += displacement
    elif phase_encode_direction == 'y':
        coords[1] += displacement
    elif phase_encode_direction == 'z':
        coords[2] += displacement
    
    # Create corrected image
    corrected = np.zeros_like(image)
    
    # Forward warp
    for i in range(shape[0]):
        for j in range(shape[1]):
            if len(shape) == 3:
                for k in range(shape[2]):
                    new_coords = (coords[0][i, j, k], coords[1][i, j, k], coords[2][i, j, k])
                    if all(0 <= c < s for c, s in zip(new_coords, shape)):
                        corrected[new_coords] = image[i, j, k]
            else:
                new_coords = (coords[0][i, j], coords[1][i, j])
                if all(0 <= c < s for c, s in zip(new_coords, shape)):
                    corrected[new_coords] = image[i, j]
    
    return corrected


def _correct_distortion_backward_warping(image: np.ndarray,
                                       fieldmap: np.ndarray,
                                       echo_spacing: float,
                                       phase_encode_direction: str) -> np.ndarray:
    """Correct distortion using backward warping."""
    # Calculate displacement field
    displacement = fieldmap * echo_spacing
    
    # Create coordinate grids
    shape = image.shape
    coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    
    # Apply displacement (opposite direction for backward warping)
    if phase_encode_direction == 'x':
        coords[0] -= displacement
    elif phase_encode_direction == 'y':
        coords[1] -= displacement
    elif phase_encode_direction == 'z':
        coords[2] -= displacement
    
    # Create interpolator
    if len(shape) == 3:
        interpolator = RegularGridInterpolator(
            (np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])),
            image, method='linear', bounds_error=False, fill_value=0
        )
    else:
        interpolator = RegularGridInterpolator(
            (np.arange(shape[0]), np.arange(shape[1])),
            image, method='linear', bounds_error=False, fill_value=0
        )
    
    # Interpolate at displaced coordinates
    points = np.column_stack([c.ravel() for c in coords])
    corrected = interpolator(points).reshape(shape)
    
    return corrected


def correct_susceptibility_distortion(image: np.ndarray,
                                    fieldmap: np.ndarray,
                                    echo_time: float,
                                    bandwidth_per_pixel: float,
                                    phase_encode_direction: str = 'y') -> np.ndarray:
    """
    Correct susceptibility-induced geometric distortion.
    
    Parameters:
    -----------
    image : np.ndarray
        Distorted image
    fieldmap : np.ndarray
        B0 field map in Hz
    echo_time : float
        Echo time in seconds
    bandwidth_per_pixel : float
        Bandwidth per pixel in Hz
    phase_encode_direction : str
        Phase encoding direction
        
    Returns:
    --------
    corrected_image : np.ndarray
        Corrected image
    """
    # Calculate displacement in pixels
    displacement_pixels = fieldmap * echo_time / bandwidth_per_pixel
    
    # Apply correction using backward warping
    return _correct_distortion_backward_warping(
        image, displacement_pixels, 1.0, phase_encode_direction
    )


def estimate_distortion_field(fieldmap: np.ndarray,
                            echo_spacing: float,
                            phase_encode_direction: str = 'y') -> np.ndarray:
    """
    Estimate geometric distortion field from B0 field map.
    
    Parameters:
    -----------
    fieldmap : np.ndarray
        B0 field map in Hz
    echo_spacing : float
        Echo spacing in seconds
    phase_encode_direction : str
        Phase encoding direction
        
    Returns:
    --------
    distortion_field : np.ndarray
        Distortion field in pixels
    """
    # Calculate displacement in pixels
    displacement = fieldmap * echo_spacing
    
    # Create distortion field
    shape = fieldmap.shape
    distortion_field = np.zeros((*shape, len(shape)))
    
    if phase_encode_direction == 'x':
        distortion_field[..., 0] = displacement
    elif phase_encode_direction == 'y':
        distortion_field[..., 1] = displacement
    elif phase_encode_direction == 'z':
        distortion_field[..., 2] = displacement
    
    return distortion_field


def apply_distortion_field(image: np.ndarray,
                          distortion_field: np.ndarray,
                          method: str = 'backward') -> np.ndarray:
    """
    Apply distortion field to image.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    distortion_field : np.ndarray
        Distortion field (shape: [..., ndim])
    method : str
        Warping method ('forward' or 'backward')
        
    Returns:
    --------
    warped_image : np.ndarray
        Warped image
    """
    shape = image.shape
    ndim = len(shape)
    
    if method == 'backward':
        # Create coordinate grids
        coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
        
        # Apply distortion field
        for i in range(ndim):
            coords[i] += distortion_field[..., i]
        
        # Create interpolator
        if ndim == 3:
            interpolator = RegularGridInterpolator(
                (np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])),
                image, method='linear', bounds_error=False, fill_value=0
            )
        else:
            interpolator = RegularGridInterpolator(
                (np.arange(shape[0]), np.arange(shape[1])),
                image, method='linear', bounds_error=False, fill_value=0
            )
        
        # Interpolate
        points = np.column_stack([c.ravel() for c in coords])
        warped = interpolator(points).reshape(shape)
        
    else:  # forward warping
        warped = np.zeros_like(image)
        
        # Forward warp
        for idx in np.ndindex(shape):
            new_coords = tuple(int(c + distortion_field[idx + (i,)]) for i, c in enumerate(idx))
            if all(0 <= c < s for c, s in zip(new_coords, shape)):
                warped[new_coords] = image[idx]
    
    return warped


def correct_multi_echo_distortion(echo_images: List[np.ndarray],
                                echo_times: List[float],
                                fieldmap: np.ndarray,
                                echo_spacing: float,
                                phase_encode_direction: str = 'y') -> List[np.ndarray]:
    """
    Correct distortion for multiple echo images.
    
    Parameters:
    -----------
    echo_images : list of np.ndarray
        List of echo images
    echo_times : list of float
        List of echo times in seconds
    fieldmap : np.ndarray
        B0 field map in Hz
    echo_spacing : float
        Echo spacing in seconds
    phase_encode_direction : str
        Phase encoding direction
        
    Returns:
    --------
    corrected_images : list of np.ndarray
        List of corrected images
    """
    corrected_images = []
    
    for echo_image, te in zip(echo_images, echo_times):
        # Calculate echo-specific displacement
        echo_displacement = fieldmap * echo_spacing * (te / echo_times[0])
        
        # Apply correction
        corrected = _correct_distortion_backward_warping(
            echo_image, echo_displacement, 1.0, phase_encode_direction
        )
        corrected_images.append(corrected)
    
    return corrected_images


def assess_distortion_severity(fieldmap: np.ndarray,
                             echo_spacing: float,
                             voxel_size: Tuple[float, ...],
                             phase_encode_direction: str = 'y') -> dict:
    """
    Assess severity of geometric distortion.
    
    Parameters:
    -----------
    fieldmap : np.ndarray
        B0 field map in Hz
    echo_spacing : float
        Echo spacing in seconds
    voxel_size : tuple
        Voxel size in mm
    phase_encode_direction : str
        Phase encoding direction
        
    Returns:
    --------
    severity_metrics : dict
        Dictionary containing distortion severity metrics
    """
    # Calculate displacement in pixels
    displacement_pixels = fieldmap * echo_spacing
    
    # Calculate displacement in mm
    pe_axis = {'x': 0, 'y': 1, 'z': 2}[phase_encode_direction]
    displacement_mm = displacement_pixels * voxel_size[pe_axis]
    
    # Calculate metrics
    metrics = {
        'max_displacement_pixels': np.max(np.abs(displacement_pixels)),
        'mean_displacement_pixels': np.mean(np.abs(displacement_pixels)),
        'std_displacement_pixels': np.std(displacement_pixels),
        'max_displacement_mm': np.max(np.abs(displacement_mm)),
        'mean_displacement_mm': np.mean(np.abs(displacement_mm)),
        'std_displacement_mm': np.std(displacement_mm),
        'distortion_volume_fraction': np.sum(np.abs(displacement_pixels) > 1.0) / displacement_pixels.size
    }
    
    return metrics


def optimize_echo_spacing(fieldmap: np.ndarray,
                        target_max_displacement: float = 1.0,
                        phase_encode_direction: str = 'y') -> float:
    """
    Optimize echo spacing to minimize distortion.
    
    Parameters:
    -----------
    fieldmap : np.ndarray
        B0 field map in Hz
    target_max_displacement : float
        Target maximum displacement in pixels
    phase_encode_direction : str
        Phase encoding direction
        
    Returns:
    --------
    optimal_echo_spacing : float
        Optimal echo spacing in seconds
    """
    # Calculate maximum field map value
    max_fieldmap = np.max(np.abs(fieldmap))
    
    # Calculate optimal echo spacing
    optimal_echo_spacing = target_max_displacement / max_fieldmap
    
    return optimal_echo_spacing


def create_distortion_correction_lut(fieldmap: np.ndarray,
                                   echo_spacing: float,
                                   phase_encode_direction: str = 'y',
                                   resolution: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lookup table for distortion correction.
    
    Parameters:
    -----------
    fieldmap : np.ndarray
        B0 field map in Hz
    echo_spacing : float
        Echo spacing in seconds
    phase_encode_direction : str
        Phase encoding direction
    resolution : int
        LUT resolution
        
    Returns:
    --------
    fieldmap_values : np.ndarray
        Field map values for LUT
    displacement_values : np.ndarray
        Corresponding displacement values
    """
    # Calculate displacement range
    displacement = fieldmap * echo_spacing
    min_disp = np.min(displacement)
    max_disp = np.max(displacement)
    
    # Create LUT
    fieldmap_values = np.linspace(min_disp / echo_spacing, max_disp / echo_spacing, resolution)
    displacement_values = fieldmap_values * echo_spacing
    
    return fieldmap_values, displacement_values


def apply_distortion_correction_lut(image: np.ndarray,
                                  fieldmap: np.ndarray,
                                  fieldmap_values: np.ndarray,
                                  displacement_values: np.ndarray) -> np.ndarray:
    """
    Apply distortion correction using lookup table.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image
    fieldmap : np.ndarray
        B0 field map in Hz
    fieldmap_values : np.ndarray
        Field map values for LUT
    displacement_values : np.ndarray
        Corresponding displacement values
        
    Returns:
    --------
    corrected_image : np.ndarray
        Corrected image
    """
    # Interpolate displacement from LUT
    displacement = np.interp(fieldmap, fieldmap_values, displacement_values)
    
    # Apply correction
    corrected = _correct_distortion_backward_warping(
        image, displacement, 1.0, 'y'  # Assuming y-direction
    )
    
    return corrected


