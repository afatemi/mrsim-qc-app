"""
B0 field map utilities for MRI geometric distortion correction.

This module provides comprehensive tools for:
- B0 field map estimation from multi-echo data
- Phase unwrapping algorithms
- Geometric distortion correction
- Field map quality assessment
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize_scalar
from scipy.interpolate import griddata
from typing import Tuple, Optional, Union
import warnings


def estimate_b0_fieldmap(echo1: np.ndarray, echo2: np.ndarray, 
                        te1: float, te2: float, 
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate B0 field map from two echo times.
    
    Parameters:
    -----------
    echo1 : np.ndarray
        First echo magnitude image
    echo2 : np.ndarray  
        Second echo magnitude image
    te1 : float
        First echo time in seconds
    te2 : float
        Second echo time in seconds
    mask : np.ndarray, optional
        Brain mask to limit processing
        
    Returns:
    --------
    fieldmap : np.ndarray
        B0 field map in Hz
    """
    # Calculate phase difference
    phase_diff = np.angle(echo2 * np.conj(echo1))
    
    # Convert to field map in Hz
    delta_te = te2 - te1
    fieldmap = phase_diff / (2 * np.pi * delta_te)
    
    # Apply mask if provided
    if mask is not None:
        fieldmap = fieldmap * mask
        
    return fieldmap


def phase_unwrap_region_growing(phase: np.ndarray, 
                               seed: Optional[Tuple[int, int, int]] = None,
                               threshold: float = np.pi) -> np.ndarray:
    """
    Phase unwrapping using region growing algorithm.
    
    Parameters:
    -----------
    phase : np.ndarray
        Wrapped phase image
    seed : tuple, optional
        Starting point (x, y, z) for region growing
    threshold : float
        Phase jump threshold for unwrapping
        
    Returns:
    --------
    unwrapped_phase : np.ndarray
        Unwrapped phase image
    """
    unwrapped = phase.copy()
    shape = phase.shape
    
    # Find seed point if not provided
    if seed is None:
        # Use center of mass of high magnitude regions
        magnitude = np.abs(phase)
        center = ndi.center_of_mass(magnitude)
        seed = tuple(int(c) for c in center)
    
    # Initialize queue and visited array
    queue = [seed]
    visited = np.zeros(shape, dtype=bool)
    visited[seed] = True
    
    # 6-connectivity for 3D
    neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    while queue:
        current = queue.pop(0)
        cx, cy, cz = current
        
        for dx, dy, dz in neighbors:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            
            # Check bounds
            if (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                if not visited[nx, ny, nz]:
                    # Calculate phase difference
                    phase_diff = unwrapped[nx, ny, nz] - unwrapped[cx, cy, cz]
                    
                    # Unwrap if necessary
                    if phase_diff > threshold:
                        unwrapped[nx, ny, nz] -= 2 * np.pi
                    elif phase_diff < -threshold:
                        unwrapped[nx, ny, nz] += 2 * np.pi
                    
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))
    
    return unwrapped


def phase_unwrap_goldstein(phase: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Goldstein branch cut phase unwrapping algorithm.
    
    Parameters:
    -----------
    phase : np.ndarray
        Wrapped phase image
    mask : np.ndarray, optional
        Region of interest mask
        
    Returns:
    --------
    unwrapped_phase : np.ndarray
        Unwrapped phase image
    """
    if mask is None:
        mask = np.ones_like(phase, dtype=bool)
    
    # Calculate phase gradients
    grad_x = np.diff(phase, axis=0)
    grad_y = np.diff(phase, axis=1)
    
    # Wrap gradients
    grad_x = np.arctan2(np.sin(grad_x), np.cos(grad_x))
    grad_y = np.arctan2(np.sin(grad_y), np.cos(grad_y))
    
    # Calculate curl (residues)
    curl = np.zeros_like(phase)
    curl[:-1, :-1] = grad_x[:-1, :-1] - grad_x[:-1, 1:] - grad_y[:-1, :-1] + grad_y[1:, :-1]
    
    # Find residues
    residues = np.abs(curl) > np.pi/2
    
    # Place branch cuts to connect residues
    branch_cuts = _place_branch_cuts(residues, mask)
    
    # Unwrap phase avoiding branch cuts
    unwrapped = _unwrap_with_branch_cuts(phase, branch_cuts, mask)
    
    return unwrapped


def _place_branch_cuts(residues: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Place branch cuts to connect residues."""
    branch_cuts = np.zeros_like(residues, dtype=bool)
    
    # Simple implementation: connect nearest residues
    residue_coords = np.where(residues & mask)
    
    for i in range(len(residue_coords[0])):
        for j in range(i + 1, len(residue_coords[0])):
            # Connect residues with straight line
            start = (residue_coords[0][i], residue_coords[1][i])
            end = (residue_coords[0][j], residue_coords[1][j])
            
            # Bresenham line algorithm
            line_coords = _bresenham_line(start, end)
            for coord in line_coords:
                if (0 <= coord[0] < branch_cuts.shape[0] and 
                    0 <= coord[1] < branch_cuts.shape[1]):
                    branch_cuts[coord] = True
    
    return branch_cuts


def _bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> list:
    """Bresenham line algorithm for connecting points."""
    x0, y0 = start
    x1, y1 = end
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    line = []
    
    while True:
        line.append((x0, y0))
        
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        
        if e2 > -dy:
            err -= dy
            x0 += sx
            
        if e2 < dx:
            err += dx
            y0 += sy
    
    return line


def _unwrap_with_branch_cuts(phase: np.ndarray, branch_cuts: np.ndarray, 
                           mask: np.ndarray) -> np.ndarray:
    """Unwrap phase while avoiding branch cuts."""
    unwrapped = phase.copy()
    shape = phase.shape
    
    # Use region growing avoiding branch cuts
    visited = np.zeros(shape, dtype=bool)
    queue = []
    
    # Find starting point
    valid_points = mask & ~branch_cuts
    if np.any(valid_points):
        start_idx = np.where(valid_points)
        queue.append((start_idx[0][0], start_idx[1][0]))
        visited[start_idx[0][0], start_idx[1][0]] = True
    
    neighbors = [(1,0), (-1,0), (0,1), (0,-1)]
    
    while queue:
        current = queue.pop(0)
        cx, cy = current
        
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            
            if (0 <= nx < shape[0] and 0 <= ny < shape[1] and 
                not visited[nx, ny] and mask[nx, ny] and not branch_cuts[nx, ny]):
                
                # Calculate phase difference
                phase_diff = unwrapped[nx, ny] - unwrapped[cx, cy]
                
                # Unwrap if necessary
                if phase_diff > np.pi:
                    unwrapped[nx, ny] -= 2 * np.pi
                elif phase_diff < -np.pi:
                    unwrapped[nx, ny] += 2 * np.pi
                
                visited[nx, ny] = True
                queue.append((nx, ny))
    
    return unwrapped


def correct_geometric_distortion(image: np.ndarray, 
                                fieldmap: np.ndarray,
                                echo_spacing: float,
                                phase_encode_direction: str = 'y') -> np.ndarray:
    """
    Correct geometric distortion using B0 field map.
    
    Parameters:
    -----------
    image : np.ndarray
        Distorted image to correct
    fieldmap : np.ndarray
        B0 field map in Hz
    echo_spacing : float
        Echo spacing in seconds
    phase_encode_direction : str
        Phase encoding direction ('x', 'y', or 'z')
        
    Returns:
    --------
    corrected_image : np.ndarray
        Geometrically corrected image
    """
    # Calculate displacement field
    displacement = fieldmap * echo_spacing
    
    # Create coordinate grids
    shape = image.shape
    coords = list(np.meshgrid(*[np.arange(s, dtype=float) for s in shape], indexing='ij'))
    
    # Apply displacement in phase encode direction
    if phase_encode_direction == 'x':
        coords[0] += displacement
    elif phase_encode_direction == 'y':
        coords[1] += displacement
    elif phase_encode_direction == 'z':
        coords[2] += displacement
    else:
        raise ValueError("phase_encode_direction must be 'x', 'y', or 'z'")
    
    # Interpolate to correct positions
    corrected = griddata(
        points=np.column_stack([c.ravel() for c in coords]),
        values=image.ravel(),
        xi=np.column_stack([c.ravel() for c in np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')]),
        method='linear',
        fill_value=0
    ).reshape(shape)
    
    return corrected


def assess_fieldmap_quality(fieldmap: np.ndarray, 
                           mask: Optional[np.ndarray] = None) -> dict:
    """
    Assess quality of B0 field map.
    
    Parameters:
    -----------
    fieldmap : np.ndarray
        B0 field map in Hz
    mask : np.ndarray, optional
        Region of interest mask
        
    Returns:
    --------
    quality_metrics : dict
        Dictionary containing quality assessment metrics
    """
    if mask is None:
        mask = np.ones_like(fieldmap, dtype=bool)
    
    masked_fieldmap = fieldmap[mask]
    
    metrics = {
        'mean': np.mean(masked_fieldmap),
        'std': np.std(masked_fieldmap),
        'min': np.min(masked_fieldmap),
        'max': np.max(masked_fieldmap),
        'range': np.max(masked_fieldmap) - np.min(masked_fieldmap),
        'snr': np.mean(masked_fieldmap) / np.std(masked_fieldmap) if np.std(masked_fieldmap) > 0 else 0,
        'outlier_ratio': np.sum(np.abs(masked_fieldmap) > 3 * np.std(masked_fieldmap)) / len(masked_fieldmap)
    }
    
    return metrics


def smooth_fieldmap(fieldmap: np.ndarray, 
                   sigma: float = 1.0,
                   mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Smooth B0 field map using Gaussian filtering.
    
    Parameters:
    -----------
    fieldmap : np.ndarray
        B0 field map in Hz
    sigma : float
        Gaussian smoothing parameter
    mask : np.ndarray, optional
        Region of interest mask
        
    Returns:
    --------
    smoothed_fieldmap : np.ndarray
        Smoothed B0 field map
    """
    if mask is None:
        mask = np.ones_like(fieldmap, dtype=bool)
    
    # Apply Gaussian smoothing
    smoothed = ndi.gaussian_filter(fieldmap, sigma=sigma)
    
    # Preserve masked regions
    smoothed = smoothed * mask + fieldmap * (~mask)
    
    return smoothed


def estimate_echo_times_optimal(echo1: np.ndarray, echo2: np.ndarray,
                               te1: float, te2: float,
                               target_phase_diff: float = np.pi/2) -> Tuple[float, float]:
    """
    Estimate optimal echo times for B0 field map estimation.
    
    Parameters:
    -----------
    echo1 : np.ndarray
        First echo magnitude image
    echo2 : np.ndarray
        Second echo magnitude image
    te1 : float
        Current first echo time
    te2 : float
        Current second echo time
    target_phase_diff : float
        Target phase difference for optimal SNR
        
    Returns:
    --------
    optimal_te1 : float
        Optimal first echo time
    optimal_te2 : float
        Optimal second echo time
    """
    # Calculate current phase difference
    current_phase_diff = np.angle(echo2 * np.conj(echo1))
    mean_phase_diff = np.mean(current_phase_diff)
    
    # Calculate optimal echo time difference
    optimal_delta_te = target_phase_diff / (2 * np.pi * np.abs(mean_phase_diff) / (te2 - te1))
    
    # Adjust echo times while maintaining center
    center_te = (te1 + te2) / 2
    optimal_te1 = center_te - optimal_delta_te / 2
    optimal_te2 = center_te + optimal_delta_te / 2
    
    return optimal_te1, optimal_te2
