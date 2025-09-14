"""
Advanced phase unwrapping algorithms for MRI field mapping.

This module provides specialized phase unwrapping methods optimized for
B0 field map estimation and correction.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Tuple, Optional, List
import warnings


def phase_unwrap_laplacian(phase: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Laplacian-based phase unwrapping algorithm.
    
    This method solves the Poisson equation to unwrap phase by minimizing
    the difference between the wrapped and unwrapped phase gradients.
    
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
    
    # Calculate wrapped phase gradients
    grad_x = np.diff(phase, axis=0)
    grad_y = np.diff(phase, axis=1)
    
    # Wrap gradients to [-pi, pi]
    grad_x = np.arctan2(np.sin(grad_x), np.cos(grad_x))
    grad_y = np.arctan2(np.sin(grad_y), np.cos(grad_y))
    
    # Calculate Laplacian
    laplacian = np.zeros_like(phase)
    # Ensure proper shape handling for gradients
    grad_x_padded = np.zeros_like(phase)
    grad_y_padded = np.zeros_like(phase)
    grad_x_padded[:-1, :] = grad_x
    grad_y_padded[:, :-1] = grad_y
    
    laplacian[:-1, :-1] = (grad_x_padded[:-1, :-1] - grad_x_padded[:-1, 1:] - 
                          grad_y_padded[:-1, :-1] + grad_y_padded[1:, :-1])
    
    # Solve Poisson equation using discrete cosine transform
    unwrapped = _solve_poisson_dct(laplacian, mask)
    
    return unwrapped


def _solve_poisson_dct(laplacian: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Solve Poisson equation using DCT."""
    from scipy.fft import dctn, idctn
    
    # Apply DCT to Laplacian
    laplacian_dct = dctn(laplacian, norm='ortho')
    
    # Create frequency domain operator
    shape = laplacian.shape
    kx = np.arange(shape[0])[:, np.newaxis]
    ky = np.arange(shape[1])[np.newaxis, :]
    
    # DCT frequency domain Laplacian operator
    operator = 2 * (np.cos(np.pi * kx / shape[0]) + 
                   np.cos(np.pi * ky / shape[1]) - 2)
    
    # Avoid division by zero
    operator[0, 0] = 1
    
    # Solve in frequency domain
    solution_dct = laplacian_dct / operator
    solution_dct[0, 0] = 0  # Set DC component to zero
    
    # Inverse DCT
    unwrapped = idctn(solution_dct, norm='ortho')
    
    # Apply mask
    unwrapped = unwrapped * mask
    
    return unwrapped


def phase_unwrap_quality_guided(phase: np.ndarray,
                               magnitude: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Quality-guided phase unwrapping algorithm.
    
    This method uses magnitude information to guide the unwrapping process,
    starting from high-quality regions and propagating to lower-quality areas.
    
    Parameters:
    -----------
    phase : np.ndarray
        Wrapped phase image
    magnitude : np.ndarray
        Magnitude image for quality assessment
    mask : np.ndarray, optional
        Region of interest mask
        
    Returns:
    --------
    unwrapped_phase : np.ndarray
        Unwrapped phase image
    """
    if mask is None:
        mask = np.ones_like(phase, dtype=bool)
    
    # Calculate quality map
    quality = _calculate_quality_map(phase, magnitude)
    
    # Initialize unwrapped phase
    unwrapped = phase.copy()
    shape = phase.shape
    
    # Create priority queue based on quality
    visited = np.zeros(shape, dtype=bool)
    priority_queue = []
    
    # Find starting point (highest quality)
    valid_quality = quality * mask
    if np.any(valid_quality):
        start_idx = np.unravel_index(np.argmax(valid_quality), shape)
        priority_queue.append((start_idx, quality[start_idx]))
        visited[start_idx] = True
    
    # 8-connectivity for 2D, 26-connectivity for 3D
    if len(shape) == 2:
        neighbors = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] 
                    if not (dx == 0 and dy == 0)]
    else:
        neighbors = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
                    if not (dx == 0 and dy == 0 and dz == 0)]
    
    while priority_queue:
        # Sort by quality (highest first)
        priority_queue.sort(key=lambda x: x[1], reverse=True)
        current, _ = priority_queue.pop(0)
        
        for neighbor_offset in neighbors:
            neighbor = tuple(c + o for c, o in zip(current, neighbor_offset))
            
            # Check bounds
            if all(0 <= n < s for n, s in zip(neighbor, shape)):
                if not visited[neighbor] and mask[neighbor]:
                    # Calculate phase difference
                    phase_diff = unwrapped[neighbor] - unwrapped[current]
                    
                    # Unwrap if necessary
                    if phase_diff > np.pi:
                        unwrapped[neighbor] -= 2 * np.pi
                    elif phase_diff < -np.pi:
                        unwrapped[neighbor] += 2 * np.pi
                    
                    visited[neighbor] = True
                    priority_queue.append((neighbor, quality[neighbor]))
    
    return unwrapped


def _calculate_quality_map(phase: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """Calculate quality map for phase unwrapping."""
    # Phase derivative variance (PDV) quality measure
    grad_x = np.diff(phase, axis=0)
    grad_y = np.diff(phase, axis=1)
    
    # Wrap gradients
    grad_x = np.arctan2(np.sin(grad_x), np.cos(grad_x))
    grad_y = np.arctan2(np.sin(grad_y), np.cos(grad_y))
    
    # Calculate variance of gradients
    pdv = np.zeros_like(phase)
    # Create arrays with proper shapes
    grad_x_00 = grad_x[:-1, :-1]
    grad_x_01 = grad_x[:-1, 1:]
    grad_y_00 = grad_y[:-1, :-1]
    grad_y_10 = grad_y[1:, :-1]
    
    # Calculate variance element-wise
    for i in range(grad_x_00.shape[0]):
        for j in range(grad_x_00.shape[1]):
            values = [grad_x_00[i, j], grad_x_01[i, j], grad_y_00[i, j], grad_y_10[i, j]]
            pdv[i, j] = np.var(values)
    
    # Combine with magnitude information
    quality = magnitude / (1 + pdv)
    
    return quality


def phase_unwrap_multiresolution(phase: np.ndarray,
                                mask: Optional[np.ndarray] = None,
                                levels: int = 3) -> np.ndarray:
    """
    Multi-resolution phase unwrapping algorithm.
    
    This method unwraps phase at multiple resolutions, starting from
    coarse resolution and refining at higher resolutions.
    
    Parameters:
    -----------
    phase : np.ndarray
        Wrapped phase image
    mask : np.ndarray, optional
        Region of interest mask
    levels : int
        Number of resolution levels
        
    Returns:
    --------
    unwrapped_phase : np.ndarray
        Unwrapped phase image
    """
    if mask is None:
        mask = np.ones_like(phase, dtype=bool)
    
    # Start with coarsest resolution
    current_phase = phase.copy()
    current_mask = mask.copy()
    
    for level in range(levels):
        # Downsample
        if level > 0:
            scale_factor = 2 ** level
            current_phase = ndi.zoom(current_phase, 1/scale_factor, order=1)
            current_mask = ndi.zoom(current_mask.astype(float), 1/scale_factor, order=0) > 0.5
        
        # Unwrap at current resolution
        if level == 0:
            # Use region growing for finest resolution
            unwrapped = phase_unwrap_region_growing(current_phase, mask=current_mask)
        else:
            # Use Laplacian method for coarser resolutions
            unwrapped = phase_unwrap_laplacian(current_phase, mask=current_mask)
        
        # Upsample to next resolution
        if level < levels - 1:
            next_scale = 2 ** (level + 1)
            unwrapped = ndi.zoom(unwrapped, next_scale, order=1)
            current_phase = ndi.zoom(phase, 1/next_scale, order=1)
            current_mask = ndi.zoom(mask.astype(float), 1/next_scale, order=0) > 0.5
    
    return unwrapped


def phase_unwrap_region_growing(phase: np.ndarray, 
                               mask: Optional[np.ndarray] = None,
                               seed: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """
    Improved region growing phase unwrapping with better seed selection.
    
    Parameters:
    -----------
    phase : np.ndarray
        Wrapped phase image
    mask : np.ndarray, optional
        Region of interest mask
    seed : tuple, optional
        Starting point coordinates
        
    Returns:
    --------
    unwrapped_phase : np.ndarray
        Unwrapped phase image
    """
    if mask is None:
        mask = np.ones_like(phase, dtype=bool)
    
    unwrapped = phase.copy()
    shape = phase.shape
    
    # Find seed point if not provided
    if seed is None:
        # Use center of mass of masked region
        center = ndi.center_of_mass(mask.astype(float))
        seed = tuple(int(c) for c in center)
    
    # Initialize
    visited = np.zeros(shape, dtype=bool)
    queue = [seed]
    visited[seed] = True
    
    # Define connectivity
    if len(shape) == 2:
        neighbors = [(1,0), (-1,0), (0,1), (0,-1)]
    else:
        neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    
    while queue:
        current = queue.pop(0)
        
        for neighbor_offset in neighbors:
            neighbor = tuple(c + o for c, o in zip(current, neighbor_offset))
            
            # Check bounds and mask
            if (all(0 <= n < s for n, s in zip(neighbor, shape)) and
                not visited[neighbor] and mask[neighbor]):
                
                # Calculate phase difference
                phase_diff = unwrapped[neighbor] - unwrapped[current]
                
                # Unwrap if necessary
                if phase_diff > np.pi:
                    unwrapped[neighbor] -= 2 * np.pi
                elif phase_diff < -np.pi:
                    unwrapped[neighbor] += 2 * np.pi
                
                visited[neighbor] = True
                queue.append(neighbor)
    
    return unwrapped


def detect_phase_jumps(phase: np.ndarray, 
                      threshold: float = np.pi) -> np.ndarray:
    """
    Detect phase jumps in wrapped phase image.
    
    Parameters:
    -----------
    phase : np.ndarray
        Wrapped phase image
    threshold : float
        Threshold for detecting phase jumps
        
    Returns:
    --------
    jump_map : np.ndarray
        Binary map indicating phase jump locations
    """
    # Calculate phase differences
    grad_x = np.diff(phase, axis=0)
    grad_y = np.diff(phase, axis=1)
    
    # Wrap gradients
    grad_x = np.arctan2(np.sin(grad_x), np.cos(grad_x))
    grad_y = np.arctan2(np.sin(grad_y), np.cos(grad_y))
    
    # Detect jumps
    jump_map = np.zeros_like(phase, dtype=bool)
    jump_map[:-1, :-1] = (np.abs(grad_x[:-1, :-1]) > threshold) | \
                        (np.abs(grad_y[:-1, :-1]) > threshold)
    
    return jump_map


def estimate_unwrapping_reliability(phase: np.ndarray,
                                  magnitude: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate reliability of phase unwrapping at each pixel.
    
    Parameters:
    -----------
    phase : np.ndarray
        Wrapped phase image
    magnitude : np.ndarray
        Magnitude image
    mask : np.ndarray, optional
        Region of interest mask
        
    Returns:
    --------
    reliability : np.ndarray
        Reliability map (0-1, higher is more reliable)
    """
    if mask is None:
        mask = np.ones_like(phase, dtype=bool)
    
    # Calculate phase derivative variance
    grad_x = np.diff(phase, axis=0)
    grad_y = np.diff(phase, axis=1)
    
    grad_x = np.arctan2(np.sin(grad_x), np.cos(grad_x))
    grad_y = np.arctan2(np.sin(grad_y), np.cos(grad_y))
    
    # Calculate local variance
    pdv = np.zeros_like(phase)
    # Create arrays with proper shapes
    grad_x_00 = grad_x[:-1, :-1]
    grad_x_01 = grad_x[:-1, 1:]
    grad_y_00 = grad_y[:-1, :-1]
    grad_y_10 = grad_y[1:, :-1]
    
    # Calculate variance element-wise
    for i in range(grad_x_00.shape[0]):
        for j in range(grad_x_00.shape[1]):
            values = [grad_x_00[i, j], grad_x_01[i, j], grad_y_00[i, j], grad_y_10[i, j]]
            pdv[i, j] = np.var(values)
    
    # Normalize magnitude
    norm_magnitude = magnitude / np.max(magnitude[mask])
    
    # Calculate reliability
    reliability = norm_magnitude / (1 + pdv)
    reliability = np.clip(reliability, 0, 1)
    
    # Apply mask
    reliability = reliability * mask
    
    return reliability
