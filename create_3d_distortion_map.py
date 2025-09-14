"""
Create 3D visualization of the geometrical distortion map.

This script generates interactive 3D visualizations of the B0 field map
and distortion map from the Siemens MRI analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path


def load_distortion_data():
    """Load the distortion analysis results."""
    results_path = Path("results")
    
    print("📊 Loading Distortion Data for 3D Visualization")
    print("=" * 60)
    
    # Load data
    fieldmap_hz = np.load(results_path / "corrected_b0_fieldmap_hz.npy")
    distortion_mm = np.load(results_path / "corrected_distortion_map_mm.npy")
    metadata = np.load(results_path / "metadata.npy", allow_pickle=True).item()
    
    print(f"✅ Data loaded successfully!")
    print(f"   📏 Data shape: {fieldmap_hz.shape}")
    print(f"   📏 Voxel size: {metadata['voxel_size_mm']}")
    print(f"   📊 Distortion range: {np.min(distortion_mm):.2f} to {np.max(distortion_mm):.2f} mm")
    
    return fieldmap_hz, distortion_mm, metadata


def create_3d_matplotlib_visualization(fieldmap_hz, distortion_mm, metadata):
    """Create 3D visualization using matplotlib."""
    print("\n🎨 Creating 3D Matplotlib Visualization")
    print("=" * 50)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Get voxel size for proper scaling
    voxel_size = metadata['voxel_size_mm']
    
    # Create coordinate grids
    x = np.arange(fieldmap_hz.shape[0]) * voxel_size[0]
    y = np.arange(fieldmap_hz.shape[1]) * voxel_size[1]
    z = np.arange(fieldmap_hz.shape[2]) * voxel_size[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create mask for non-zero values
    mask = distortion_mm != 0
    
    # 1. B0 Field Map 3D
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(X[mask], Y[mask], Z[mask], 
                          c=fieldmap_hz[mask], 
                          cmap='RdBu_r', 
                          s=1, 
                          alpha=0.6)
    ax1.set_title('B0 Field Map (Hz)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20)
    
    # 2. Distortion Map 3D
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(X[mask], Y[mask], Z[mask], 
                          c=distortion_mm[mask], 
                          cmap='RdBu_r', 
                          s=1, 
                          alpha=0.6)
    ax2.set_title('Distortion Map (mm)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20)
    
    # 3. Absolute Distortion 3D
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    scatter3 = ax3.scatter(X[mask], Y[mask], Z[mask], 
                          c=np.abs(distortion_mm[mask]), 
                          cmap='hot', 
                          s=1, 
                          alpha=0.6)
    ax3.set_title('Absolute Distortion (mm)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5, aspect=20)
    
    # 4. Distortion Severity 3D
    severity_map = np.zeros_like(distortion_mm)
    severity_map[np.abs(distortion_mm) > 5.0] = 4  # Severe
    severity_map[(np.abs(distortion_mm) > 2.0) & (np.abs(distortion_mm) <= 5.0)] = 3  # Moderate
    severity_map[(np.abs(distortion_mm) > 1.0) & (np.abs(distortion_mm) <= 2.0)] = 2  # Mild
    severity_map[(np.abs(distortion_mm) > 0.0) & (np.abs(distortion_mm) <= 1.0)] = 1  # Minimal
    
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    scatter4 = ax4.scatter(X[mask], Y[mask], Z[mask], 
                          c=severity_map[mask], 
                          cmap='RdYlGn', 
                          s=1, 
                          alpha=0.6,
                          vmin=0, vmax=4)
    ax4.set_title('Distortion Severity', fontsize=14, fontweight='bold')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_zlabel('Z (mm)')
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.5, aspect=20)
    cbar4.set_ticks([0, 1, 2, 3, 4])
    cbar4.set_ticklabels(['None', 'Minimal', 'Mild', 'Moderate', 'Severe'])
    
    # 5. Isosurface of high distortion
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    high_distortion_mask = np.abs(distortion_mm) > 3.0
    if np.any(high_distortion_mask):
        scatter5 = ax5.scatter(X[high_distortion_mask], Y[high_distortion_mask], Z[high_distortion_mask], 
                              c=np.abs(distortion_mm[high_distortion_mask]), 
                              cmap='Reds', 
                              s=2, 
                              alpha=0.8)
        ax5.set_title('High Distortion Regions (>3mm)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('X (mm)')
        ax5.set_ylabel('Y (mm)')
        ax5.set_zlabel('Z (mm)')
        plt.colorbar(scatter5, ax=ax5, shrink=0.5, aspect=20)
    else:
        ax5.text(0.5, 0.5, 0.5, 'No high distortion\nregions found', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('High Distortion Regions (>3mm)', fontsize=14, fontweight='bold')
    
    # 6. Volume rendering style
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    # Sample every 2nd voxel for better performance
    sample_mask = mask[::2, ::2, ::2]
    X_sample = X[::2, ::2, ::2]
    Y_sample = Y[::2, ::2, ::2]
    Z_sample = Z[::2, ::2, ::2]
    distortion_sample = distortion_mm[::2, ::2, ::2]
    
    scatter6 = ax6.scatter(X_sample[sample_mask], Y_sample[sample_mask], Z_sample[sample_mask], 
                          c=np.abs(distortion_sample[sample_mask]), 
                          cmap='plasma', 
                          s=0.5, 
                          alpha=0.4)
    ax6.set_title('Volume Rendering Style', fontsize=14, fontweight='bold')
    ax6.set_xlabel('X (mm)')
    ax6.set_ylabel('Y (mm)')
    ax6.set_zlabel('Z (mm)')
    plt.colorbar(scatter6, ax=ax6, shrink=0.5, aspect=20)
    
    plt.tight_layout()
    plt.savefig('results/3d_distortion_visualization.png', dpi=300, bbox_inches='tight')
    print("   ✅ 3D matplotlib visualization saved to 'results/3d_distortion_visualization.png'")
    
    return fig


def create_3d_plotly_visualization(fieldmap_hz, distortion_mm, metadata):
    """Create interactive 3D visualization using Plotly."""
    print("\n🎨 Creating Interactive 3D Plotly Visualization")
    print("=" * 50)
    
    # Get voxel size for proper scaling
    voxel_size = metadata['voxel_size_mm']
    
    # Create coordinate grids
    x = np.arange(fieldmap_hz.shape[0]) * voxel_size[0]
    y = np.arange(fieldmap_hz.shape[1]) * voxel_size[1]
    z = np.arange(fieldmap_hz.shape[2]) * voxel_size[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create mask for phantom data only (exclude background)
    # Use a combination of distortion threshold and spatial constraints
    phantom_threshold = 3.0  # mm - only show distortion > 3.0mm
    
    # Also create a spatial mask to exclude outer regions
    center_x, center_y, center_z = fieldmap_hz.shape[0]//2, fieldmap_hz.shape[1]//2, fieldmap_hz.shape[2]//2
    max_radius = min(center_x, center_y) * 0.8  # Keep only central 80% of the volume
    
    # Create spatial mask
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(fieldmap_hz.shape[0]) - center_x,
        np.arange(fieldmap_hz.shape[1]) - center_y, 
        np.arange(fieldmap_hz.shape[2]) - center_z,
        indexing='ij'
    )
    spatial_mask = (x_coords**2 + y_coords**2)**0.5 < max_radius
    
    # Combine distortion threshold and spatial mask
    distortion_mask = np.abs(distortion_mm) > phantom_threshold
    mask = distortion_mask & spatial_mask
    
    # Sample data for better performance (every 2nd voxel)
    sample_mask = mask[::2, ::2, ::2]
    X_sample = X[::2, ::2, ::2]
    Y_sample = Y[::2, ::2, ::2]
    Z_sample = Z[::2, ::2, ::2]
    fieldmap_sample = fieldmap_hz[::2, ::2, ::2]
    distortion_sample = distortion_mm[::2, ::2, ::2]
    
    # Create single figure for distortion map only
    fig = go.Figure()
    
    # Distortion Map (single, larger view)
    fig.add_trace(
        go.Scatter3d(
            x=X_sample[sample_mask].flatten(),
            y=Y_sample[sample_mask].flatten(),
            z=Z_sample[sample_mask].flatten(),
            mode='markers',
            marker=dict(
                size=3,
                color=distortion_sample[sample_mask].flatten(),
                colorscale='RdBu',
                reversescale=True,
                opacity=0.7,
                colorbar=dict(
                    title="Distortion (mm)",
                    tickmode="auto",
                    nticks=10
                )
            ),
            name='Distortion Map',
            hovertemplate='X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<br>Distortion: %{marker.color:.2f} mm<br><extra></extra>'
        )
    )
    
    # Update layout for single 3D scene
    fig.update_layout(
        title=dict(
            text="3D Phantom Distortion Map (mm) - MRsim QC",
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='lightgray'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='lightgray'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='lightgray')
        ),
        height=700,
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),
        font=dict(family="Arial, sans-serif", size=12, color="black")
    )
    
    # Save as HTML
    fig.write_html('results/3d_distortion_interactive.html')
    print("   ✅ Interactive 3D visualization saved to 'results/3d_distortion_interactive.html'")
    
    return fig


def create_3d_isosurface_visualization(distortion_mm, metadata):
    """Create 3D isosurface visualization of distortion levels."""
    print("\n🎨 Creating 3D Isosurface Visualization")
    print("=" * 50)
    
    # Get voxel size for proper scaling
    voxel_size = metadata['voxel_size_mm']
    
    # Create coordinate grids
    x = np.arange(distortion_mm.shape[0]) * voxel_size[0]
    y = np.arange(distortion_mm.shape[1]) * voxel_size[1]
    z = np.arange(distortion_mm.shape[2]) * voxel_size[2]
    
    # Create isosurfaces for different distortion levels
    fig = go.Figure()
    
    # Define isosurface levels
    levels = [1.0, 2.0, 3.0, 4.0]  # mm
    colors = ['green', 'yellow', 'orange', 'red']
    opacities = [0.3, 0.4, 0.5, 0.6]
    
    for i, (level, color, opacity) in enumerate(zip(levels, colors, opacities)):
        # Create isosurface
        fig.add_trace(go.Isosurface(
            x=x,
            y=y,
            z=z,
            value=np.abs(distortion_mm),
            isomin=level,
            isomax=level + 0.5,
            surface_count=1,
            colorscale=[[0, color], [1, color]],
            opacity=opacity,
            name=f'{level}mm isosurface',
            hovertemplate=f'Distortion Level: {level}mm<br><extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="3D Isosurface Visualization of Distortion Levels",
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    # Save as HTML
    fig.write_html('results/3d_isosurface_distortion.html')
    print("   ✅ 3D isosurface visualization saved to 'results/3d_isosurface_distortion.html'")
    
    return fig


def main():
    """Main function to create 3D visualizations."""
    print("🎨 Creating 3D Distortion Map Visualizations")
    print("=" * 70)
    
    try:
        # Load data
        fieldmap_hz, distortion_mm, metadata = load_distortion_data()
        
        # Create matplotlib 3D visualization
        create_3d_matplotlib_visualization(fieldmap_hz, distortion_mm, metadata)
        
        # Create interactive Plotly 3D visualization
        create_3d_plotly_visualization(fieldmap_hz, distortion_mm, metadata)
        
        # Create isosurface visualization
        create_3d_isosurface_visualization(distortion_mm, metadata)
        
        print("\n" + "=" * 70)
        print("🎉 3D VISUALIZATIONS COMPLETE!")
        print("=" * 70)
        print("📁 Files generated:")
        print("   • results/3d_distortion_visualization.png - Static 3D matplotlib plots")
        print("   • results/3d_distortion_interactive.html - Interactive 3D Plotly visualization")
        print("   • results/3d_isosurface_distortion.html - 3D isosurface visualization")
        
        print(f"\n🎯 3D Visualization Features:")
        print(f"   • B0 field map in 3D space")
        print(f"   • Geometrical distortion map in mm")
        print(f"   • Absolute distortion magnitude")
        print(f"   • Distortion severity classification")
        print(f"   • Interactive rotation and zoom")
        print(f"   • Isosurface levels for different distortion thresholds")
        
        print(f"\n🚀 How to view:")
        print(f"   • Open the .html files in your web browser for interactive 3D viewing")
        print(f"   • Use mouse to rotate, zoom, and pan the 3D models")
        print(f"   • Hover over points to see exact values")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
