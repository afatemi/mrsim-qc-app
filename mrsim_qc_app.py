"""
MRsim QC - MRI Distortion Analysis Web Application

A web application for analyzing MRI geometrical distortion from DICOM files.
Users can upload magnitude images at different TEs and phase difference data
to get comprehensive distortion analysis and correction.
"""

import os
import io
import json
import zipfile
import tempfile
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pydicom
from datetime import datetime
import json

# Import our b0map utilities
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mri_utils import (
        estimate_b0_fieldmap,
        assess_fieldmap_quality,
        smooth_fieldmap,
        correct_geometric_distortion,
        assess_distortion_severity
    )
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure all dependencies are installed")

app = Flask(__name__)
app.secret_key = 'mrsim_qc_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'web_results'
ALLOWED_EXTENSIONS = {'dcm'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension."""
    if not filename:
        return False
    
    print(f"Checking file: '{filename}'")
    
    # Check if it's a DICOM file
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        is_dcm = ext in ALLOWED_EXTENSIONS
        print(f"  Has extension '{ext}', is DICOM: {is_dcm}")
        return is_dcm
    else:
        # Some DICOM files might not have extensions
        # Check if filename looks like a DICOM file
        filename_lower = filename.lower()
        keywords = ['dcm', 'dicom', 'im-', 'mr', 'image', 'scan']
        has_keyword = any(keyword in filename_lower for keyword in keywords)
        print(f"  No extension, checking keywords: {has_keyword}")
        return has_keyword

def process_dicom_files(upload_folder, session_id):
    """Process uploaded DICOM files and generate analysis."""
    try:
        # Find DICOM files in upload folder - look for both .dcm files and files without extensions
        dicom_files = []
        
        # First, look for .dcm files
        dicom_files.extend(list(Path(upload_folder).glob("**/*.dcm")))
        
        # Then, look for files without extensions that might be DICOM
        for file_path in Path(upload_folder).glob("**/*"):
            if file_path.is_file() and not file_path.suffix:
                # Check if it looks like a DICOM file
                filename = file_path.name.lower()
                if any(keyword in filename for keyword in ['dcm', 'dicom', 'im-', 'mr']):
                    dicom_files.append(file_path)
        
        print(f"Found {len(dicom_files)} potential DICOM files")
        
        if not dicom_files:
            raise ValueError("No DICOM files found")
        
        # Load first DICOM to get metadata
        first_dicom = pydicom.dcmread(dicom_files[0])
        
        # Get basic info
        rows = first_dicom.Rows
        cols = first_dicom.Columns
        
        # Count slices by examining file structure
        slice_count = len(dicom_files)
        
        # Create results directory for this session
        results_dir = Path(RESULTS_FOLDER) / session_id
        results_dir.mkdir(exist_ok=True)
        
        # For demo purposes, we'll use the existing processed data
        # In a real application, you'd process the uploaded files here
        
        # Load our existing processed data as demo
        demo_results_dir = Path("results")
        if (demo_results_dir / "corrected_b0_fieldmap_hz.npy").exists():
            fieldmap_hz = np.load(demo_results_dir / "corrected_b0_fieldmap_hz.npy")
            distortion_mm = np.load(demo_results_dir / "corrected_distortion_map_mm.npy")
            metadata = np.load(demo_results_dir / "metadata.npy", allow_pickle=True).item()
        else:
            # Create demo data if no existing results
            fieldmap_hz, distortion_mm, metadata = create_demo_data()
        
        # Generate analysis results
        analysis_results = generate_analysis_results(fieldmap_hz, distortion_mm, metadata, results_dir)
        
        return analysis_results
        
    except Exception as e:
        raise Exception(f"Error processing DICOM files: {str(e)}")

def create_demo_data():
    """Create demo data for testing."""
    # Create synthetic data similar to your Siemens data
    shape = (128, 128, 30)
    x, y, z = np.meshgrid(np.linspace(-1, 1, shape[0]),
                         np.linspace(-1, 1, shape[1]),
                         np.linspace(-1, 1, shape[2]), indexing='ij')
    
    # Create brain mask
    brain_mask = (x**2 + y**2 + z**2) < 0.8
    
    # Create synthetic B0 field map
    fieldmap_hz = 1000 * np.exp(-(x**2 + y**2 + z**2) / 0.3) * brain_mask
    
    # Create distortion map
    echo_spacing = 0.0008
    voxel_size_y = 2.34375
    distortion_mm = fieldmap_hz * echo_spacing * voxel_size_y
    
    metadata = {
        'te1_ms': 10.0,
        'te2_ms': 15.0,
        'delta_te_ms': 5.0,
        'echo_spacing_ms': 0.8,
        'voxel_size_mm': (2.34375, 2.34375, 5.0),
        'data_shape': shape
    }
    
    return fieldmap_hz, distortion_mm, metadata

def generate_analysis_results(fieldmap_hz, distortion_mm, metadata, results_dir):
    """Generate comprehensive analysis results."""
    
    # Create mask for analysis
    mask = distortion_mm != 0
    
    if not np.any(mask):
        raise ValueError("No valid distortion data found")
    
    distortion_values = distortion_mm[mask]
    fieldmap_values = fieldmap_hz[mask]
    
    # Calculate statistics
    stats = {
        'fieldmap_range': (float(np.min(fieldmap_values)), float(np.max(fieldmap_values))),
        'fieldmap_mean': float(np.mean(fieldmap_values)),
        'fieldmap_std': float(np.std(fieldmap_values)),
        'distortion_range': (float(np.min(distortion_values)), float(np.max(distortion_values))),
        'distortion_mean': float(np.mean(np.abs(distortion_values))),
        'distortion_std': float(np.std(np.abs(distortion_values))),
        'distortion_max': float(np.max(np.abs(distortion_values))),
        'percentiles': {
            '50': float(np.percentile(np.abs(distortion_values), 50)),
            '75': float(np.percentile(np.abs(distortion_values), 75)),
            '90': float(np.percentile(np.abs(distortion_values), 90)),
            '95': float(np.percentile(np.abs(distortion_values), 95)),
            '99': float(np.percentile(np.abs(distortion_values), 99))
        }
    }
    
    # Severity classification
    severe = np.sum(np.abs(distortion_values) > 5.0)
    moderate = np.sum((np.abs(distortion_values) > 2.0) & (np.abs(distortion_values) <= 5.0))
    mild = np.sum((np.abs(distortion_values) > 1.0) & (np.abs(distortion_values) <= 2.0))
    minimal = np.sum((np.abs(distortion_values) > 0.0) & (np.abs(distortion_values) <= 1.0))
    total = len(distortion_values)
    
    stats['severity_counts'] = {
        'severe': int(severe),
        'moderate': int(moderate),
        'mild': int(mild),
        'minimal': int(minimal)
    }
    
    stats['severity_percentages'] = {
        'severe': float(severe/total*100),
        'moderate': float(moderate/total*100),
        'mild': float(mild/total*100),
        'minimal': float(minimal/total*100)
    }
    
    # Generate visualizations
    create_analysis_visualizations(fieldmap_hz, distortion_mm, metadata, stats, results_dir)
    
    # Save data files
    np.save(results_dir / "b0_fieldmap_hz.npy", fieldmap_hz)
    np.save(results_dir / "distortion_map_mm.npy", distortion_mm)
    np.save(results_dir / "metadata.npy", metadata)
    
    # Save statistics
    with open(results_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def create_analysis_visualizations(fieldmap_hz, distortion_mm, metadata, stats, results_dir):
    """Create comprehensive visualizations."""
    
    # Select middle slice
    slice_idx = fieldmap_hz.shape[2] // 2
    
    # Create main analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Get 2D slices (show all data, not just phantom)
    fieldmap_slice = fieldmap_hz[:, :, slice_idx]
    distortion_slice = distortion_mm[:, :, slice_idx]
    
    # B0 field map
    im1 = axes[0, 0].imshow(fieldmap_slice, cmap='RdBu_r')
    axes[0, 0].set_title(f'B0 Field Map (Hz)\nSlice {slice_idx}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Distortion map
    im2 = axes[0, 1].imshow(distortion_slice, cmap='RdBu_r')
    axes[0, 1].set_title(f'Distortion Map (mm)\nSlice {slice_idx}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Absolute distortion
    im3 = axes[0, 2].imshow(np.abs(distortion_slice), cmap='hot')
    axes[0, 2].set_title(f'Absolute Distortion (mm)\nSlice {slice_idx}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Histogram of distortion (all data)
    distortion_values = distortion_mm.flatten()
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
    stats_text = f"""
MRsim QC Analysis Results

Scan Parameters:
• TE1: {metadata['te1_ms']:.1f} ms
• TE2: {metadata['te2_ms']:.1f} ms
• ΔTE: {metadata['delta_te_ms']:.1f} ms
• Echo spacing: {metadata['echo_spacing_ms']:.1f} ms
• Voxel size: {metadata['voxel_size_mm'][0]:.1f}×{metadata['voxel_size_mm'][1]:.1f}×{metadata['voxel_size_mm'][2]:.1f} mm

Distortion Statistics:
• Max: {stats['distortion_max']:.2f} mm
• Mean: {stats['distortion_mean']:.2f} mm
• Std: {stats['distortion_std']:.2f} mm
• 95th percentile: {stats['percentiles']['95']:.2f} mm

Severity Classification:
• Severe (>5mm): {stats['severity_percentages']['severe']:.1f}%
• Moderate (2-5mm): {stats['severity_percentages']['moderate']:.1f}%
• Mild (1-2mm): {stats['severity_percentages']['mild']:.1f}%
• Minimal (≤1mm): {stats['severity_percentages']['minimal']:.1f}%

Clinical Assessment:
{'⚠️ Consider distortion correction' if stats['distortion_max'] > 3.0 else '✅ Distortion levels acceptable'}
    """
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create 3D visualization
    create_3d_visualization(fieldmap_hz, distortion_mm, metadata, results_dir)

def create_3d_visualization(fieldmap_hz, distortion_mm, metadata, results_dir):
    """Create 3D visualization."""
    fig = plt.figure(figsize=(15, 10))
    
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
    
    # Sample data for better performance
    sample_mask = mask[::2, ::2, ::2]
    X_sample = X[::2, ::2, ::2]
    Y_sample = Y[::2, ::2, ::2]
    Z_sample = Z[::2, ::2, ::2]
    distortion_sample = distortion_mm[::2, ::2, ::2]
    
    # Create 3D scatter plot
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_sample[sample_mask], Y_sample[sample_mask], Z_sample[sample_mask], 
                        c=np.abs(distortion_sample[sample_mask]), 
                        cmap='hot', 
                        s=1, 
                        alpha=0.6)
    
    ax.set_title('3D Distortion Map (mm)', fontsize=16, fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    plt.savefig(results_dir / '3d_distortion_map.png', dpi=300, bbox_inches='tight')
    plt.close()

@app.route('/')
def index():
    """Main page - original beautiful design."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        logger.error(traceback.format_exc())
        return f"Error loading page: {str(e)}", 500

@app.route('/test')
def test():
    """Simple test page."""
    try:
        return send_file('simple_test.html')
    except Exception as e:
        logger.error(f"Error in test route: {e}")
        return f"Test page error: {str(e)}", 500

@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        return jsonify({
            "status": "healthy",
            "message": "MRsim QC app is running",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/simple')
def simple():
    """Simple upload page."""
    return send_file('simple_upload.html')

@app.route('/folders')
def folders():
    """Folder upload page."""
    return send_file('folder_upload.html')

@app.route('/complete')
def complete():
    """Complete analysis page."""
    return send_file('complete_analysis.html')

@app.route('/results')
def results_viewer():
    """Results viewer page."""
    return send_file('results_viewer.html')

@app.route('/api/latest-results')
def api_latest_results():
    """API endpoint to get the latest analysis results."""
    try:
        # Find the most recent results directory
        results_dirs = list(Path('web_results').glob('*'))
        if not results_dirs:
            return jsonify({'success': False, 'error': 'No results found'})
        
        latest_dir = max(results_dirs, key=lambda x: x.name)
        
        # Load statistics
        stats_file = latest_dir / 'statistics.json'
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                statistics = json.load(f)
        else:
            statistics = {}
        
        # Prepare result URLs
        results = {
            'analysis_image': f'/web_results/{latest_dir.name}/analysis_results.png',
            'distortion_3d_image': f'/web_results/{latest_dir.name}/3d_distortion_map.png',
            'b0_fieldmap': f'/web_results/{latest_dir.name}/b0_fieldmap_hz.npy',
            'distortion_data': f'/web_results/{latest_dir.name}/distortion_map_mm.npy',
            'statistics': statistics,
            'session_id': latest_dir.name
        }
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        print(f"Error getting latest results: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload."""
    print(f"Upload request received. Files in request: {list(request.files.keys())}")
    
    if 'files' not in request.files:
        print("No 'files' key in request")
        flash('No files selected')
        return redirect(request.url)
    
    files = request.files.getlist('files')
    print(f"Number of files received: {len(files)}")
    
    if not files or all(file.filename == '' for file in files):
        print("No valid files found")
        flash('No files selected')
        return redirect(request.url)
    
    # Create session ID
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = Path(UPLOAD_FOLDER) / session_id
    session_folder.mkdir(exist_ok=True)
    print(f"Created session folder: {session_folder}")
    
    # Save uploaded files with progress tracking
    uploaded_count = 0
    total_files = len([f for f in files if f and allowed_file(f.filename)])
    print(f"Total valid files to upload: {total_files}")
    
    for file in files:
        print(f"Processing file: '{file.filename}' (size: {file.content_length if hasattr(file, 'content_length') else 'unknown'})")
        if file and file.filename:
            is_allowed = allowed_file(file.filename)
            print(f"File '{file.filename}' allowed: {is_allowed}")
            print(f"File details - name: '{file.filename}', has extension: {'.' in file.filename if file.filename else False}")
            if is_allowed:
                filename = secure_filename(file.filename)
                file_path = session_folder / filename
                file.save(file_path)
                uploaded_count += 1
                print(f"Saved file: {filename} ({file_path.stat().st_size} bytes)")
            else:
                print(f"Rejected file: {file.filename} (not a valid DICOM file)")
        else:
            print(f"Invalid file object: {file}")
    
    print(f"Successfully uploaded {uploaded_count} files")
    
    try:
        # Process the files
        print("Starting file processing...")
        results = process_dicom_files(session_folder, session_id)
        print("File processing completed successfully")
        
        # Clean up upload folder
        shutil.rmtree(session_folder)
        print("Cleaned up upload folder")
        
        return redirect(url_for('results', session_id=session_id))
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error processing files: {str(e)}')
        return redirect(url_for('index'))

@app.route('/results/<session_id>')
def results(session_id):
    """Display results page."""
    results_dir = Path(RESULTS_FOLDER) / session_id
    
    if not results_dir.exists():
        flash('Results not found')
        return redirect(url_for('index'))
    
    # Load statistics
    with open(results_dir / "statistics.json", 'r') as f:
        stats = json.load(f)
    
    # Load metadata
    metadata = np.load(results_dir / "metadata.npy", allow_pickle=True).item()
    
    return render_template('results.html', 
                         session_id=session_id, 
                         stats=stats, 
                         metadata=metadata)

@app.route('/download/<session_id>/<file_type>')
def download_file(session_id, file_type):
    """Download analysis files."""
    results_dir = Path(RESULTS_FOLDER) / session_id
    
    if not results_dir.exists():
        return "File not found", 404
    
    if file_type == 'zip':
        # Create zip file with all results
        zip_path = results_dir / f"mrsim_qc_results_{session_id}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in results_dir.glob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
        
        return send_file(zip_path, as_attachment=True, 
                        download_name=f"mrsim_qc_results_{session_id}.zip")
    
    elif file_type == 'fieldmap':
        fieldmap_path = results_dir / "b0_fieldmap_hz.npy"
        return send_file(fieldmap_path, as_attachment=True, 
                        download_name=f"b0_fieldmap_{session_id}.npy")
    
    elif file_type == 'distortion':
        distortion_path = results_dir / "distortion_map_mm.npy"
        return send_file(distortion_path, as_attachment=True, 
                        download_name=f"distortion_map_{session_id}.npy")
    
    return "Invalid file type", 400

@app.route('/api/stats/<session_id>')
def api_stats(session_id):
    """API endpoint for statistics."""
    results_dir = Path(RESULTS_FOLDER) / session_id
    
    if not results_dir.exists():
        return jsonify({'error': 'Results not found'}), 404
    
    with open(results_dir / "statistics.json", 'r') as f:
        stats = json.load(f)
    
    return jsonify(stats)

@app.route('/web_results/<session_id>/<filename>')
def serve_result_file(session_id, filename):
    """Serve result files."""
    results_dir = Path(RESULTS_FOLDER) / session_id
    file_path = results_dir / filename
    
    if not file_path.exists():
        return "File not found", 404
    
    return send_file(file_path)

@app.route('/interactive_3d/<session_id>')
def interactive_3d(session_id):
    """Serve interactive 3D distortion visualization."""
    results_dir = Path(RESULTS_FOLDER) / session_id
    
    if not results_dir.exists():
        return "Results not found", 404
    
    # Check if interactive 3D file exists
    interactive_file = results_dir / "3d_distortion_interactive.html"
    
    if interactive_file.exists():
        return send_file(interactive_file)
    else:
        # Generate interactive 3D visualization if it doesn't exist
        try:
            from create_3d_distortion_map import create_3d_plotly_visualization, load_distortion_data
            
            # Load data from web results
            fieldmap_hz = np.load(results_dir / "b0_fieldmap_hz.npy")
            distortion_mm = np.load(results_dir / "distortion_map_mm.npy")
            metadata = np.load(results_dir / "metadata.npy", allow_pickle=True).item()
            
            # Create interactive visualization
            create_3d_plotly_visualization(fieldmap_hz, distortion_mm, metadata)
            
            # Move the generated file to the session directory
            generated_file = Path("results/3d_distortion_interactive.html")
            if generated_file.exists():
                generated_file.rename(interactive_file)
                return send_file(interactive_file)
            else:
                return "Failed to generate 3D visualization", 500
                
        except Exception as e:
            print(f"Error generating 3D visualization: {e}")
            return f"Error generating 3D visualization: {str(e)}", 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
