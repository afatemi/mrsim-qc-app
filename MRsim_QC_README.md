# MRsim QC - MRI Distortion Analysis Web Application

A comprehensive web application for analyzing MRI geometrical distortion from DICOM files. Upload your magnitude images at different echo times and phase difference data to get detailed distortion analysis and correction maps.

## 🚀 Quick Start

### 1. **Access the Application**
- **URL**: http://localhost:5001
- **Status**: ✅ Currently Running

### 2. **Upload Your Data**
Required DICOM files:
- **Magnitude images at TE1** (e.g., 10ms)
- **Magnitude images at TE2** (e.g., 15ms) 
- **Phase difference images**

### 3. **Get Results**
- Comprehensive distortion analysis
- B0 field maps in Hz
- Geometrical distortion maps in mm
- Clinical assessment and recommendations
- Downloadable results packages

## 📊 Features

### **Analysis Capabilities**
- ✅ **B0 Field Map Estimation** - From multi-echo DICOM data
- ✅ **Phase Unwrapping** - Advanced algorithms for accurate field mapping
- ✅ **Distortion Quantification** - Precise measurements in millimeters
- ✅ **Severity Classification** - Minimal, Mild, Moderate, Severe categories
- ✅ **Clinical Assessment** - Automated recommendations

### **Visualizations**
- ✅ **2D Analysis Maps** - Field maps, distortion maps, severity maps
- ✅ **3D Distortion Visualization** - Interactive 3D scatter plots
- ✅ **Statistical Plots** - Histograms and distribution analysis
- ✅ **Multi-slice Views** - Comprehensive spatial analysis

### **Output Formats**
- ✅ **ZIP Package** - Complete analysis results
- ✅ **NumPy Arrays** - B0 field maps and distortion maps (.npy)
- ✅ **PNG Images** - High-resolution visualizations
- ✅ **JSON Statistics** - Machine-readable analysis data

## 🎯 How to Use

### **Step 1: Upload Files**
1. Go to http://localhost:5001
2. Drag and drop DICOM files or click "Browse Files"
3. Upload magnitude images from both echo times
4. Upload phase difference images
5. Click "Start Analysis"

### **Step 2: View Results**
- **Quick Stats**: Max distortion, mean distortion, percentiles
- **Visualizations**: 2D maps and 3D plots
- **Clinical Assessment**: Automated recommendations
- **Severity Analysis**: Percentage breakdown by distortion level

### **Step 3: Download Results**
- **Complete Package**: ZIP file with all results
- **Individual Files**: Field maps, distortion maps
- **Use in Pipeline**: Import .npy files for further processing

## 📈 Example Results

Based on your Siemens 1.5T phantom data:
- **Maximum Distortion**: 4.27 mm
- **Mean Distortion**: 2.13 mm  
- **95th Percentile**: 3.93 mm
- **Severe Distortion**: 0% of volume
- **Moderate Distortion**: 58.2% of volume

## 🔧 Technical Details

### **Scan Parameters Supported**
- **Echo Times**: Dual-echo sequences (e.g., 10ms and 15ms)
- **Phase Difference**: Direct phase difference data
- **Field Strengths**: 1.5T, 3T, and higher
- **Sequences**: EPI, gradient echo with phase information

### **Algorithms Used**
- **Field Map Estimation**: Phase difference method
- **Phase Unwrapping**: Region growing and Laplacian methods
- **Distortion Calculation**: Echo spacing and voxel size calibrated
- **Quality Assessment**: SNR, outlier detection, spatial analysis

### **Data Processing**
- **DICOM Reading**: Automatic metadata extraction
- **Scaling Correction**: Handles different DICOM scaling formats
- **Brain Masking**: Automatic tissue segmentation
- **Spatial Calibration**: Accurate mm measurements

## 🌐 API Endpoints

### **Web Interface**
- `GET /` - Main upload page
- `POST /upload` - File upload and processing
- `GET /results/<session_id>` - Results display page

### **File Downloads**
- `GET /download/<session_id>/zip` - Complete results package
- `GET /download/<session_id>/fieldmap` - B0 field map (.npy)
- `GET /download/<session_id>/distortion` - Distortion map (.npy)

### **API Access**
- `GET /api/stats/<session_id>` - JSON statistics
- `GET /web_results/<session_id>/<filename>` - Individual result files

## 📝 Session Management

Each analysis creates a unique session with:
- **Session ID**: Timestamp-based (YYYYMMDD_HHMMSS)
- **Temporary Storage**: Automatic cleanup of uploaded files
- **Result Persistence**: Analysis results saved for download
- **Security**: Session-based access control

## 🔍 Quality Control Features

### **Automatic Validation**
- DICOM file format checking
- Metadata consistency verification
- Image dimension validation
- Echo time parameter extraction

### **Quality Metrics**
- **Field Map SNR**: Signal-to-noise ratio assessment
- **Outlier Detection**: Statistical outlier identification
- **Spatial Consistency**: Neighboring voxel analysis
- **Clinical Thresholds**: Comparison to established limits

## 💡 Tips for Best Results

### **Data Acquisition**
- Use consistent scan parameters
- Ensure proper phase encoding direction
- Optimize echo times for field strength
- Maintain stable shimming

### **File Organization**
- Keep magnitude and phase files together
- Use consistent naming conventions
- Verify DICOM metadata completeness
- Check for motion artifacts

## 🚨 Troubleshooting

### **Common Issues**
- **Upload Fails**: Check DICOM file format and size
- **No Results**: Verify phase difference data presence
- **High Distortion Values**: Check data scaling and units
- **Missing Images**: Ensure all required files uploaded

### **Support**
- Check browser console for JavaScript errors
- Verify network connectivity to localhost:5001
- Restart application if processing stalls
- Contact support for persistent issues

## 🔄 Updates and Maintenance

### **Current Version**: 1.0.0
### **Last Updated**: September 2024
### **Dependencies**: Flask, NumPy, SciPy, Matplotlib, PyDICOM

The application is actively maintained and updated with the latest MRI distortion correction algorithms.
