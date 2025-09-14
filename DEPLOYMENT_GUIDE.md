# MRsim QC Web App - Deployment Guide

## 🚀 Quick Deployment Options

### Option 1: Railway (Recommended - Easiest)

1. **Sign up at [Railway.app](https://railway.app)**
2. **Connect your GitHub account**
3. **Create a new project from GitHub**
4. **Select this repository**
5. **Railway will automatically detect it's a Python app**
6. **Deploy!** - You'll get a URL like `https://your-app-name.railway.app`

### Option 2: Render (Free tier available)

1. **Sign up at [Render.com](https://render.com)**
2. **Create a new Web Service**
3. **Connect your GitHub repository**
4. **Use these settings:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python3 mrsim_qc_app.py`
5. **Deploy!**

### Option 3: Heroku (Free tier discontinued, but still works)

1. **Install Heroku CLI**
2. **Login:** `heroku login`
3. **Create app:** `heroku create your-app-name`
4. **Deploy:** `git push heroku main`

## 📋 Pre-deployment Checklist

✅ **Files ready for deployment:**
- `Procfile` - Tells the platform how to run your app
- `runtime.txt` - Specifies Python version
- `requirements.txt` - Lists all dependencies
- `mrsim_qc_app.py` - Main Flask application

## 🌐 After Deployment

Once deployed, you'll get a public URL like:
- Railway: `https://mrsim-qc.railway.app`
- Render: `https://mrsim-qc.onrender.com`
- Heroku: `https://your-app-name.herokuapp.com`

**Share this URL with your colleague!**

## 🔧 Troubleshooting

- **If deployment fails:** Check that all dependencies are in `requirements.txt`
- **If app crashes:** Check the logs in your platform's dashboard
- **If files don't upload:** Make sure the upload directory is writable

## 📱 Usage

Your colleague can now:
1. Visit the public URL
2. Upload their DICOM files (Magnitude TE1, TE2, and Phase Difference)
3. Run the MRI distortion analysis
4. View 2D and 3D results
5. Download all generated files
