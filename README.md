# Denoising Autoencoder Project

## Overview
This project demonstrates the training and deployment of a denoising autoencoder on 100 grayscale images (64×64). It includes:
- A **Colab notebook** for model development, evaluation, and analysis.  
- A **Gradio web application** for interactive denoising.

## What It Does

### 1. Colab Notebook
- **Data Preparation:**  
  - Fetches the Olivetti Faces dataset (400 × 64×64 grayscale images).  
  - Randomly samples 100 images, splits into 80% train / 20% test.  
  - Adds Gaussian noise (σ=0.1) to both sets.  
- **Model Definition:**  
  - Builds a Residual U-Net autoencoder with skip-connections and residual blocks.  
- **Training:**  
  - Trains with MSE loss, using EarlyStopping and ModelCheckpoint to save `best_model.h5`.  
- **Evaluation:**  
  - Computes Mean Squared Error (MSE) on train and test sets.  
  - Plots training vs. validation loss curves.  
  - Visualizes side-by-side comparisons of noisy, reconstructed, and original images.  
- **Analysis:**  
  - Discusses reconstruction errors, loss curves, and visual results to assess generalization.

### 2. Gradio Web App
- **Interactive Interface:**  
  - Upload any image (grayscale or color).  
  - Displays three outputs:  
    1. **Original** (resized to 64×64 grayscale)  
    2. **Noisy** (Gaussian-corrupted)  
    3. **Reconstructed** (denoised by the autoencoder)  
- **Deployment:**  
  - Simple `app.py` script and `requirements.txt` file enable one-step launch and public sharing.

## Model & Tech Stack
- **Model:**  
  - Denoising autoencoder based on a Residual U-Net backbone.  
- **Libraries & Frameworks:**  
  - **TensorFlow / Keras** for model building and training  
  - **scikit-learn** for dataset loading  
  - **OpenCV** for image processing  
  - **NumPy** & **Pandas** for data handling and metrics  
  - **Matplotlib** for plotting loss curves and examples  
  - **Gradio** for the interactive web interface  

## Analysis & Observations
- **Reconstruction Error:**  
  - **Train MSE:** Low, indicating strong fitting on training data.  
  - **Test MSE:** Very close to train MSE, demonstrating good generalization.  
- **Loss Curves:**  
  - Smooth convergence by 50 epochs.  
  - Minimal gap between training and validation curves shows limited overfitting.  
- **Visual Quality:**  
  - Effective noise removal while preserving key features and textures.  
  - Slight blurring of very fine details suggests exploring perceptual (SSIM) or adversarial losses in future work.

**Conclusion:**  
The Residual U-Net autoencoder exhibits strong denoising performance and generalization. Future enhancements could include multi-noise conditioning, alternative loss functions, or extension to color images and higher resolutions.
