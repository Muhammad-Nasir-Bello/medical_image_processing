# Interpretable MRI Brain Tumor Classification

This repository contains the source code and trained model for an MSc thesis project at the **University of Debrecen**. The system uses a modified **ResNet-18** architecture to classify MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

## Live Application
Access the interactive diagnostic tool here: 
[Streamlit Web App](https://muhammad-nasir-bello-medical-image-processing-app-li2yjo.streamlit.app/)

## Key Results
- **Test Accuracy**: 98.86% 
- **Macro AUC**: 0.9993 
- **Recall (Healthy Scans)**: 0.9967 (demonstrating high screening safety) 

## Explainability (Grad-CAM)
To move beyond "black-box" AI, this project integrates **Grad-CAM** visualizations. This allows the model to highlight the specific anatomical regions (such as the sellar region for pituitary tumors) that drive its classification decisions.

## Repository Structure
- `src/`: Core model architecture and preprocessing scripts.
- `experiments/run1/`: Final model weights (`best.pt`) and performance curves.
- `app.py`: Streamlit application code.
- `requirements.txt`: Necessary libraries for deployment.

## Academic Context
**Candidate**: Muhammad Nasir Bello 

**Supervisor**: Prof. Dr. András Hajdu  

**Institution**: University of Debrecen, Faculty of Informatics 
