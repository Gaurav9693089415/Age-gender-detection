

# 🌟 Gender and Age Detection in Python with OpenCV
This project implements a deep learning-based system for predicting age and gender from facial images. Built using PyTorch, OpenCV, and Gradio, it supports both web-based and command-line interfaces. The project also integrates DVC for dataset and model version control.

## Overview

The system is trained on real-world datasets annotated with age and gender labels. It uses a convolutional neural network (CNN) architecture with dual heads:

* A regression head for age prediction  
* A classification head for gender prediction

Face detection is performed using OpenCV’s Haarcascade classifier.

## Features

* Predicts age (as a numerical value) and gender (male/female)  
* Real-time face detection using Haarcascade  
* Preprocessing including resizing and normalization  
* Gradio-based web interface for interactive use  
* Jupyter notebooks for data preprocessing and experimentation  
* Version control for data and models using DVC  
## Project Structure


FACEAGE-AND-GENDER-DETECTION/
* pycache/ — Cached bytecode files
* .dvc/ — DVC configuration and metadata
* notebooks/ — Jupyter notebooks for development
* Face and Age Detection.ipynb
* realdatapreprocess.ipynb
* templates/ — HTML template (optional use)
  * index.html
* app.py — Web interface using Gradio
* haarcascade_frontalface_default.xml — Haarcascade for face detection
* requirements.txt — Required packages
* data.dvc — DVC-tracked data reference
* models.dvc — DVC-tracked model reference
* .gitignore — Git ignore rules
* .dvcignore — DVC ignore rules


## Model Architecture

- **Input:** RGB facial image of shape (224, 224, 3)
- **Feature Extractor:** Convolutional and MaxPooling layers
- **Outputs:**
  - Age: Linear activation for regression
  - Gender: Softmax or Sigmoid activation for classification

### Loss Functions and Metrics

- **Age Prediction:** Mean Squared Error (MSE), Mean Absolute Error (MAE)
- **Gender Classification:** Binary Cross Entropy, Accuracy

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
````

### requirements.txt

```
torch==2.0.0
opencv-python==4.5.5
dlib==19.22.0
Pillow==8.4.0
numpy==1.21.4
matplotlib==3.5.0
gradio==3.0.0
```

## Usage

### Run Web Interface

Start the Gradio web application:

```bash
python app.py
```

After starting, Gradio will provide a local URL where the interface can be accessed.

### Training and Notebooks

Use the Jupyter notebooks in the `notebooks/` directory for data preprocessing, training experiments, and evaluation.

### Prediction

Use the web app (`app.py`) to upload a facial image and receive predicted age and gender. Output includes:

* Detected face bounding box
* Predicted age
* Predicted gender label

## Dataset

The model was trained using real-world datasets with facial age and gender labels, such as:

* [UTKFace Dataset](https://susanqq.github.io/UTKFace/)

## Results

The model achieved the following performance on the test dataset:

- **Gender Classification Accuracy:** 87.63%
- **Age Prediction Mean Absolute Error (MAE):**  7 years
* Predictions include visual annotations with face, age, and gender

## Version Control

[DVC (Data Version Control)](https://dvc.org/) is used to manage large data files and trained model versions. This enables reproducibility and model tracking across development cycles.

## Acknowledgements

* UTKFace Dataset for labeled face images
* OpenCV for deep learning and image processing
* Gradio for building the interactive interface
* DVC for data and model version control

```


```
