# Dog vs Cat Classification with a Simple Perceptron

This project corresponds to **MT3006 - Laboratory 4**, where a simple perceptron is implemented in PyTorch to classify images of **dogs (0)** and **cats (1)** from grayscale 32x32 images.

---

## Repository Contents

- `cat_data.mat` → Dataset containing only cat images.  
- `dog_data.mat` → Dataset containing only dog images.  
- `dog_cat_data.mat` → Combined dataset of dogs and cats (used for training).  
- `laboratorio4.py` → Main Python script for training and evaluating the model.  

---

## Requirements

The project requires **Python 3.9+** (although it should work with nearby versions).  

Main dependencies:

- [PyTorch](https://pytorch.org/)  
- numpy  
- matplotlib  
- scikit-learn  
- scipy  
- scikit-image  

Install all dependencies with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn scipy scikit-image
