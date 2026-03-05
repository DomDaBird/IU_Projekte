# Technical Architecture Overview

## Goal

The system detects human facial emotions from images and classifies them
into five categories:

-   Angry
-   Fear
-   Happy
-   Sad
-   Surprise

## High-level workflow

1.  Dataset is stored in a directory-based structure (train/val/test)
2.  Input pipeline loads images using tf.data
3.  EfficientNetB0 (ImageNet weights) is used for transfer learning
4.  A custom classification head predicts one of the five emotions
5.  The trained model is evaluated on a test set
6.  A Streamlit app provides a simple user interface for inference

## Model architecture

Backbone: - EfficientNetB0 (pretrained on ImageNet)

Classification head: - GlobalAveragePooling2D - Dense layer - Dropout -
Softmax output layer (5 classes)

## Repository entry points

-   train.py -- training pipeline
-   evaluate.py -- evaluation pipeline
-   infer_single.py -- single image inference
-   app_streamlit.py -- Streamlit demo
-   data.py -- dataset pipeline
-   model.py -- model definition
-   config.py -- configuration
