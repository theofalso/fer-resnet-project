# Facial Expression Recognition (FER) 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/theofalso/fer-resnet)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#)

An end-to-end Machine Learning pipeline for Facial Expression Recognition, built with PyTorch and deployed via Gradio. This project utilizes Transfer Learning on a ResNet18 architecture to classify human faces into 7 distinct emotions: *Angry, Disgust, Fear, Happy, Neutral, Sad,* and *Surprise*.

## Live Demo

You can test the model directly using your webcam or by uploading an image here:  
**[Try the App on Hugging Face Spaces](https://huggingface.co/spaces/theofalso/fer-resnet)**

## Technical Overview

- **Dataset:** FER-2013 (Processed with PyTorch DataLoaders & Data Augmentation).
- **Model:** ResNet18 (Pre-trained weights from ImageNet, with a customized fully-connected layer for 7 classes).
- **Training:** Custom PyTorch training loop optimizing Cross-Entropy Loss with Adam.
- **Inference & GUI:** Built with Gradio for seamless local and cloud deployment.