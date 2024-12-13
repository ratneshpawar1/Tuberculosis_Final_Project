# Tuberculosis Detection System Using Deep Learning

## Project Overview
A comprehensive deep learning system for detecting tuberculosis from chest X-ray images, implemented in both Keras (primary) and PyTorch (backup) frameworks.

## Installation Requirements

**Core Dependencies**
```bash
# For Keras Implementation
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn

# For PyTorch Implementation (Backup)
pip install torch torchvision
```

**System Requirements**
- Python 3.7+
- CUDA-capable GPU (recommended)
- Minimum 8GB RAM
- 10GB free disk space
- Operating System: Windows 10/11, macOS, or Linux

## Dataset Setup

1. **Download Dataset**
   - Visit Kaggle: "Tuberculosis X-ray Diagnosis CNN Classifier Dataset"
   - Download chest X-ray images
   - Create directory structure:
```
project_directory/
├── normal/
│   └── [normal chest X-ray images]
├── Tuberculosis/
│   └── [tuberculosis chest X-ray images]
├── resnetnew.py
└── pytorchimplementation.py
```

2. **Data Preprocessing**
   - Images automatically resized to 224x224 pixels
   - Pixel values normalized to [0,1] range
   - Training-validation split: 70-30
   - Data augmentation applied during training

## Implementation Details

**Keras Version (Primary)**
- VGG16 architecture with transfer learning
- Advanced data augmentation
- Early stopping and learning rate reduction
- Class weight balancing
- Grad-CAM visualization
- K-fold cross-validation

**PyTorch Version (Backup)**
- ResNet18 architecture
- Custom dataset class
- Early stopping implementation
- Learning rate scheduling
- Similar evaluation metrics

## Usage Instructions

1. **Environment Setup**
```bash
python -m venv tb_detection_env
source tb_detection_env/bin/activate  # Unix/MacOS
tb_detection_env\Scripts\activate     # Windows
```

2. **Running the Implementation**
```bash
# For Keras implementation
python keras_implementation.py

# For PyTorch implementation (backup)
python pytorch_implementation.py
```

## Output and Visualization

The system generates:
- Training/validation metrics
- Classification report
- Confusion matrix
- ROC-AUC score
- F1 score
- Training history plots
- Grad-CAM visualizations
- Misclassified samples analysis

## Performance Monitoring

- Real-time training/validation metrics
- Cross-validation results
- Model checkpointing
- Early stopping monitoring
- Learning rate adaptation
- Class balance tracking

## Error Handling

- Automatic input validation
- Dataset integrity checks
- GPU memory management
- Training stability monitoring
- Comprehensive error logging

## Additional Features

- Automated model checkpointing
- Cross-validation implementation
- Advanced data augmentation
- Class weight balancing
- Visualization tools
- Performance metrics calculation

This implementation provides a robust framework for tuberculosis detection with comprehensive evaluation tools and visualization capabilities, offering both Keras and PyTorch implementations for maximum flexibility and reliability.
