# Lung Disease Prediction Using Deep Learning

A deep learning project that predicts pneumonia from chest X-ray images using transfer learning with VGG16 architecture.

## 🎯 Project Overview

This project implements a binary classification system to detect pneumonia in chest X-ray images. The model leverages transfer learning using a pre-trained VGG16 network to achieve high accuracy in medical image analysis.

## 📊 Dataset

**Source**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

**Dataset Structure**:
- **Training Set**: 5,216 images
- **Test Set**: 624 images
- **Classes**: 2 (NORMAL, PNEUMONIA)
- **Image Format**: JPEG
- **Target Size**: 224x224 pixels

## 🏗️ Model Architecture

### Transfer Learning Approach
- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Input Shape**: (224, 224, 3)
- **Frozen Layers**: All VGG16 layers (14,714,688 parameters)
- **Custom Layers**:
  - Flatten layer
  - Dense layer with softmax activation (50,178 trainable parameters)

### Model Summary
```
Total Parameters: 14,764,866
Trainable Parameters: 50,178
Non-trainable Parameters: 14,714,688
```

## 🔧 Technical Implementation

### Prerequisites
```python
tensorflow>=2.0.0
keras
numpy
matplotlib
pillow
glob
```

### Key Components

1. **Data Preprocessing**
   - Image rescaling (1./255)
   - Data augmentation (shear, zoom, horizontal flip)
   - VGG16 preprocessing pipeline

2. **Model Configuration**
   - Loss Function: Categorical Crossentropy
   - Optimizer: Adam
   - Metrics: Accuracy

3. **Training Parameters**
   - Epochs: 5
   - Batch Size: 32
   - Steps per Epoch: 163
   - Validation Steps: 20

## 📈 Results

### Training Performance
| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|---------------|-------------------|-----------------|-------------------|
| 1     | 0.2333        | 91.12%           | 0.2477          | 90.38%           |
| 2     | 0.1143        | 95.90%           | 0.4380          | 87.82%           |
| 3     | 0.0981        | 96.18%           | 0.2839          | 91.51%           |
| 4     | 0.0820        | 96.66%           | 0.2549          | 90.38%           |
| 5     | 0.0758        | 97.18%           | 0.4023          | 88.30%           |

### Model Performance Analysis
- **Final Training Accuracy**: 97.18%
- **Final Validation Accuracy**: 88.30%
- **Status**: Shows signs of overfitting (increasing validation loss)

## 🚀 Usage

### 1. Setup Environment
```python
# Import required libraries
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Load and Prepare Data
```python
# Set image dimensions
IMAGE_SIZE = [224, 224]

# Setup data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
```

### 3. Build Model
```python
# Load VGG16 base model
vgg = VGG16(input_shape=IMAGE_SIZE + [3], 
            weights='imagenet', 
            include_top=False)

# Freeze base model layers
for layer in vgg.layers:
    layer.trainable = False

# Add custom classifier
x = Flatten()(vgg.output)
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
```

### 4. Train Model
```python
# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
```

### 5. Make Predictions
```python
# Load saved model
model = load_model('model_vgg16.h5')

# Predict on new image
img = image.load_img(image_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
classes = model.predict(img_data)

# Interpret results
if classes[0][0] > classes[0][1]:
    print('X-Ray image is NORMAL')
else:
    print('X-Ray image shows PNEUMONIA')
```
--
