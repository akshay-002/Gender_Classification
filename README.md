# Gender Classification using Convolutional Neural Networks

## Overview

This project focuses on implementing a gender classification system using Convolutional Neural Networks (CNNs). The aim is to develop a model capable of accurately predicting the gender of individuals based on facial images.

## Dataset

The dataset used for training and evaluation consists of images of male and female faces. The dataset is divided into training and validation sets for model training and testing purposes. It contains a balanced distribution of male and female images.

## Implementation

### Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib

### Model Architecture

The CNN model architecture used for gender classification comprises several convolutional layers followed by max-pooling and fully connected layers. Batch normalization and dropout layers are incorporated to improve the model's generalization ability and prevent overfitting.

### Data Preprocessing

Image data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping are applied to the training images using Keras' `ImageDataGenerator` to enhance the model's robustness and prevent overfitting.

### Training

The model is trained using the training dataset and evaluated on the validation dataset. Binary cross-entropy loss and Adam optimizer are utilized for model compilation. Training progress is monitored using accuracy metrics.

### Evaluation

The trained model's performance is evaluated based on accuracy metrics on both the training and validation sets. Additionally, visualization of training and validation accuracy and loss curves is provided using Matplotlib.

## Usage

To use the trained model for gender classification:

1. Ensure all necessary dependencies are installed.
2. Load the trained model weights.
3. Provide an image containing a human face as input to the model.
4. The model predicts the gender of the individual in the image.

## Example

```python
# Load the model
model = tf.keras.models.load_model('gender_classification_model.h5')

# Provide image input
path = "path/to/your/image.jpg"
img = image.load_img(path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Predict gender
classes = model.predict(x)


# Check if the predicted probability is greater than 0.5 (threshold for classification)
if classes[0] > 0.5:
    print("The image is predicted to be of a man.")
else:
    print("The image is predicted to be of a woman.")

plt.imshow(img)
