import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing import image


loaded_model =  tf.keras.models.load_model('/Users/akshay/projects/Gender_classification/gender_classification_model.h5')
test_dir = '/Users/akshay/projects/Gender_classification/Dataset/Test'

# Paths to male and female subdirectories
male_dir = os.path.join(test_dir, "male")
female_dir = os.path.join(test_dir, "female")

# Initialize variables for male and female accuracies
male_correct_predictions = 0
female_correct_predictions = 0
male_total_predictions = 0
female_total_predictions = 0

# Process images in the male subdirectory
male_image_filenames = os.listdir(male_dir)
for filename in male_image_filenames:
    try:
        img_path = os.path.join(male_dir, filename)
        img = image.load_img(img_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        classes = loaded_model.predict(x)
        predicted_label = 1 if classes[0] > 0.5 else 0
        true_label = 1  # Male
        male_total_predictions += 1
        if predicted_label == true_label:
            male_correct_predictions += 1
    except Exception as e:
        print(f"Error processing male image {filename}: {e}")

# Process images in the female subdirectory
female_image_filenames = os.listdir(female_dir)
for filename in female_image_filenames:
    try:
        img_path = os.path.join(female_dir, filename)
        img = image.load_img(img_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        classes = loaded_model.predict(x)
        predicted_label = 0  # Female
        true_label = 0
        female_total_predictions += 1
        if predicted_label == true_label:
            female_correct_predictions += 1
    except Exception as e:
        print(f"Error processing female image {filename}: {e}")

# Calculate accuracies for male and female images
male_accuracy = male_correct_predictions / male_total_predictions if male_total_predictions != 0 else 0
female_accuracy = female_correct_predictions / female_total_predictions if female_total_predictions != 0 else 0

# Calculate total accuracy
total_correct_predictions = male_correct_predictions + female_correct_predictions
total_predictions = male_total_predictions + female_total_predictions
total_accuracy = total_correct_predictions / total_predictions if total_predictions != 0 else 0

print("Accuracy on male images:", male_accuracy)
print("Accuracy on female images:", female_accuracy)
print("Total accuracy:", total_accuracy)
