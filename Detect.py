import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage import io

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.applications import VGG19


# Define a custom loss function for Vgg19 UNet model

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

import numpy as np
import pandas as pd
import cv2
from skimage import io
import matplotlib.pyplot as plt
import os

def visualize_prediction(original_image, predicted_mask):
    """Function to visualize the original image, predicted mask, and combined image."""
    
    # Assuming original_image is already an array (not a path)
    # No need to load the original image again
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Ensure it's in RGB format
    
    # Create a combined image
    combined_image = original_image.copy()
    combined_image[predicted_mask.squeeze() == 1] = (255, 0, 0)  # Color for predicted mask
    
    # Plotting
    plt.figure(figsize=(15, 10))

    # Display original image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')
    
    # Display predicted mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask.squeeze(), cmap='jet', alpha=0.5)  # Squeeze to remove extra dimensions
    plt.axis('off')

    # Display combined image
    plt.subplot(1, 3, 3)
    plt.title("Combined Image")
    plt.imshow(combined_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
# Example usage:
# visualize_prediction(original_image, predicted_mask, 'path_to_save_original.png', 'path_to_save_mask.png')


def predict_single_image(image_path, model_seg):
    # Create an empty array of shape 1, 256, 256, 3
    X = np.empty((1, 256, 256, 3))
    
    # Read the image
    img = io.imread(image_path)
    
    # Resize the image
    img = cv2.resize(img, (256, 256))
    
    # Convert to array of type float64
    img = np.array(img, dtype=np.float64)
    
    # Standardizing the image
    img -= img.mean()
    img /= img.std()
    
    # Reshape the image to (1, 256, 256, 3)
    X[0,] = img
    
    # Make prediction of mask
    predict = model_seg.predict(X)
    
    # If the sum of the predicted mask is 0, there's no tumor
    if predict.round().astype(int).sum() == 0:
        has_mask = 0
        mask = 'No mask :)'
    else:
        has_mask = 1
        mask = predict

    return {'image_path': image_path, 'predicted_mask': mask, 'has_mask': has_mask}

# Load the model
model = load_model(".\\models\\Brain_seg.h5",
                   custom_objects={"focal_tversky": focal_tversky, "tversky": tversky, "tversky_loss": tversky_loss})

# Specify the image path
#image_path = "D:/Sem5/Projects/Brain_tumor/Brain_tumor_seg2/Brain_tumor_segmentation_Vgg19UNet-20241021T112515Z-001/Brain_tumor_segmentation_Vgg19UNet/Dataset/archive/kaggle_3m/TCGA_HT_7881_19981015/TCGA_HT_7881_19981015_30.tif"
image_path = ".\\Dataset\\kaggle_3m\\TCGA_CS_6669_20020102\\TCGA_CS_6669_20020102_15.tif"
# Predict the mask
result = predict_single_image(image_path, model)

# Check if there's a predicted mask and visualize
if result['has_mask'] == 1:
    original_image = io.imread(image_path)  # Load the original image for visualization
    visualize_prediction(original_image, result['predicted_mask'])
else:
    # original_image = io.imread(image_path)  # Load the original image for visualization
    # visualize_prediction(original_image, result['predicted_mask'])
    print(result)

