import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the DeepLabV3+ model for segmentation
segmentation_model = tf.keras.applications.DenseNet201(input_shape=(None, None, 3), weights='imagenet', include_top=False)

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((512, 512))  # Resize to input size
    img_array = np.array(img_resized) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img_resized

# Function to apply segmentation model
def apply_segmentation_model(img_array):
    segmentation_output = segmentation_model.predict(img_array)  # Get segmentation mask
    return segmentation_output

# Function to generate a mask for body parts
def get_body_part_mask(segmentation_output, original_image):
    segmentation_output = segmentation_output.squeeze()  # Remove batch dimension
    mask = np.argmax(segmentation_output, axis=-1)  # Get the predicted mask
    
    # Resize mask to match original image size (512x512)
    mask_resized = cv2.resize(mask.astype(np.uint8), (original_image.size[0], original_image.size[1]))
    
    # Apply mask to the original image
    original_image_array = np.array(original_image)
    body_part_mask = mask_resized[:, :, np.newaxis] * original_image_array  # Masked body part

    return body_part_mask, mask_resized

# Function to detect infected areas based on redness (color-based detection)
def detect_infected_area(body_part_mask):
    # Convert to HSV (Hue-Saturation-Value) to isolate red patches (redness)
    hsv_image = cv2.cvtColor(body_part_mask, cv2.COLOR_RGB2HSV)
    
    # Define the range for detecting redness in the image (adjust this as needed)
    lower_red = np.array([0, 50, 50])  # Lower bound for red hue
    upper_red = np.array([10, 255, 255])  # Upper bound for red hue
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)  # Create a red mask
    
    # Apply red mask to the original image
    infected_area = cv2.bitwise_and(body_part_mask, body_part_mask, mask=red_mask)
    
    # Calculate the infected area percentage
    total_pixels = infected_area.size
    infected_pixels = np.sum(infected_area > 0)
    infected_percentage = (infected_pixels / total_pixels) * 100
    return infected_percentage, infected_area
