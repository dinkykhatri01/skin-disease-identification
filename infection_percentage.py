import cv2
import numpy as np
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops

# Function to predict infection area percentage
def predict_infection(image, segmentation_model):
    # Step 1: Preprocess the image for prediction
    # Resize or preprocess the image if needed
    image_resized = cv2.resize(image, (224, 224))  # Adjust as per your model's input size
    image_input = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    image_input = image_input / 255.0  # Normalize if needed (depending on your model)

    # Step 2: Run segmentation model to get the mask
    # Assuming `segmentation_model.predict(image_input)` gives a binary mask where 1 represents infection
    mask = segmentation_model.predict(image_input)[0]  # [0] because of batch dimension, adjust if necessary

    # Step 3: Apply post-processing to refine the mask
    cleaned_mask = post_process_mask(mask)

    # Step 4: Calculate infection percentage
    infection_percentage = calculate_infection_percentage(cleaned_mask)
    
    return cleaned_mask, infection_percentage

# Post-process the segmentation mask to refine it
def post_process_mask(mask):
    binary_mask = mask > 0.5  # Convert to binary (adjust threshold as necessary)
    cleaned_mask = closing(opening(binary_mask, disk(3)), disk(3))  # Apply opening and closing to remove noise
    return cleaned_mask

# Calculate the infection area percentage
def calculate_infection_percentage(mask):
    # Label the connected components in the mask
    labeled_mask = label(mask)
    
    # Get the total area of the image and the area of the infected regions
    total_area = mask.size  # Total number of pixels in the image
    infected_area = np.sum(mask)  # Number of pixels that are part of the infection
    
    # Calculate the infection percentage
    infection_percentage = (infected_area / total_area) * 100  # Percentage of infected area
    return infection_percentage
