import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate

# Directories for your images and masks
image_dir = 'ISIC/Training data'  # Replace with the actual path to your images
mask_dir = 'ISIC/Training Ground Truth'    # Replace with the actual path to your masks

# Get the list of image and mask filenames
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_filenames = [f for f in os.listdir(mask_dir) if f.endswith('_Segmentation.png')]

# Initialize lists to hold valid image-mask pairs
valid_images = []
valid_masks = []

# Loop through each image to match with its corresponding mask
for img_file in image_filenames:
    image_name = img_file.replace('.jpg', '')
    for mask_file in mask_filenames:
        mask_name = mask_file.replace('_Segmentation.png', '')
        if image_name == mask_name:
            valid_images.append(os.path.join(image_dir, img_file))
            valid_masks.append(os.path.join(mask_dir, mask_file))

# Check the number of valid image-mask pairs
print(f"Number of valid image-mask pairs: {len(valid_images)}")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(valid_images, valid_masks, test_size=0.2, random_state=42)

# Function to load and preprocess images and masks
def load_image_and_mask(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    mask = cv2.resize(mask, (128, 128))
    img = img / 255.0
    mask = mask / 255.0
    img = img_to_array(img)
    mask = np.expand_dims(mask, axis=-1)
    return img, mask

# Process all training and validation data
def process_all_data(image_paths, mask_paths):
    images = []
    masks = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img, mask = load_image_and_mask(img_path, mask_path)
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)

X_train_processed, y_train_processed = process_all_data(X_train, y_train)
X_val_processed, y_val_processed = process_all_data(X_val, y_val)

# Build the updated segmentation model
def build_model(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.2)(p1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.2)(p2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.2)(p3)

    # Bottleneck
    b1 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    b1 = Conv2D(256, (3, 3), activation='relu', padding='same')(b1)

    # Decoder
    u1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b1)
    u1 = concatenate([u1, c3])
    u1 = Dropout(0.2)(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)

    u2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = concatenate([u2, c2])
    u2 = Dropout(0.2)(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)

    u3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u3 = concatenate([u3, c1])
    u3 = Dropout(0.2)(u3)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u3)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and summarize the model
model = build_model()
model.summary()

# Train the model
model.fit(X_train_processed, y_train_processed, 
          validation_data=(X_val_processed, y_val_processed), 
          epochs=10, batch_size=16)

# Save the trained model
model.save('segmentation_model.h5')
print("Model saved successfully!")
