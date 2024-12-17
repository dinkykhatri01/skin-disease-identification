# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import build_cnn_model

# Constants
batch_size = 32
epochs = 45
img_height = 128
img_width = 128

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    'Images/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the model
model = build_cnn_model(input_shape=(img_height, img_width, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=epochs)

# Save the model weights
model.save('cnn_model.h5')
print("Model training complete and saved as cnn_model.h5.")


