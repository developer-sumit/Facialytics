import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir='image', target_size=(48, 48), batch_size=32):
    """Loads and preprocesses data from directory."""
    
    # Get the absolute paths for train, validation, and test data
    train_dir = 'D:/face/images/train'
    test_dir = 'D:/face/images/test'

    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Directory not found: {test_dir}")
    
    # Initialize ImageDataGenerator with rescaling and validation split
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    # Train generator
    train_generator = datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
    )

    # Validation generator
    val_generator = datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
    )

    # Test generator
    test_generator = datagen.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    # Return the generators
    return train_generator, val_generator, test_generator
