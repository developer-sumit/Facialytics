import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, LeakyReLU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__)) 

# Enhanced Data Augmentation
def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        os.path.join(current_dir, "dataset/train"),
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training",
    )

    val_generator = train_datagen.flow_from_directory(
        os.path.join(current_dir, "dataset/train"),
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation",
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(current_dir, "dataset/test"),
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical",
    )

    return train_generator, val_generator, test_generator


# Define Model Architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", input_shape=(48, 48, 1)),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2)),

        GlobalAveragePooling2D(),
        Dense(256),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        Dense(128),
        BatchNormalization(),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),

        Dense(7, activation="softmax"),
    ])
    return model


# Compile and Train Model
def train_model(model, train_generator, val_generator):
    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.96, staircase=True
    )

    # Optimizer with LearningRateSchedule
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("best_emotion_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger("training_log.csv", append=True),
    ]

    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks,
    )
    return history


# Visualize Training Performance
def plot_performance(history):
    plt.figure(figsize=(14, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main Execution
if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    model = create_model()
    history = train_model(model, train_gen, val_gen)

    # Save the final model in H5 format
    model.save("final_emotion_model.h5")

    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Visualize training results
    plot_performance(history)
