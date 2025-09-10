import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import json
import os

# --- You can adjust these parameters ---
# Path to your training data
DATASET_PATH = 'dataset/train'

# Image dimensions that the model expects
IMG_SIZE = (224, 224)

# How many images to process in one go
BATCH_SIZE = 16 # Lower this to 8 if you run into memory issues

# How many times the model sees the entire dataset. 
# For a hackathon, keep this between 5 and 10.
EPOCHS = 5      

# --- 1. Prepare the Data Generators ---

# Check if the dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: The directory '{DATASET_PATH}' was not found.")
    print("Please make sure you have created the 'dataset/train' folder and copied your image class folders into it.")
else:
    # Create an ImageDataGenerator to load and augment images for better training
    # Rescaling normalizes pixel values from a 0-255 range to a 0-1 range.
    # Validation split reserves a part of the data to check the model's performance.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,      # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2, # Randomly shift images vertically
        shear_range=0.2,        # Shear transformations
        zoom_range=0.2,         # Randomly zoom in on images
        horizontal_flip=True,   # Randomly flip images horizontally
        fill_mode='nearest',
        validation_split=0.2  # Use 20% of the data for validation
    )

    # Create a generator for loading training data from the directory
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training' # Specify this is for training
    )

    # Create a generator for loading validation data from the directory
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation' # Specify this is for validation
    )

    # Get the number of classes (diseases) from the generator
    num_classes = train_generator.num_classes
    print(f"Found {num_classes} disease classes in the dataset.")

    # --- 2. Build the Model using Transfer Learning ---
    # Load the pre-trained MobileNetV2 model, without its top classification layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the base model so we don't have to retrain the whole thing
    for layer in base_model.layers:
        layer.trainable = False

    # Add our custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # A pooling layer to reduce dimensions
    x = Dense(1024, activation='relu')(x) # A large, fully-connected layer to learn complex patterns
    x = Dropout(0.5)(x) # Dropout helps prevent overfitting by randomly ignoring some neurons during training
    predictions = Dense(num_classes, activation='softmax')(x) # The final output layer for our classes

    # Combine the base model with our custom layers into a single, trainable model
    model = Model(inputs=base_model.input, outputs=predictions)

    # --- 3. Compile and Train the Model ---
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting model training... This may take a while depending on your hardware.")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )
    print("Training complete.")

    # --- 4. Save the Trained Model and Labels ---
    model.save('crop_vision_model.h5')
    print("Trained model has been saved to 'crop_vision_model.h5'")

    # Also save the class labels in the correct order for prediction
    class_indices = train_generator.class_indices
    labels = list(class_indices.keys())
    # We need to import the json library to save the labels
    import json
    with open('class_labels.json', 'w') as f:
        json.dump(labels, f)
    print("Class labels have been saved to 'class_labels.json'")
