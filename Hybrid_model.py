import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20
CLASS_NAMES = [
    'Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick',
    'Fenugreek', 'Guava', 'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit',
    'Jamaica_Cherry-Gasagase', 'Jamun', 'Jasmine', 'Karanda', 'Lemon', 'Mango',
    'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal', 'Pomegranate',
    'Rasna', 'Rose_apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi'
]

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/Identification_plant/dataset",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Split dataset
total_batches = tf.data.experimental.cardinality(dataset).numpy()
train_batches = int(0.7 * total_batches)
val_batches = int(0.2 * total_batches)
test_batches = total_batches - (train_batches + val_batches)

train_dataset = dataset.take(train_batches)
val_dataset = dataset.skip(train_batches).take(val_batches)
test_dataset = dataset.skip(train_batches + val_batches).take(test_batches)

# Data augmentation and preprocessing
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0 / 255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])

# Define model
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    
    # CNN layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten and reshape for RNN
    layers.Reshape(target_shape=(IMAGE_SIZE // 8 * IMAGE_SIZE // 8, 64)),
    
    # RNN layer
    layers.SimpleRNN(128, return_sequences=False),
    
    # Dense layers for classification
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset
)

# Evaluate the model
for images_batch, labels_batch in test_dataset.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("First image to predict")
    plt.imshow(first_image)
    plt.show()

    print("First image's actual label: ", CLASS_NAMES[first_label])

    batch_prediction = model.predict(images_batch)
    predicted_probabilities = batch_prediction[0]
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    print("Predicted class probabilities: ", predicted_probabilities)
    print("Predicted class: ", predicted_class_name)

# Save the model
import os
model_directory = "D:/Identification_plant/models"
model_version = 1
model_filename = f"model_version_{model_version}.keras"
model_path = os.path.join(model_directory, model_filename)
os.makedirs(model_directory, exist_ok=True)
model.save(model_path)
print(f"Model saved to {model_path}")
