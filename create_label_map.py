import numpy as np
import os

# Define the list of class names
CLASS_NAMES = [
    'Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Drumstick',
    'Fenugreek', 'Guava', 'Hibiscus', 'Indian_Beech', 'Indian_Mustard', 'Jackfruit',
    'Jamaica_Cherry-Gasagase', 'Jamun', 'Jasmine', 'Karanda', 'Lemon', 'Mango',
    'Mexican_Mint', 'Mint', 'Neem', 'Oleander', 'Parijata', 'Peepal', 'Pomegranate',
    'Rasna', 'Rose_apple', 'Roxburgh_fig', 'Sandalwood', 'Tulsi'
]

# Create a dictionary mapping class indices to class names
label_map = {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}

# Define the path to the models directory
models_dir = os.path.join(os.getcwd(), 'models')

# Ensure the models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the dictionary as a .npy file in the models directory
np.save(os.path.join(models_dir, 'label_map.npy'), label_map)

print("label_map.npy created successfully in the models directory.")
