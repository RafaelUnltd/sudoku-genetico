import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical

# Load data from CSV
df = pd.read_csv('neural_test_data/sudoku_m.csv')  # Replace 'your_dataset.csv' with your actual file path

# Convert strings to NumPy arrays
def string_to_numpy_array(sudoku_string):
    return np.array([int(digit) for digit in sudoku_string])

df['input'] = df['quizzes'].apply(string_to_numpy_array)
df['output'] = df['solutions'].apply(string_to_numpy_array)

# Encode input and output as integers (0 to 9)
df['input_encoded'] = df['input'].apply(lambda x: to_categorical(x, num_classes=10).flatten())
df['output_encoded'] = df['output']# Encode input

# Split data into training and testing sets
train_size = int(0.8 * len(df))
train_data, test_data = df[:train_size], df[train_size:]

checkpoint_path = "models/sudoku.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# Create and compile the model
model = Sequential()
model.add(Dense(units=128, input_dim=81 * 10, activation='relu'))
model.add(Dense(units=81 * 10, activation='relu'))
model.add(Dense(units=81 * 10, activation='softmax'))
model.add(Reshape((81, 10)))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    np.stack(train_data['input_encoded'].to_numpy()),  # Convert Pandas Series to NumPy array
    np.stack(train_data['output_encoded'].to_numpy()),  # Convert Pandas Series to NumPy array
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    callbacks=[cp_callback]
)

# Evaluate the model on the test set
results = model.evaluate(
    np.stack(test_data['input_encoded'].to_numpy()),  # Convert Pandas Series to NumPy array
    np.stack(test_data['output_encoded'].to_numpy()),  # Convert Pandas Series to NumPy array
)

print("Test Loss:", results[0])
print("Test Accuracy:", results[1])

input_sudoku = '000500007064208005730010000000746002980005710020800603002009400500100260619030000'
input_array = string_to_numpy_array(input_sudoku)
input_encoded = to_categorical(input_array, num_classes=10).flatten()
input_encoded = np.expand_dims(input_encoded, axis=0)

a = model.predict(input_encoded)
a = np.argmax(a, axis=2).flatten()
a = a.reshape((9, 9))
print(a)
