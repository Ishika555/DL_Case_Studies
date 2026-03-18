# Import libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Convert 2D to 1D
    keras.layers.Dense(128, activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax')  # Output layer
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_acc)

# Predict on first image
prediction = model.predict(x_test)

print("Predicted Digit:", prediction[0].argmax())
print("Actual Digit:", y_test[0])

# Show image
plt.imshow(x_test[0], cmap='gray')
plt.title("Sample Image")
plt.show()