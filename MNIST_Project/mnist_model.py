# Import libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize data (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 🔥 CNN needs 3D input (height, width, channels)
# MNIST is grayscale → 1 channel
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = keras.Sequential([
    
    # Convolution layer (feature extraction)
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    
    # Pooling layer (downsampling)
    keras.layers.MaxPooling2D((2,2)),
    
    # Second convolution layer
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    
    # Second pooling layer
    keras.layers.MaxPooling2D((2,2)),
    
    # Convert 3D → 1D
    keras.layers.Flatten(),
    
    # Fully connected layer
    keras.layers.Dense(128, activation='relu'),
    
    # Output layer
    keras.layers.Dense(10, activation='softmax')
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

# Predict
prediction = model.predict(x_test)

print("Predicted Digit:", prediction[0].argmax())
print("Actual Digit:", y_test[0])

# Show image
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.title("Sample Image")
plt.show()