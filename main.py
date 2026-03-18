# ============================
# IMPORTS
# ============================
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================
# DATASET
# ============================
data_dir = "dataset"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ============================
# MODEL
# ============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# build fix
model.build((None,128,128,3))

# ============================
# TRAIN
# ============================
model.fit(train_data, validation_data=val_data, epochs=5)

# ============================
# TEST IMAGE
# ============================
img_path = "dataset/no/no2.jpg"   # change if needed

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128,128))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Tumor Detected")
else:
    print("No Tumor")

# ============================
# GRAD-CAM (FINAL SAFE VERSION)
# ============================

# last conv layer
last_conv_layer = model.layers[4]

# grad model (stable version)
grad_model = Model(
    inputs=model.inputs,
    outputs=[last_conv_layer.output, model.outputs]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[0][:, 0]

grads = tape.gradient(loss, conv_outputs)

# safety fix
if grads is None:
    grads = tf.ones_like(conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

# ============================
# DISPLAY RESULT
# ============================

img_original = cv2.imread(img_path)
img_original = cv2.resize(img_original, (128,128))

heatmap = cv2.resize(heatmap.numpy(), (128,128))
heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img_original

plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
plt.title("CAM Result")
plt.axis('off')
plt.show()