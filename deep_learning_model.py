# CODTECH Internship – Task 2
# Deep Learning Project: Image Classification Using TensorFlow (MNIST Dataset)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 1️ Load Dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2️ Build the Model
print("Building Neural Network model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3️ Train Model
print(" Training the model...")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 4️ Evaluate
print(" Evaluating the model on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"\n Test Accuracy: {test_acc * 100:.2f}%")

# 5️ Visualizations

# Plot Accuracy
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot Loss
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 6️ Predict sample images
predictions = model.predict(x_test)

# Show 6 sample results
plt.figure(figsize=(8,4))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()

print(" Deep Learning Task Completed Successfully!")
