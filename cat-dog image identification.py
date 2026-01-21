import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Cat = 3, Dog = 5
train_filter = (y_train == 3) | (y_train == 5)
test_filter = (y_test == 3) | (y_test == 5)

X_train, y_train = X_train[train_filter.flatten()], y_train[train_filter.flatten()]
X_test, y_test = X_test[test_filter.flatten()], y_test[test_filter.flatten()]
X_train = X_train / 255.0
X_test = X_test / 255.0
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    X_train, y_train == 5,
    epochs=5,
    validation_data=(X_test, y_test == 5)
)
test_loss, test_acc = model.evaluate(X_test, y_test == 5)
print("Test Accuracy:", test_acc)
print(X_train.shape)
print(y_train.shape)
model.summary()
test_loss, test_acc = model.evaluate(X_test, y_test == 5)
print("Test Accuracy:", test_acc)
import numpy as np

sample_image = X_test[0]
prediction = model.predict(sample_image.reshape(1,32,32,3))

if prediction > 0.5:
    print("Predicted: DOG 🐶")
else:
    print("Predicted: CAT 🐱")


