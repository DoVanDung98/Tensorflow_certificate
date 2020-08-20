import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

model = Sequential([
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer=tf.optimizers.Adam(), loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(training_images,training_labels, epochs=5)

model.evaluate(test_images, test_labels)

# Excercise
classification = model.predict(test_images)
print(classification[0])
print(test_labels[0])
