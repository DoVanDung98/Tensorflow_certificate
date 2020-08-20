import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Note: code not run becaus error output class

mnist = tf.keras.datasets.mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    Flatten(),
    Dense(64, activation="relu"),
    Dense(5, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.fit(training_images,training_labels, epochs=5)
model.evaluate(test_images,test_labels)

classification = model.predict(test_images)
print(classification[0])
print(test_labels[0])