import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

mnist = tf.keras.datasets.mnist
(training_images, training_label), (testing_images, testing_labels) = mnist.load_data()

training_images = training_images / 255.0
testing_images = testing_images / 255

model = Sequential([
    Flatten(),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["acc"])
model.fit(training_images, training_label, epochs=5)
model.evaluate(testing_images, testing_labels)

classification = model.predict(testing_images)

print(classification[0])
print(testing_labels[0])

