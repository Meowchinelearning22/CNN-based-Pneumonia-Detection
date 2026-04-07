import tensorflow as tf
import matplotlib.pyplot as plt

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "chest_xray/train",
    image_size=(224,224),
    batch_size=32
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "chest_xray/test",
    image_size=(224,224),
    batch_size=32
)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)
model.save("pneumonia_model.h5")

model.evaluate(test_data)