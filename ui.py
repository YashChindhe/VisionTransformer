import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

st.title("CIFAR-10 Image Classification")

# Load data
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# Model builder
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load or Train model
MODEL_PATH = "cifar10_model.h5"

if os.path.exists(MODEL_PATH):
    st.session_state.model = keras.models.load_model(MODEL_PATH)
    st.success("Model loaded from disk.")
else:
    st.session_state.model = build_model()
    with st.spinner("Training model for first-time use..."):
        history = st.session_state.model.fit(x_train, y_train, epochs=5,
                                             validation_split=0.2,
                                             batch_size=64, verbose=0)
        st.session_state.model.save(MODEL_PATH)
        st.session_state.history = history.history
        st.success("Training Complete and Model Saved!")

# Plotting
if 'history' in st.session_state:
    st.subheader("Training Accuracy and Loss")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(st.session_state.history['accuracy'], label='Train Acc')
    ax[0].plot(st.session_state.history['val_accuracy'], label='Val Acc')
    ax[0].legend()
    ax[0].set_title("Accuracy")

    ax[1].plot(st.session_state.history['loss'], label='Train Loss')
    ax[1].plot(st.session_state.history['val_loss'], label='Val Loss')
    ax[1].legend()
    ax[1].set_title("Loss")

    st.pyplot(fig)

# Test random image
if st.button("Test Random Image"):
    index = np.random.randint(0, len(x_test))
    image = x_test[index]
    label = np.argmax(y_test[index])
    pred = np.argmax(st.session_state.model.predict(np.expand_dims(image, axis=0)))

    st.image(image, caption=f"True: {label}, Predicted: {pred}")