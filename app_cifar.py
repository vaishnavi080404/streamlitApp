# app_cifar.py
# A simple Streamlit app to demonstrate a CNN on CIFAR-10 (built-in Keras dataset).
# It shows: training logs, history plots, test evaluation, and lets you select random test images to predict.

import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Simple CNN Demo (CIFAR-10)", layout="wide")
st.title("🧠 Simple CNN Demo (CIFAR-10)")
st.write(
    "This app trains a small CNN on **CIFAR-10** (32×32 color images, 10 classes) and lets you test predictions."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Training Settings")
epochs = st.sidebar.slider("Epochs", 1, 15, 5)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], index=2)

use_small_subset = st.sidebar.checkbox("Use small subset (faster demo)", value=True)
subset_train = st.sidebar.slider("Train subset size", 2000, 50000, 10000, 1000, disabled=not use_small_subset)
subset_test = st.sidebar.slider("Test subset size", 500, 10000, 2000, 500, disabled=not use_small_subset)

dropout = st.sidebar.slider("Dropout", 0.0, 0.6, 0.25, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 999999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.header("Prediction Settings")
num_random_images = st.sidebar.slider("How many random test images to show", 6, 20, 12, 1)

# -----------------------------
# Class names for CIFAR-10
# -----------------------------
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# -----------------------------
# Cache dataset load
# -----------------------------
@st.cache_data(show_spinner=False)
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return (x_train, y_train), (x_test, y_test)

# -----------------------------
# Build model
# -----------------------------
def build_cnn(lr: float, dropout_rate: float):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),  # 32->16

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),  # 16->8

        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -----------------------------
# Load + preprocess data
# -----------------------------
(x_train, y_train), (x_test, y_test) = load_cifar10()

# Normalize to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

rng = np.random.default_rng(int(seed))

if use_small_subset:
    train_idx = rng.choice(len(x_train), size=int(subset_train), replace=False)
    test_idx = rng.choice(len(x_test), size=int(subset_test), replace=False)
    x_train_s, y_train_s = x_train[train_idx], y_train[train_idx]
    x_test_s, y_test_s = x_test[test_idx], y_test[test_idx]
else:
    x_train_s, y_train_s = x_train, y_train
    x_test_s, y_test_s = x_test, y_test

st.subheader("1) Data preview")
c1, c2 = st.columns(2)
with c1:
    st.write(f"Train images: **{len(x_train_s)}**")
    st.write(f"Test images: **{len(x_test_s)}**")
    st.write(f"Image shape: `{x_train_s[0].shape}` (height, width, channels)")
with c2:
    i = int(rng.integers(0, len(x_train_s)))
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(x_train_s[i])
    plt.title(f"Label: {CLASS_NAMES[int(y_train_s[i])]}")
    plt.axis("off")
    st.pyplot(fig)

# -----------------------------
# Session state
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = None
if "train_logs" not in st.session_state:
    st.session_state.train_logs = ""

class StreamlitLogCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        st.session_state.train_logs = ""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"Epoch {epoch+1}: "
            f"loss={logs.get('loss', np.nan):.4f}, "
            f"acc={logs.get('accuracy', np.nan):.4f}, "
            f"val_loss={logs.get('val_loss', np.nan):.4f}, "
            f"val_acc={logs.get('val_accuracy', np.nan):.4f}\n"
        )
        st.session_state.train_logs += msg

# -----------------------------
# Train
# -----------------------------
st.subheader("2) Train the CNN")

train_col, info_col = st.columns([1, 2])

with train_col:
    if st.button("🚀 Train / Retrain Model", use_container_width=True):
        with st.spinner("Training..."):
            tf.keras.utils.set_random_seed(int(seed))
            model = build_cnn(float(learning_rate), float(dropout))

            history = model.fit(
                x_train_s, y_train_s,
                validation_split=0.2,
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=0,
                callbacks=[StreamlitLogCallback()],
            )

            st.session_state.model = model
            st.session_state.history = history.history

with info_col:
    st.write(
        "- CIFAR-10 is **harder than MNIST** because images are colorful and more complex.\n"
        "- A CNN learns **filters** that detect edges/textures/shapes.\n"
        "- The last layer outputs **10 probabilities** (one per class)."
    )

if st.session_state.model is not None:
    with st.expander("Show model summary"):
        lines = []
        st.session_state.model.summary(print_fn=lambda x: lines.append(x))
        st.code("\n".join(lines))

# -----------------------------
# Logs + plots + evaluation
# -----------------------------
if st.session_state.model is not None and st.session_state.history is not None:
    st.subheader("3) Logs, history, and evaluation")

    log_left, plot_right = st.columns([1, 2])

    with log_left:
        st.write("📋 Training logs")
        st.text_area("Logs", st.session_state.train_logs, height=240)

        test_loss, test_acc = st.session_state.model.evaluate(x_test_s, y_test_s, verbose=0)
        st.metric("Test accuracy", f"{test_acc:.3f}")
        st.metric("Test loss", f"{test_loss:.3f}")

    with plot_right:
        hist = st.session_state.history

        fig1 = plt.figure()
        plt.plot(hist["loss"], label="train loss")
        plt.plot(hist["val_loss"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.plot(hist["accuracy"], label="train acc")
        plt.plot(hist["val_accuracy"], label="val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        st.pyplot(fig2)

# -----------------------------
# Prediction section
# -----------------------------
st.subheader("4) Predict on random test images")

if st.session_state.model is None:
    st.info("Train the model first to enable predictions.")
    st.stop()

if "candidate_indices" not in st.session_state or st.button("🔁 Refresh random images"):
    st.session_state.candidate_indices = rng.choice(len(x_test_s), size=int(num_random_images), replace=False).tolist()

candidate_indices = st.session_state.candidate_indices

choice = st.selectbox(
    "Choose an image from the random set",
    options=list(range(len(candidate_indices))),
    format_func=lambda i: f"Option {i+1} (test row #{candidate_indices[i]})"
)

idx = candidate_indices[int(choice)]
img = x_test_s[idx]
true_label = int(y_test_s[idx])

probs = st.session_state.model.predict(img[None, ...], verbose=0)[0]
pred_label = int(np.argmax(probs))

cA, cB = st.columns([1, 1])

with cA:
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {CLASS_NAMES[true_label]}")
    st.pyplot(fig)

with cB:
    st.write("🤖 Prediction")
    st.success(f"Predicted: **{CLASS_NAMES[pred_label]}**")
    st.write("Top probabilities:")
    top = np.argsort(probs)[::-1][:5]
    top_table = pd.DataFrame({
        "class": [CLASS_NAMES[i] for i in top],
        "probability": [float(probs[i]) for i in top]
    })
    st.dataframe(top_table, use_container_width=True)

    figp = plt.figure(figsize=(10, 3))
    plt.bar(CLASS_NAMES, probs)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    st.pyplot(figp)

st.caption("Tip: Increase epochs or use more training data for better accuracy.")
