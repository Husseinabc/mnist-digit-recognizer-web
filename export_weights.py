"""Export the Dense weights of mnist_model.keras to weights.json (plain JSON).

Run from this folder:
    pip install tensorflow
    python export_weights.py

It also writes test_samples.json (30 MNIST test images + the probabilities
Keras gives for them) so the JavaScript version can be checked against Keras.
"""
import json
import os

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "mnist-digit-recognizer", "mnist_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
assert len(dense) == 2, "expected 2 Dense layers, found %d" % len(dense)

W1, b1 = dense[0].get_weights()  # (784, 64), (64,)
W2, b2 = dense[1].get_weights()  # (64, 10), (10,)
print("shapes:", W1.shape, b1.shape, W2.shape, b2.shape)


def flat(a):
    # row-major, rounded to keep the file small
    return [round(float(v), 6) for v in a.reshape(-1)]


weights = {
    "W1": flat(W1), "b1": flat(b1), "W2": flat(W2), "b2": flat(b2),
    "shapes": {"W1": list(W1.shape), "b1": list(b1.shape),
               "W2": list(W2.shape), "b2": list(b2.shape)},
}
with open(os.path.join(HERE, "weights.json"), "w") as f:
    json.dump(weights, f, separators=(",", ":"))
print("wrote weights.json")

# --- samples for checking the JS implementation against Keras ---
try:
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = x_test.astype("float32") / 255.0
    probs = model.predict(x, verbose=0)
    acc = float((probs.argmax(1) == y_test).mean())
    print("Keras test accuracy: %.4f" % acc)
    n = 30
    samples = {
        "images": [x_test[i].reshape(-1).astype(int).tolist() for i in range(n)],
        "labels": [int(y_test[i]) for i in range(n)],
        "probs": [[round(float(p), 6) for p in probs[i]] for i in range(n)],
        "keras_accuracy": acc,
    }
    with open(os.path.join(HERE, "test_samples.json"), "w") as f:
        json.dump(samples, f, separators=(",", ":"))
    print("wrote test_samples.json")
except Exception as e:
    print("Could not write test_samples.json (optional):", e)
