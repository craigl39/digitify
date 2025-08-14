import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# CONFIG
IMG_WIDTH, IMG_HEIGHT = 160, 50
BATCH_SIZE = 32
EPOCHS = 20
CHARS = list("0123456789.")
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}  # 0 reserved for blank

AUTOTUNE = tf.data.AUTOTUNE


def load_data(csv_path="data/labels.csv", data_dir="data"):
    print(f"[INFO] Loading dataset from '{csv_path}' and images from '{data_dir}'...")
    df = pd.read_csv(csv_path)
    df["filepath"] = df["filename"].apply(lambda x: os.path.join(data_dir, x))
    print(f"[INFO] Loaded {len(df)} samples.")
    return df


def encode_label_tf(label):
    chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(CHARS),
            values=tf.constant(list(range(len(CHARS)))),  # 0-10
        ),
        default_value=NUM_CLASSES - 1,  # blank if not found
    )
    return table.lookup(chars)


def validate_label(label_tensor):
    min_label = tf.reduce_min(label_tensor)
    max_label = tf.reduce_max(label_tensor)

    valid_min = tf.reduce_all(label_tensor >= 0)
    valid_max = tf.reduce_all(label_tensor < NUM_CLASSES - 1)  # must be < 11

    tf.debugging.assert_equal(
        valid_min,
        True,
        message="Label contains negative index.",
    )
    tf.debugging.assert_equal(
        valid_max,
        True,
        message=f"Label contains index >= blank ({NUM_CLASSES - 1}).",
    )

    return label_tensor


def process_row(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    label_encoded = encode_label_tf(label)
    label_encoded = validate_label(label_encoded)

    input_len = tf.constant(IMG_WIDTH // 4, dtype=tf.int32)  # scalar
    label_len = tf.shape(label_encoded)[0]  # scalar

    return (
        {
            "image": image,
            "label": label_encoded,
            "input_len": input_len,
            "label_len": label_len,
        },
        tf.zeros(()),  # dummy target for Keras fit API
    )


def make_dataset(df, batch_size=BATCH_SIZE):
    paths = df["filepath"].astype(str).values
    labels = df["label"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(process_row, num_parallel_calls=AUTOTUNE)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            {
                "image": [IMG_HEIGHT, IMG_WIDTH, 1],
                "label": [None],
                "input_len": (),  # scalar shape
                "label_len": (),  # scalar shape
            },
            (),
        ),
        padding_values=(
            {
                "image": 0.0,
                "label": 0,
                "input_len": 0,
                "label_len": 0,
            },
            0.0,
        ),
    )
    ds = ds.prefetch(AUTOTUNE)

    return ds


class CTCLossLayer(layers.Layer):
    def call(self, inputs):
        y_pred, labels, input_len, label_len = inputs

        loss = keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)
        self.add_loss(tf.reduce_mean(loss))
        tf.print("[DEBUG] Current batch CTC loss:", tf.reduce_mean(loss))
        return y_pred


def build_model_with_ctc():
    print("[INFO] Building CRNN model with CTC loss layer...")
    # Inputs
    image = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    labels = keras.Input(shape=(None,), dtype="int32", name="label")
    input_len = keras.Input(shape=(1,), dtype="int32", name="input_len")
    label_len = keras.Input(shape=(1,), dtype="int32", name="label_len")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(image)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    def reshape_for_rnn(t):
        shape = tf.shape(t)
        batch_size, h, w, c = shape[0], shape[1], shape[2], shape[3]
        return tf.reshape(t, [batch_size, w, h * c])

    x = layers.Lambda(reshape_for_rnn)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    y_pred = layers.Dense(NUM_CLASSES, activation="softmax")(
        x
    )  # +1 for CTC blank token

    loss_out = CTCLossLayer()([y_pred, labels, input_len, label_len])

    model = keras.Model(
        inputs=[image, labels, input_len, label_len], outputs=loss_out, name="ocr_model"
    )
    model.compile(optimizer="adam")

    print("[INFO] Model compiled.")
    return model


def plot_training_history(history, filename="training_loss.png"):
    print(f"[INFO] Plotting training loss to '{filename}'...")
    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(filename)
    plt.close()
    print("[INFO] Training loss plot saved.")


def save_inference_model(model, folder="model"):
    print(f"[INFO] Saving inference model to '{folder}'...")
    image_input = model.get_layer(name="image").input
    y_pred = model.get_layer(index=-2).output  # Dense output before loss layer
    inference_model = keras.Model(inputs=image_input, outputs=y_pred)
    inference_model.save(folder)
    print("[INFO] Inference model saved.")


def main():
    print("[START] OCR training script started.")
    df = load_data()
    train_ds = make_dataset(df)

    print("[INFO] Dataset created with batch size:", BATCH_SIZE)
    print("[INFO] Number of classes:", NUM_CLASSES)

    model = build_model_with_ctc()
    model.summary()

    print("[INFO] Starting training...")
    history = model.fit(train_ds, epochs=EPOCHS)
    print("[INFO] Training complete.")

    plot_training_history(history)
    save_inference_model(model)
    print("[END] OCR training script finished.")


if __name__ == "__main__":
    main()
