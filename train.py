import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

# Set random seed for reproducibility
tf.random.set_seed(123)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "GPU" if tf.test.is_gpu_available() else "CPU"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.h5"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

# Define data augmentation
def transform(image, bboxes):
    image = tf.image.resize(image, (448, 448))
    image = image / 255.0
    return image, bboxes

# Training function
@tf.function
def train_step(model, x, y, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_fn(train_dataset, model, optimizer, loss_fn):
    loop = tqdm(train_dataset, leave=True)
    mean_loss = []

    for x, y in loop:
        loss = train_step(model, x, y, loss_fn, optimizer)
        mean_loss.append(loss)

        # Update progress bar
        loop.set_postfix(loss=loss.numpy())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(LOAD_MODEL_FILE, model, optimizer)

    train_dataset = VOCDataset(
        "data/8examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = tf.data.Dataset.from_generator(
        lambda: train_dataset,
        output_signature=(
            tf.TensorSpec(shape=(448, 448, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(7, 7, 30), dtype=tf.float32)
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_loader = tf.data.Dataset.from_generator(
        lambda: test_dataset,
        output_signature=(
            tf.TensorSpec(shape=(448, 448, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(7, 7, 30), dtype=tf.float32)
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.01
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)

        if mean_avg_prec > 0.9:
            save_checkpoint(model, optimizer, filename=LOAD_MODEL_FILE)
            break

if __name__ == "__main__":
    main()