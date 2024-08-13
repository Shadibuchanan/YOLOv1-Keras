import tensorflow as tf
import os
import pandas as pd
import numpy as np

class VOCDataset(tf.keras.utils.Sequence):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        boxes = tf.constant(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = tf.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.numpy()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix = tf.tensor_scatter_nd_update(
                    label_matrix, 
                    [[i, j, 20]], 
                    [1]
                )

                # Box coordinates
                box_coordinates = tf.constant(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix = tf.cast(label_matrix, tf.float32)
                box_coordinates = tf.cast(box_coordinates, tf.float32)

                label_matrix = tf.tensor_scatter_nd_update(
                    label_matrix,
                    [[i, j, 21], [i, j, 22], [i, j, 23], [i, j, 24]],
                    box_coordinates
                )

                # Set one hot encoding for class_label
                label_matrix = tf.tensor_scatter_nd_update(
                    label_matrix,
                    [[i, j, class_label]],
                    [1]
                )

        return image, label_matrix