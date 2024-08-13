import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(
    predicted_boxes: tf.constant, 
    ground_boxes: tf.constant,
    bbox_format = "midpoint",
    ):
    if bbox_format == "midpoint":
        predicted_x_min = predicted_boxes[..., 0:1] - predicted_boxes[..., 2:3] / 2
        predicted_y_min = predicted_boxes[..., 1:2] - predicted_boxes[..., 3:4] / 2
        predicted_x_max = predicted_boxes[..., 0:1] + predicted_boxes[..., 2:3] / 2
        predicted_y_max = predicted_boxes[..., 1:2] + predicted_boxes[..., 3:4] / 2
        ground_x_min = ground_boxes[..., 0:1] - ground_boxes[..., 2:3] / 2
        ground_y_min = ground_boxes[..., 1:2] - ground_boxes[..., 3:4] / 2
        ground_x_max = ground_boxes[..., 0:1] + ground_boxes[..., 2:3] / 2
        ground_x_max = ground_boxes[..., 1:2] + ground_boxes[..., 3:4] / 2
    elif bbox_format == "corners":
        predicted_x_min = predicted_boxes[..., 0:1]
        predicted_y_min = predicted_boxes[..., 1:2]
        predicted_x_max = predicted_boxes[..., 2:3]
        predicted_y_max = predicted_boxes[..., 3:4]
        ground_x_min = ground_boxes[..., 0:1]
        ground_y_min = ground_boxes[..., 1:2]
        ground_x_max = ground_boxes[..., 2:3]
        ground_y_max = ground_boxes[..., 3:4]

    predicted_height = predicted_y_max - predicted_y_min
    predicted_width = predicted_x_max - predicted_x_min 
    predicted_area = tf.abs(predicted_height * predicted_width)

    ground_height = ground_y_max - ground_y_min
    ground_width = ground_x_max - ground_x_min 
    ground_area = tf.abs(ground_height * ground_width)

    intersection_x_min = tf.maximum(predicted_x_min, ground_x_min)
    intersection_y_min = tf.maximum(predicted_y_min, ground_y_min)
    intersection_x_max = tf.minimum(predicted_x_max, ground_x_max)
    intersection_y_max = tf.minimum(predicted_y_max, ground_y_max)

    intersection_width = tf.maximum(intersection_x_max - intersection_x_min, 0)
    intersection_height = tf.maximum(intersection_y_max - intersection_y_min, 0)

    intersection = intersection_height * intersection_width
    union = predicted_area + ground_area - intersection + tf.constant(1e6)
    jaccard_index = intersection / union
    return jaccard_index

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert isinstance(bboxes, list)

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                tf.constant(chosen_box[2:]),
                tf.constant(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def tf_trapz(y, x):
    """
    Approximate the integral of f(x) from a to b using the trapezoidal rule.
    """
    dx = x[1:] - x[:-1]
    return tf.reduce_sum((y[:-1] + y[1:]) / 2.0 * dx)

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        print(f"Class {c}: {len(detections)} detections, {len(ground_truths)} ground truths")

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:tf.zeros(3), 1:tf.zeros(5)}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = tf.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = tf.zeros((len(detections)))
        FP = tf.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    tf.constant(detection[3:]),
                    tf.constant(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP = tf.tensor_scatter_nd_update(TP, [[detection_idx]], [1])
                    amount_bboxes[detection[0]] = tf.tensor_scatter_nd_update(
                        amount_bboxes[detection[0]], [[best_gt_idx]], [1])
                else:
                    FP = tf.tensor_scatter_nd_update(FP, [[detection_idx]], [1])

            # if IOU is lower then the detection is a false positive
            else:
                FP = tf.tensor_scatter_nd_update(FP, [[detection_idx]], [1])

        TP_cumsum = tf.cumsum(TP, axis=0)
        FP_cumsum = tf.cumsum(FP, axis=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        
        # Convert to float32 and use tf.ones instead of tf.constant([1])
        precisions = tf.cast(precisions, dtype=tf.float32)
        recalls = tf.cast(recalls, dtype=tf.float32)
        precisions = tf.concat([tf.ones(1, dtype=tf.float32), precisions], axis=0)
        recalls = tf.concat([tf.zeros(1, dtype=tf.float32), recalls], axis=0)
        
        # tf.math.trapz for numerical integration
        average_precisions.append(tf_trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = image.numpy()
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", box_format="midpoint", device="GPU"):
    all_pred_boxes = []
    all_true_boxes = []
    
    for batch_idx, (x, labels) in enumerate(loader):
        x = x
        labels = labels

        with tf.GradientTape() as tape:
            predictions = model(x, training=False)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([batch_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([batch_idx] + box)

    print("Number of predicted boxes:", len(all_pred_boxes))
    print("Number of true boxes:", len(all_true_boxes))
    return all_pred_boxes, all_true_boxes
def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios.
    """
    predictions = tf.convert_to_tensor(predictions)
    batch_size = tf.shape(predictions)[0]
    predictions = tf.reshape(predictions, (batch_size, 7, 7, 30))
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = tf.concat(
        [tf.expand_dims(predictions[..., 20], -1), tf.expand_dims(predictions[..., 25], -1)],
        axis=-1
    )
    best_box = tf.cast(tf.argmax(scores, axis=-1), tf.float32)
    best_box = tf.expand_dims(best_box, -1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    
    cell_indices = tf.tile(tf.reshape(tf.range(7, dtype=tf.float32), [1, 7, 1]), [batch_size, 1, 7])
    cell_indices = tf.expand_dims(cell_indices, -1)
    
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + tf.transpose(cell_indices, [0, 2, 1, 3]))
    w_y = 1 / S * best_boxes[..., 2:4]
    
    converted_bboxes = tf.concat([x, y, w_y], axis=-1)
    
    predicted_class = tf.cast(tf.argmax(predictions[..., :20], axis=-1), tf.float32)
    predicted_class = tf.expand_dims(predicted_class, -1)
    
    best_confidence = tf.reduce_max(predictions[..., 20:22], axis=-1, keepdims=True)
    
    converted_preds = tf.concat([predicted_class, best_confidence, converted_bboxes], axis=-1)

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out)
    converted_pred = tf.reshape(converted_pred, [tf.shape(out)[0], S * S, -1])
    converted_pred = tf.cast(converted_pred, tf.float32)
    
    return [
        [x.numpy() for x in example]
        for example in converted_pred
    ]

def save_checkpoint(model, optimizer, filename="my_checkpoint"):
    print("=> Saving checkpoint")
    model.save_weights(filename)
    # Note: TensorFlow doesn't have a direct equivalent to saving optimizer state

def load_checkpoint(checkpoint_file, model, optimizer):
    print("=> Loading checkpoint")
    model.load_weights(checkpoint_file)
    # Note: TensorFlow doesn't have a direct equivalent to loading optimizer state