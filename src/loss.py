# %load ./src/loss.py
"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn

from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_class = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants for loss weighting
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Identify where objects and no objects are in target
        obj = target[..., 0] == 1  # Object exists
        noobj = target[..., 0] == 0  # No object

        # -------------------------
        # No Object Loss
        # -------------------------
        no_object_loss = self.bce(
            predictions[..., 0:1][noobj], target[..., 0:1][noobj],
        )

        # -------------------------
        # Object Loss
        # -------------------------
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat(
            [
                self.sigmoid(predictions[..., 1:3]),
                torch.exp(predictions[..., 3:5]) * anchors,
            ],
            dim=-1,
        )
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce(
            self.sigmoid(predictions[..., 0:1][obj]),
            ious * target[..., 0:1][obj],
        )

        # -------------------------
        # Box Coordinates Loss
        # -------------------------
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x, y coordinates
        target[..., 3:5] = torch.log(
            1e-16 + target[..., 3:5] / anchors
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # -------------------------
        # Class Loss
        # -------------------------
        class_predictions = predictions[..., 5][obj]
        class_targets = torch.ones_like(class_predictions)  # Since only one class
        class_loss = self.bce_class(class_predictions, class_targets)

        # Total loss
        loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

        return loss, (box_loss, object_loss, no_object_loss, class_loss)
