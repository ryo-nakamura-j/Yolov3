"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import sys
import os

sys.path.append(os.path.abspath("./configs"))
import config

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import YOLOv3
from tqdm import tqdm
from dataset import YOLODataset

from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

# Get the root directory of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Paths to specific files
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

checkpoint_file = os.path.join(CHECKPOINTS_DIR, config.LOAD_MODEL_FILE)


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    current_date = datetime.now().strftime("%Y-%m-%d")
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # train_loader, test_loader, train_eval_loader = get_loaders(
    #     train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/valid.csv"
    # )
    dataset = YOLODataset(
    csv_file='/content/drive/MyDrive/Yolo/OD-weapon-detection/Pistol detection/OD-dataset.csv',
    img_dir='/content/drive/MyDrive/Yolo/OD-weapon-detection/Pistol detection/Weapons',
    label_dir='/content/drive/MyDrive/Yolo/OD-weapon-detection/Pistol detection/labels',
    anchors=config.ANCHORS,
    image_size=416,
    S=[13, 26, 52],
    C=1,  # Since you have only one class
    transform=config.train_transforms,
    )
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    if config.LOAD_MODEL:
        load_checkpoint(
            checkpoint_file, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)



        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 1 == 0:
            if config.SAVE_MODEL:
              save_checkpoint(model, optimizer, filename=os.path.join(CHECKPOINTS_DIR, f"checkpoint_C1_OD_wd_epoch_{epoch}_{current_date}.pth.tar"))
              
            # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            # pred_boxes, true_boxes = get_evaluation_bboxes(
            #     test_loader,
            #     model,
            #     iou_threshold=config.NMS_IOU_THRESH,
            #     anchors=config.ANCHORS,
            #     threshold=config.CONF_THRESHOLD,
            # )
            # mapval = mean_average_precision(
            #     pred_boxes,
            #     true_boxes,
            #     iou_threshold=config.MAP_IOU_THRESH,
            #     box_format="midpoint",
            #     num_classes=config.NUM_CLASSES,
            # )
            # print(f"MAP: {mapval.item()}")
            model.train()


if __name__ == "__main__":
    main()