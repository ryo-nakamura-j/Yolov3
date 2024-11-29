
# %load ./src/train.py
"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""
import csv
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
# Paths to specific files
DATASET_DIR = os.path.join(PROJECT_ROOT, "OD-WeaponDetection/Pistol detection/")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

checkpoint_file = os.path.join(CHECKPOINTS_DIR, config.LOAD_MODEL_FILE)


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    box_losses = []
    object_losses = []
    no_object_losses = []
    class_losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            # Get loss and individual components
            loss0, components0 = loss_fn(out[0], y0, scaled_anchors[0])
            loss1, components1 = loss_fn(out[1], y1, scaled_anchors[1])
            loss2, components2 = loss_fn(out[2], y2, scaled_anchors[2])

            loss = loss0 + loss1 + loss2
            # Sum individual components
            total_box_loss = components0[0] + components1[0] + components2[0]
            total_object_loss = components0[1] + components1[1] + components2[1]
            total_no_object_loss = components0[2] + components1[2] + components2[2]
            total_class_loss = components0[3] + components1[3] + components2[3]

        losses.append(loss.item())
        box_losses.append(total_box_loss.item())
        object_losses.append(total_object_loss.item())
        no_object_losses.append(total_no_object_loss.item())
        class_losses.append(total_class_loss.item())
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
        
    mean_loss = sum(losses) / len(losses)
    mean_box_loss = sum(box_losses) / len(box_losses)
    mean_object_loss = sum(object_losses) / len(object_losses)
    mean_no_object_loss = sum(no_object_losses) / len(no_object_losses)
    mean_class_loss = sum(class_losses) / len(class_losses)

    return mean_loss, mean_box_loss, mean_object_loss, mean_no_object_loss, mean_class_loss

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
    csv_file=os.path.join(DATASET_DIR,'OD-dataset.csv'),
    img_dir=os.path.join(DATASET_DIR,'Weapons'),
    label_dir=os.path.join(DATASET_DIR,'labels'),
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
    
    all_losses = []
    all_box_losses = []
    all_object_losses = []
    all_no_object_losses = []
    all_class_losses = []
    
    # Check if the CSV file exists and is empty
    loss_file = 'epoch_losses.csv'
    write_header = False
    if not os.path.exists(loss_file) or os.stat(loss_file).st_size == 0:
        write_header = True

    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        mean_loss, mean_box_loss, mean_object_loss, mean_no_object_loss, mean_class_loss = train_fn(
            train_loader, model, optimizer, loss_fn, scaler, scaled_anchors
        )
        
        # Store epoch losses
        all_losses.append(mean_loss)
        all_box_losses.append(mean_box_loss)
        all_object_losses.append(mean_object_loss)
        all_no_object_losses.append(mean_no_object_loss)
        all_class_losses.append(mean_class_loss)

        # Save losses to CSV
        with open('epoch_losses.csv', 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'total_loss', 'box_loss', 'object_loss', 'no_object_loss', 'class_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if write_header:
                writer.writeheader()
                write_header = False  # Reset the flag after writing the header

            writer.writerow({
                'epoch': epoch + 1,
                'total_loss': mean_loss,
                'box_loss': mean_box_loss,
                'object_loss': mean_object_loss,
                'no_object_loss': mean_no_object_loss,
                'class_loss': mean_class_loss,
            })

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        if epoch > 0 and epoch % 2 == 0:
            if config.SAVE_MODEL:
              save_checkpoint(model, optimizer, filename=os.path.join(CHECKPOINTS_DIR, f"checkpoint_C1_OD_wd_epoch_3{epoch}_{current_date}.pth.tar"))
              
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
