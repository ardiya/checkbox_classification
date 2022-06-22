from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
import torch
import yaml
import numpy as np


from checkbox_classification.dataset import TripletDataset
from checkbox_classification.net import Net
from checkbox_classification.trainer import Trainer


def collate_fn(batch):
    """Custom collate so we can use image with different resolution"""
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.LongTensor(labels)
    return [data, labels]


def save_best_model(cfg, net):
    save_dir = Path(cfg["save"]["dir"])
    if not save_dir.exists():
        save_dir.mkdir(exist_ok=True)
    save_path = save_dir / cfg["save"]["filename"]
    print("Saving to", str(save_path))
    torch.save(net.state_dict(), str(save_path))


def save_tensorboard_embedding(cfg, net, device, writer, global_step):
    infer_transforms = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    vis_transforms = A.Compose(
        [
            A.Resize(50, 50),
            A.Normalize(mean=(0, 0, 0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ]
    )
    dset = TripletDataset(
        Path(cfg["train"]["images_path"])
    )
    net.eval()
    with torch.no_grad():
        embeddings = list()
        labels = list()
        label_imgs = list()
        for image, label in dset:
            embedding, _ = net(infer_transforms(image=image)["image"].unsqueeze(0).to(device))
            label_img = vis_transforms(image=image)["image"].unsqueeze(0)
            embeddings.append(embedding)
            labels.append(dset.classes[label])
            label_imgs.append(label_img)
        embeddings = torch.cat(embeddings, dim=0)
        labels = np.array(labels)
        label_imgs = torch.cat(label_imgs, dim=0)
    writer.add_embedding(embeddings, metadata=labels, label_img=label_imgs, global_step=global_step)


def prepare_train_loader(cfg):
    train_transforms = A.Compose(
        [
            A.Perspective(keep_size=True, fit_output=True, p=0.25),
            A.CoarseDropout(min_height=2, max_height=16, min_width=2, max_width=16, p=0.1),
            A.HorizontalFlip(p=0.05),
            A.VerticalFlip(p=0.05),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    traindataset = TripletDataset(
        Path(cfg["train"]["images_path"]), transform=train_transforms
    )
    trainloader = DataLoader(
        dataset=traindataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return trainloader


def evaluate_validation(cfg, net, device):
    infer_transforms = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    valdataset = TripletDataset(
        Path(cfg["val"]["images_path"]), transform=infer_transforms
    )
    valloader = DataLoader(
        dataset=valdataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valloader:
            labels = labels.to(device)
            for img_id, image in enumerate(images):
                _, out = net(image.unsqueeze(0).to(device))
                predicted = torch.argmax(out.data[0])
                total += 1
                correct += 1 if predicted == labels[img_id] else 0

    val_accuracy = correct / total

    return val_accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser("train checkbock classification")
    parser.add_argument("yaml_config_path", type=str, help="path to the config file")

    args = parser.parse_args()

    with open(args.yaml_config_path) as fp:
        cfg = yaml.safe_load(fp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = prepare_train_loader(cfg)

    torch.autograd.set_detect_anomaly(True)
    net = Net(n_classes=cfg["n_classes"]).to(device)
    net.apply(Net.initialize_weights)

    trainer = Trainer(cfg, net)

    writer = SummaryWriter("logs/" + cfg["experiment_name"])

    best_accuracy = 0.0
    for epoch in range(cfg["train"]["epoch"]):
        # Train & log
        net.train()
        for batch, (images, labels) in enumerate(train_loader):
            # move to device
            labels = labels.to(device)
            for i in range(len(images)):
                images[i] = images[i].to(device)

            cls_loss, triplet_loss, n_triplets = trainer.fit(images, labels)
            global_step = epoch * len(train_loader) + batch
            writer.add_scalar("classification_loss", cls_loss, global_step=global_step)
            writer.add_scalar("triplet_loss", triplet_loss, global_step=global_step)
            writer.add_scalar(
                "total_loss", triplet_loss + cls_loss, global_step=global_step
            )
            if batch % 10 == 0:
                print(
                    f"Epoch-{epoch} iter-{batch}: Classification-loss = {cls_loss:.3f}, Triplet-loss = {triplet_loss:.3f},  Number of mined triplets = {n_triplets}"
                )

        # Eval validation and log
        val_accuracy = evaluate_validation(cfg, net, device)
        print("Val accuracy images: %.2f%%" % (val_accuracy * 100))
        global_step = epoch * len(train_loader)
        writer.add_scalar("val_accuracy", val_accuracy, global_step=global_step)

        # Save model if it beats best accuracy
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            save_best_model(cfg, net)
            save_tensorboard_embedding(cfg, net, device, writer, global_step)

    print("Training done with best validation accuracy", best_accuracy)
    writer.close()


if __name__ == "__main__":
    main()
