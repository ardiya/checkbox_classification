from albumentations.pytorch import ToTensorV2
from checkbox_classification.dataset import TripletDataset
from checkbox_classification.net import Net
from pathlib import Path
from time import time

import albumentations as A
import argparse
import cv2
import torch
import yaml


def get_args():
    parser = argparse.ArgumentParser("train checkbock classification")
    parser.add_argument(
        "-c",
        "--yaml-config-path",
        default="config/default.yaml",
        type=str,
        help="path to the config file",
    )
    parser.add_argument(
        "--warm-up-gpu",
        default=False,
        type=bool,
        help="""If set to true, the program will run inference for several time.
Is useful to benchmark inference time because first several run is setting up network etc.""",
    )
    parser.add_argument("img", type=str, help="path to the image to infer")

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if not Path(args.img).exists():
        print("Unable to find image file", args.img)
        import sys

        sys.exit(1)

    with open(args.yaml_config_path) as fp:
        cfg = yaml.safe_load(fp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load network
    net = Net()
    net.load_state_dict(torch.load(Path(cfg["save"]["dir"]) / cfg["save"]["filename"], map_location=device))
    net = net.to(device)
    net.eval()

    # get list of classes
    dset = TripletDataset(Path(cfg["train"]["images_path"]))
    classes = dset.classes

    # inference
    image = cv2.cvtColor(cv2.imread(args.img, -1), cv2.COLOR_BGR2RGB)
    input_tensor = A.Compose([A.Normalize(), ToTensorV2()])(image=image)["image"]

    if args.warm_up_gpu:
        for _ in range(10):
            net(input_tensor.unsqueeze(0).to(device))
    start_time = time()
    _, output_tensor = net(input_tensor.unsqueeze(0).to(device))
    ellapsed_time = time() - start_time
    probabilities = torch.nn.functional.softmax(output_tensor, dim=1)
    prediction = int(torch.argmax(probabilities, dim=1)[0])

    print(
        "predicting '{}' with confidence {:.2f}%. finished in {:.5f} seconds for image with resolution {}x{}".format(
            classes[prediction],
            float(probabilities[0][prediction]) * 100.0,
            ellapsed_time,
            image.shape[1],
            image.shape[0]
        )
    )


if __name__ == "__main__":
    main()
