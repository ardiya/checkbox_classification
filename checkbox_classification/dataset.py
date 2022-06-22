from pathlib import Path
import cv2
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, images_path: Path, transform=None):
        assert images_path.exists()
        self.images_path = images_path
        self.classes = [x.name for x in sorted(images_path.iterdir()) if x.is_dir()]
        self.data = list()
        for class_idx, class_name in enumerate(self.classes):
            for img_path in (self.images_path / class_name).glob("*.png"):
                self.data.append((class_idx, img_path))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_idx, image_path = self.data[idx]
        image = cv2.cvtColor(cv2.imread(str(image_path), -1), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, class_idx


if __name__ == "__main__":
    # Test data loader
    import cv2
    import torch
    import numpy as np
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    images_path = Path("/home/ardiya/Workspace/checkbox_classification/data/train")
    train_transforms = A.Compose(
        [
            A.Perspective(keep_size=True, fit_output=True, p=0.1),
            A.CoarseDropout(min_height=2, max_height=16, min_width=2, max_width=16, p=0.2),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    dset = TripletDataset(images_path, transform=train_transforms)
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True, num_workers=2)
    for img, class_idx in loader:
        img = img[0]
        class_idx = class_idx[0]

        print("mean", torch.mean(img, dim=(1,2)))
        print("min", torch.amin(img, dim=(1,2)))
        print("max", torch.amax(img, dim=(1,2)))

        img = cv2.cvtColor(np.array(img/4+0.5).transpose((1,2,0)), cv2.COLOR_RGB2BGR)
        print("img", type(img), img.shape, img.dtype)
        cv2.imshow("image", img)
        print(dset.classes[class_idx])
        cv2.waitKey(0)
