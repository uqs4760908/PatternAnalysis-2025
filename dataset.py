from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import ToDtype
from torchvision.io.image import decode_image
from pathlib import Path
from PIL import Image
import os
import torch

from config import ImageInfo


ALZHEIMER_DISEASE = "AD"
COGNITIVE_NORMAL = "NC"
TRANSFORM = ToDtype(dtype=torch.float16, scale=True)


class ImageListDataset(Dataset[torch.Tensor]):
    def __init__(self, 
                 images: list[Path]):
        super().__init__()
        self.images = images
        with Image.open(images[0]) as image:
            self.size = image.size
            self.depth = len(image.getbands())


    def __len__(self) -> int:
        return len(self.images)


    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.images[index]
        image = decode_image(str(path))
        return TRANSFORM(image)


def make_dataloader(dataset: ImageListDataset, batch_size: int) -> DataLoader[torch.Tensor]:
    num_workers = os.cpu_count() or 0
    return DataLoader(dataset, 
                      num_workers=num_workers, 
                      shuffle=True,
                      batch_size=batch_size)


class ANDIDataset:
    def __init__(self, 
                 dataset_root: Path,
                 batch_size: int,
                 num_validation: int = 2000):
        dataset_root = dataset_root / "AD_NC"
        train_ad = dataset_root / "train" / ALZHEIMER_DISEASE
        train_cn = dataset_root / "train" / COGNITIVE_NORMAL
        test_ad = dataset_root / "test" / ALZHEIMER_DISEASE
        test_cn = dataset_root / "test" / COGNITIVE_NORMAL

        train_ad_images = [*train_ad.iterdir()]
        train_cn_images = [*train_cn.iterdir()]
        test_ad_images = [*test_ad.iterdir()]
        test_cn_images = [*test_cn.iterdir()]

        self.train_ad_dataset = ImageListDataset(train_ad_images[:-num_validation])
        self.train_ad_loader = make_dataloader(self.train_ad_dataset, batch_size)

        self.validation_ad_dataset = ImageListDataset(train_ad_images[-num_validation:])
        self.validation_ad_loader = make_dataloader(self.validation_ad_dataset, batch_size)
        
        self.train_cn_dataset = ImageListDataset(train_cn_images[:-num_validation])
        self.train_cn_loader = make_dataloader(self.train_cn_dataset, batch_size)

        self.validation_cn_dataset = ImageListDataset(train_cn_images[-num_validation:])
        self.validation_cn_loader = make_dataloader(self.validation_cn_dataset, batch_size)

        self.test_ad_dataset = ImageListDataset(test_ad_images)
        self.test_ad_loader = make_dataloader(self.test_ad_dataset, batch_size)

        self.test_cn_dataset = ImageListDataset(test_cn_images)
        self.test_cn_loader = make_dataloader(self.test_cn_dataset, batch_size)

    @property
    def image_info(self) -> ImageInfo:
        return ImageInfo(self.train_ad_dataset.size, self.train_ad_dataset.depth)
