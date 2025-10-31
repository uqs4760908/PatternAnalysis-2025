from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToDtype, Compose, RandomHorizontalFlip, Transform
from torchvision.io.image import decode_image
from pathlib import Path
from PIL import Image
import torch

from config import ImageInfo


ALZHEIMER_DISEASE = "AD"
COGNITIVE_NORMAL = "NC"
TRAIN_TRANSFORM = Compose([
    ToDtype(dtype=torch.float32, scale=True),
    RandomHorizontalFlip()
])
TEST_TRANSFORM = ToDtype(dtype=torch.float32, scale=True)


AD_LABEL = 0
NC_LABEL = 1
NUM_CLASS = 2
Item = tuple[torch.Tensor, int]


class ImageListDataset(Dataset[Item]):
    def __init__(self, 
                 ad_images: list[Path],
                 cn_images: list[Path],
                 transform: Transform):
        super().__init__()
        self.ad_images = ad_images
        self.cn_images = cn_images
        self.transform = transform
        with Image.open(ad_images[0]) as image:
            # PIL specifies size in (width, height). We want (height, width)
            self.size = image.size[1], image.size[0]
            self.depth = len(image.getbands())


    def __len__(self) -> int:
        return len(self.ad_images) + len(self.cn_images)


    def __getitem__(self, index: int) -> Item:
        if index < len(self.ad_images):
            images = self.ad_images
            label = AD_LABEL
        else:
            images = self.cn_images
            label = NC_LABEL
            index -= len(self.ad_images)

        path = images[index]
        image = decode_image(str(path))
        return self.transform(image), label


class ANDIDataset:
    def __init__(self, 
                 dataset_root: Path,
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

        self.train_dataset= ImageListDataset(
                train_ad_images[:-num_validation],
                train_cn_images[:-num_validation],
                TRAIN_TRANSFORM)

        self.validation_dataset = ImageListDataset(
                train_ad_images[-num_validation:],
                train_cn_images[-num_validation:],
                TEST_TRANSFORM)
        
        self.test_dataset = ImageListDataset(
                test_ad_images,
                test_cn_images,
                TEST_TRANSFORM)

    @property
    def image_info(self) -> ImageInfo:
        return ImageInfo(self.train_dataset.size, self.train_dataset.depth)
