from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from modules import DiffusionModel, DiffusionModelForwardMode
from train import DIFFUSION_CONFIG
from torch.optim import AdamW
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch import Tensor, nn
from lightning.pytorch import loggers
import torch
import lightning as L


NUM_CLASSES = 10


class CIFAR10Gen(L.LightningModule):
    def __init__(self, val_dataset: DataLoader):
        super().__init__()
        
        self.val_loader = val_dataset
        self.diffusion = DiffusionModel(DIFFUSION_CONFIG, 3, NUM_CLASSES)


    def do_forward(self, batch: tuple[Tensor, Tensor], tag: str, mode=DiffusionModelForwardMode.EVAL):
        image, label = batch
        label = F.one_hot(label, NUM_CLASSES).float()
        t, true_eps, predict_eps, noisy_image = self.diffusion(image, label, mode)
        loss = F.mse_loss(predict_eps, true_eps, reduction="sum")

        self.log(f"{tag} loss", loss)
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        return self.do_forward(batch, "Train", DiffusionModelForwardMode.TRAIN)


    @torch.inference_mode()
    def validation_step(self, batch: tuple[Tensor, Tensor]):
        return self.do_forward(batch, "Validation")


    @torch.inference_mode()
    def test_step(self, batch: tuple[Tensor, Tensor]):
        return self.do_forward(batch, "Test")


    def on_validation_end(self) -> None:
        image, label = next(iter(self.val_loader))
        image = image.to(self.device)
        label = F.one_hot(label, NUM_CLASSES).float().to(self.device)
        label_indices = torch.arange(0, NUM_CLASSES).repeat(10, 1).transpose(0, 1).flatten()
        gen_labels = F.one_hot(label_indices, num_classes=NUM_CLASSES).float().to(self.device)

        with torch.autocast(device_type=self.device.type), torch.inference_mode():
            full_t = torch.full((image.size(0), 1), 
                           DIFFUSION_CONFIG.denoise_steps - 1, 
                           device=self.device)

            x_t, noise = self.diffusion.add_noise(image, full_t)
            denoised = self.diffusion.denoise(x_t, label)

            size = 32, 32

            generated = self.diffusion.generate(len(gen_labels), size, 3, gen_labels)

            tensorboard: SummaryWriter = self.logger.experiment # type: ignore
            tensorboard.add_image("Validation/Denoised", make_grid(to_grayscale(denoised), nrow=16), self.current_epoch)
            tensorboard.add_image("Validation/Ground truth", make_grid(to_grayscale(image), nrow=16), self.current_epoch)
            tensorboard.add_image("Validation/Noise", make_grid(to_grayscale(noise), nrow=16), self.current_epoch)

            classes = ['plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for i, images_cls in enumerate(generated.chunk(10)):
                tensorboard.add_image(f"Validation/Generated {classes[i]}", 
                                      make_grid(to_grayscale(images_cls), nrow=16), self.current_epoch)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=DIFFUSION_CONFIG.learn_rate)


class ScaleTransform(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, image: Tensor):
        image = image * 2 - 1
        return image


def to_grayscale(image: Tensor) -> Tensor:
    return (image + 1) / 2


def main():
    torch.set_float32_matmul_precision("high")
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        ScaleTransform()
    ])
    train_dataset = CIFAR10("cifar10", train=True, download=True, transform=transforms)
    train_dataset, val_dataset = random_split(train_dataset, lengths=(0.8, 0.2))
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8)

    test_dataset = CIFAR10("cifar10", train=False, download=True, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)

    tensorboard = loggers.TensorBoardLogger(".")
    gen = CIFAR10Gen(val_loader)
    trainer = L.Trainer(accelerator="auto", max_epochs=100, precision="16-mixed", logger=tensorboard)
    trainer.fit(model=gen, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=gen, dataloaders=test_loader)


if __name__ == "__main__":
    main()
