from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from modules import DiffusionModel, DiffusionSampler
from train import DIFFUSION_CONFIG
from torch.optim import AdamW
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch import Tensor, nn
from lightning.pytorch import loggers
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import random
import torch
import lightning as L


NUM_CLASSES = 10


class CIFAR10Gen(L.LightningModule):
    def __init__(self, val_dataset: DataLoader):
        super().__init__()
        
        self.val_loader = val_dataset
        self.model = DiffusionModel(DIFFUSION_CONFIG, 3, NUM_CLASSES)
        self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(0.99))
        self.sampler = DiffusionSampler(DIFFUSION_CONFIG)


    def do_forward(self, batch: tuple[Tensor, Tensor], tag: str, train: bool):
        image, label = batch

        if not train or random.random() < 0.9:
            label = F.one_hot(label, NUM_CLASSES).float()
        else:
            label = None

        if train:
            t = torch.randint(0, DIFFUSION_CONFIG.denoise_steps, 
                              size=(image.size(0),), device=self.device, dtype=torch.int32)
        else:
            t = torch.full(image.shape[:1], DIFFUSION_CONFIG.denoise_steps - 1,
                           device=self.device, dtype=torch.int32)

        x_t, noise = self.sampler.add_noise(image, t)
        predict_eps = self.model(x_t, t, label)
        loss = F.mse_loss(predict_eps, noise, reduction="sum")

        self.log(f"{tag} loss", loss)
        return loss


    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        return self.do_forward(batch, "Train", True)


    @torch.inference_mode()
    def validation_step(self, batch: tuple[Tensor, Tensor]):
        return self.do_forward(batch, "Validation", False)


    @torch.inference_mode()
    def test_step(self, batch: tuple[Tensor, Tensor]):
        return self.do_forward(batch, "Test", False)


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

            x_t, noise = self.sampler.add_noise(image, full_t)
            denoised = self.sampler.denoise(x_t, label, self.ema_model)

            size = 32, 32

            generated = self.sampler.generate(len(gen_labels), size, 3, gen_labels, self.ema_model)

            tensorboard: SummaryWriter = self.logger.experiment # type: ignore
            tensorboard.add_image("Validation/Denoised", make_grid(to_rgb(denoised), nrow=16), self.current_epoch)
            tensorboard.add_image("Validation/Ground truth", make_grid(to_rgb(image), nrow=16), self.current_epoch)
            tensorboard.add_image("Validation/Noise", make_grid(to_rgb(noise), nrow=16), self.current_epoch)

            classes = ['plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for i, images_cls in enumerate(generated.chunk(10)):
                tensorboard.add_image(f"Validation/Generated {classes[i]}", 
                                      make_grid(to_rgb(images_cls), nrow=16), self.current_epoch)


    def on_train_epoch_end(self) -> None:
        self.ema_model.update_parameters(self.model)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=DIFFUSION_CONFIG.learn_rate)



def to_rgb(image: Tensor) -> Tensor:
    return (image + 1) / 2


def main():
    torch.set_float32_matmul_precision("high")
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize([0.5], [0.5])
    ])
    train_dataset = CIFAR10("cifar10", train=True, download=True, transform=transforms)
    train_dataset, val_dataset = random_split(train_dataset, lengths=(0.8, 0.2))
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8)

    test_dataset = CIFAR10("cifar10", train=False, download=True, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=8)

    tensorboard = loggers.TensorBoardLogger(".")
    gen = CIFAR10Gen(val_loader)
    trainer = L.Trainer(accelerator="auto", max_epochs=500, precision="16-mixed", logger=tensorboard)
    trainer.fit(model=gen, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=gen, dataloaders=test_loader)


if __name__ == "__main__":
    main()
