from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from config import DiffusionConfig, VAEConfig
from dataset import AD_LABEL, ANDIDataset, ImageListDataset
from pathlib import Path
from modules import VAE, DiffusionModel, DiffusionModelForwardMode
from torch import multiprocessing as mp
from torch import distributed as dist
from torch import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from tempfile import NamedTemporaryFile
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from dataclasses import dataclass
import functools
import dataclasses
import logging
import colorlog
import time
import typing
import abc
import contextlib
import torch
import os


# From https://github.com/CompVis/latent-diffusion/blob/main/models/ldm/celeba256/config.yaml
# and https://github.com/Stability-AI/stablediffusion/blob/main/configs/stable-diffusion/v2-inference-v.yaml
VAE_CONFIG = VAEConfig(
    learn_rate=1.0e-06,
    num_channels=(
        1 * 128,
        2 * 128,
        4 * 128,
        4 * 128
    ),
    latent_dim=4,
    num_resnet_blocks=2,
    num_attention_heads=1,
    layer_norm_num_groups=32
)

DIFFUSION_CONFIG = DiffusionConfig(
    learn_rate=1.0e-06,
    num_channels=(
        1 * 128,
        2 * 128,
        4 * 128,
        4 * 128
    ),
    num_resnet_blocks=2,
    num_attention_heads=1,
    layer_norm_num_groups=32,

    noise_start=0.00085,
    noise_end=0.0120,
    denoise_steps=1000
)


TRAIN_STATUS_KEY = "train_status"
TRAIN_STATUS_TRAINING = "training"
TRAIN_STATUS_DONE = "done"
MODEL_PARAMS_KEY = "params"


def get_accelerator():
    return torch.accelerator.current_accelerator() or torch.get_default_device()


class PortableGradScaler:
    def __init__(self):
        # GradScaler is only availbale for cpu and cuda
        accelerator = get_accelerator()
        if accelerator.type in ("cuda", "cpu"):
            self.scaler = GradScaler(accelerator.type)
        else:
            self.scaler = None


    def scale(self, loss: Tensor):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()


    def update(self, optimiser: Optimizer):
        if self.scaler is not None:
            self.scaler.step(optimiser)
            self.scaler.update()
            optimiser.zero_grad(set_to_none=True)


def pprint_stats(stats: typing.Any) -> list[str]:
    lines: list[str] = []

    for field in dataclasses.fields(stats):
        value = getattr(stats, field.name)
        if isinstance(value, Tensor) and value.numel() == 1:
            value = value.item()

        lines.append(f"{field.name}={value}")

    return lines


@dataclass(frozen=True)
class DistributedParams:
    rank: int


RunModelFn = typing.Callable[[nn.Module, Tensor], typing.Any]


@dataclass(frozen=True)
class RunStats:
    loss: Tensor
    generated_images: Tensor


@dataclass(frozen=True)
class EvalStats:
    loss: Tensor
    input_images: Tensor
    generated_images: Tensor

    
class ModelController(abc.ABC):
    """
    Controls how a model should be executed
    """
    @abc.abstractmethod
    def model(self) -> nn.Module: ...


    @abc.abstractmethod
    def num_epochs(self) -> int: ...

    
    @abc.abstractmethod
    def train_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> RunStats: ...


    @abc.abstractmethod
    def eval_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> RunStats: ...

    
    @abc.abstractmethod
    def save_path(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class RunModelParams:
    model: nn.Module
    device: torch.device
    dataset: ImageListDataset
    controller: ModelController
    dist_params: typing.Optional[DistributedParams]


    def is_master(self) -> bool:
        return self.dist_params is None or self.dist_params.rank == 0


RunModelLoop = typing.Callable[[RunModelParams], typing.Any]


class ModelRunner:
    """
    Abstracts most heavy lifting for running train/eval loop such as progress reporting,
    distributed training, snapshoting and more
    """
    def __init__(self):
        self.dataset = ANDIDataset(Path("./data"))

        log = Path("train.log")
        if log.exists():
            os.truncate(log, 0)

        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter("[%(log_color)s%(levelname)s:%(name)s%(reset)s]: %(message)s"))
        file_handler = logging.FileHandler(log)
        file_handler.setFormatter(logging.Formatter("[%(levelname)s:%(name)s]: %(message)s"))
        self.logger = colorlog.getLogger(type(self).__name__)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)


    @contextlib.contextmanager
    def setup(self, rank: int, world_size: int, file: str):
        device = torch.accelerator.current_accelerator()
        torch.set_float32_matmul_precision("high")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        if device is None:
            return

        store = dist.FileStore(file, world_size) # type:ignore
        try:
            backend = dist.get_default_backend_for_device(device)
            dist.init_process_group(backend, store=store, rank=rank, world_size=world_size)
            self.logger.info(f"Worker {os.getpid()}: rank={rank} backend={backend}")
            yield None
        finally:
            dist.destroy_process_group()


    @staticmethod
    def sync(device: torch.device):
        match device.type:
            case "cuda":
                torch.cuda.synchronize(device)
            case "mps":
                torch.mps.synchronize()
        
    
    @functools.cache
    def batch_size(self, controller: ModelController, model: nn.Module, device: torch.device) -> int:
        image = self.dataset.train_dataset[0][0]

        batch_size = 1

        for size in (2, 4, 8, 16, 32, 64):
            try:
                # It is possible that size just fit on device, and once we/some other process allocates 
                # some more memory allocation will fail
                # Therefore use 'size' only if the device will have some memory left
                data = torch.zeros((size + min(size, 4), *image.shape), device=device)
                label = torch.arange(0, 1, device=device).repeat((size + min(size, 4), 1))
                self.sync(device)
                controller.train_batch(model, data, label)
                self.sync(device)
                model.zero_grad(set_to_none=True)
                batch_size = size
            except torch.OutOfMemoryError:
                break

        return batch_size


    def optimise_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        model = model.to(device)
        model = model.to(memory_format=torch.channels_last) # type:ignore

        if device.type == "cuda":
            version = torch.cuda.get_device_capability(device)
            properties = torch.cuda.get_device_properties(device)

            # torch.compile requires CUDA 7.0+
            # Also, compile() crashes on some device
            # So far if device has more than 68 compute units compile() works
            # See is_big_gpu() in torch/_inductor/utils.py
            if version >= (7, 0) and properties.multi_processor_count >= 68:
                model.compile()
        else:
            model.compile()
        return model


    def make_dataloader(self, params: RunModelParams):
        if params.dist_params is not None:
            sampler=DistributedSampler(params.dataset)
        else:
            sampler = None

        loader = DataLoader(params.dataset, 
                            batch_size=self.batch_size(params.model, params.device), 
                            shuffle=True, 
                            sampler=sampler,
                            num_workers=os.cpu_count() or 0)
        return loader, sampler


    def summary_writer(self, params: RunModelParams):
        if params.is_master():
            return SummaryWriter()
        else:
            return contextlib.nullcontext()


    def train_loop(self, params: RunModelParams):
        loader, sampler = self.make_dataloader(params)
        batch_size = self.batch_size(params.model, params.device)
        self.logger.info(f"[{os.getpid()} {params.device}]: Using batch size {batch_size}")

        with self.summary_writer(params) as summary:
            for epoch in range(1, 2):
                params.model.train()

                epoch_start = time.time()

                if sampler is not None:
                    sampler.set_epoch(epoch)

                avg_loss = torch.zeros(1, device=params.device, requires_grad=False)
                zero = torch.tensor([0, 1], device=params.device).repeat((batch_size, 1))
                ones = torch.tensor([1, 0], device=params.device).repeat((batch_size, 1))

                for batch_idx, (batch, label) in enumerate(loader, start=1):
                    batch: Tensor = batch.to(params.device)
                    label: Tensor = label.to(params.device)
                    label = zero if label == AD_LABEL else ones

                    stats = params.controller.train_batch(params.model, batch, label)
                    avg_loss += stats.loss / len(loader)

                    if (batch_idx % 50) == 0:
                        self.logger.info(f"Training: epoch [{epoch}/{params.controller.num_epochs()}] batch [{batch_idx}/{len(loader)}]")
                        self.logger.info(f"\tLoss : {stats.loss.item()}")

                if params.is_master():
                    torch.save({
                        TRAIN_STATUS_KEY: TRAIN_STATUS_TRAINING,
                        MODEL_PARAMS_KEY: params.model.state_dict()
                    }, params.controller.save_path())

                self.logger.info(f"Validating...")
                eval_stats = self.eval_loop(RunModelParams(
                    model=params.model,
                    device=params.device,
                    dataset=self.dataset.validation_dataset,
                    dist_params=params.dist_params,
                    controller=params.controller
                ), tag="Validation")

                if summary is not None:
                    summary.add_scalar("Train loss", avg_loss, global_step=epoch)
                    summary.add_scalar("Validation loss", eval_stats.loss, global_step=epoch)
                    summary.add_image("Validation images(ground truth)", 
                                       vutils.make_grid(eval_stats.input_images.detach().cpu()), 
                                               global_step=epoch)
                    summary.add_image("Validation images(generated)", 
                                       vutils.make_grid(eval_stats.generated_images.detach().cpu()), 
                                       global_step=epoch)

                epoch_end = time.time()
                self.logger.info(f"Epoch {epoch} done, took {epoch_end - epoch_start:2} seconds")

            torch.save({
                TRAIN_STATUS_KEY: TRAIN_STATUS_DONE,
                MODEL_PARAMS_KEY: params.model.state_dict()
            }, params.controller.save_path())
            self.logger.info(f"Testing...")
            test_stats = self.eval_loop(RunModelParams(
                model=params.model,
                device=params.device,
                dataset=self.dataset.test_dataset,
                dist_params=params.dist_params,
                controller=params.controller
            ), tag="Test")

            if summary is not None:
                summary.add_scalar("Test loss", test_stats.loss)
                summary.add_image("Test images(ground truth)", 
                                   vutils.make_grid(test_stats.input_images.detach().cpu()))
                summary.add_image("Test images(generated)",
                                  vutils.make_grid(test_stats.generated_images.detach().cpu()))


    @torch.inference_mode()
    def eval_loop(self, params: RunModelParams, tag: str) -> EvalStats:
        params.model.eval()

        start = time.time()
        loader, sampler = self.make_dataloader(params)

        if sampler is not None:
            sampler.set_epoch(1)

        input_images = torch.empty(0)
        generated_images = torch.empty(0)
        avg_loss = torch.zeros(1, device=params.device, requires_grad=False)

        assert loader.batch_size
        zero = torch.tensor([0, 1], device=params.device).repeat((loader.batch_size, 1))
        ones = torch.tensor([1, 0], device=params.device).repeat((loader.batch_size, 1))

        for batch_idx, (batch, label) in enumerate(loader, start=1):
            with torch.autocast(device_type=params.device.type):
                batch: Tensor = batch.to(params.device)
                label = zero if label == AD_LABEL else ones

                stats = params.controller.eval_batch(params.model, batch, label)
            avg_loss += stats.loss / len(loader)

            if generated_images.shape[0] == 0:
                num_images = min(self.batch_size(params.model, params.device), 8)
                generated_images = stats.generated_images[:num_images]
                input_images = batch[:num_images]

            if (batch_idx % 50) == 0:
                self.logger.info(f"{tag}: batch [{batch_idx}/{len(loader)}]")
                self.logger.info(f"\tLoss {stats.loss.item()}")

        end = time.time()

        self.logger.info(f"{tag} done, took {end - start:2} seconds")
        self.logger.info(f"\tAverage loss: {avg_loss.item()}")
        return EvalStats(avg_loss, input_images, generated_images)


    def run_model_worker(self, 
                         rank: int, 
                         world_size: int, 
                         file: str, 
                         fn: RunModelLoop, 
                         dataset: ImageListDataset,
                         controller: ModelController):
        with self.setup(rank, world_size, file):
            accelerator = get_accelerator()
            device = torch.device(f"{accelerator.type}:{rank}")
            model = self.optimise_model(controller.model(), device)
            model = nn.parallel.DistributedDataParallel(model)

            fn(RunModelParams(
                model=model,
                device=device,
                dataset=dataset,
                dist_params=DistributedParams(rank),
                controller=controller
            ))


    def run_model(self, 
                  fn: RunModelLoop, 
                  dataset: ImageListDataset,
                  controller: ModelController):
        nprocs = torch.accelerator.device_count()
        device = get_accelerator()

        self.logger.info(f"Using accelerator kind {device.type}")

        if device.type in ("cuda", "xpu"):
            self.logger.info(f"Found {nprocs} devices:")

            for i in range(nprocs):
                if device.type == "cuda":
                    name = torch.cuda.get_device_name(i)
                else:
                    name = torch.xpu.get_device_name(i)

                self.logger.info(f"\t[{i + 1}]: {name}")
        else:
            self.logger.info(f"Found {nprocs} devices")

        # mps does not support DistributedDataParallel
        if nprocs == 1 or device and device.type == "mps":
            fn(RunModelParams(
                model=self.optimise_model(controller.model(), device),
                device=device,
                dataset=dataset,
                dist_params=None,
                controller=controller
            ))
        else:
            # delete=False since FileStore deletes the file on close
            with NamedTemporaryFile(delete=False) as file:
                mp.spawn(self.run_model_worker, # type:ignore
                         nprocs=nprocs, 
                         args=(nprocs, file.name, fn, dataset, controller), 
                         join=True) 

    def train(self, controller: ModelController):
        model_name = type(controller.model()).__name__

        if controller.save_path().exists():
            state = torch.load(controller.save_path())
            if isinstance(state, dict) and MODEL_PARAMS_KEY in state:
                train_status = state.get(TRAIN_STATUS_KEY)
                if train_status in (TRAIN_STATUS_TRAINING, TRAIN_STATUS_DONE):
                    controller.model().load_state_dict(state[MODEL_PARAMS_KEY])

                if train_status == TRAIN_STATUS_DONE:
                    self.logger.info(f"Loaded {model_name} from {controller.save_path()}")

        self.run_model(self.train_loop, 
                       self.dataset.train_dataset,
                       controller)
        self.logger.info(f"Train {model_name} completed")


@dataclass(frozen=True)
class VAEStats:
    loss: Tensor


class VAEController(ModelController):
    def __init__(self, runner: ModelRunner):
        super().__init__()
        self.vae = VAE(VAE_CONFIG, runner.dataset.image_info)
        self.optimiser = Adam(self.vae.parameters(), lr=VAE_CONFIG.learn_rate)
        self.scaler = PortableGradScaler()
 

    def num_epochs(self) -> int:
        return 30


    @staticmethod
    def loss_fn(image: Tensor, generated: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        reconstruction: Tensor = F.mse_loss(image, generated, reduction="sum")
        kld: Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reconstruction + kld


    def save_path(self) -> Path:
        return Path("vae.pth")


    def train_batch(self, model: nn.Module, batch: Tensor, label: torch.Tensor) -> RunStats:
        with torch.autocast(device_type=str(batch.device)):
            generated, mu, logvar = model(batch)

            loss = self.loss_fn(batch, generated, mu, logvar)

        self.scaler.scale(loss)
        nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1)
        self.scaler.update(self.optimiser)

        return RunStats(loss=loss, generated_images=generated)

    
    def eval_batch(self, model: nn.Module, batch: Tensor, label: torch.Tensor) -> RunStats:
        with torch.autocast(device_type=str(batch.device)):
            generated, mu, logvar = model(batch)

            loss = self.loss_fn(batch, generated, mu, logvar)
            return RunStats(loss=loss, generated_images=generated)


    def model(self) -> nn.Module:
        return self.vae


class DiffusionModelController(ModelController):
    def __init__(self, vae: VAE):
        super().__init__()

        self.diffusion = DiffusionModel(
                DIFFUSION_CONFIG, 
                VAE_CONFIG.latent_dim, 
                num_classes=2)
        self.vae = vae
        self.optimiser = Adam(self.diffusion.parameters(), DIFFUSION_CONFIG.learn_rate)
        self.scaler = PortableGradScaler()


    def model(self) -> nn.Module:
        return self.diffusion


    def num_epochs(self) -> int:
        return 30


    def train_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> RunStats:
        with torch.autocast(device_type=batch.device.type):
            latent = self.vae.encode(batch)
            t, true_eps, predict_eps, noisy_image = model(latent, 
                                                         label, 
                                                         DiffusionModelForwardMode.TRAIN)
            loss = F.mse_loss(predict_eps, true_eps)

        self.optimiser.zero_grad(set_to_none=True)
        self.scaler.scale(loss)
        self.scaler.update(self.optimiser)


        latent_images = self.diffusion.predicted_noise_to_image(
            noisy_image, 
            predict_eps, 
            t)
        images = self.vae.decode(latent_images)
        return RunStats(loss=loss, 
                        generated_images=images)


    @torch.inference_mode()
    def eval_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> RunStats:
        with torch.autocast(device_type=batch.device.type):
            latent = self.vae.encode(batch)
            t, true_eps, predict_eps, noisy_image = model(latent, 
                                                         label, 
                                                         DiffusionModelForwardMode.EVAL)

            latent_images = self.diffusion.predicted_noise_to_image(
                noisy_image, 
                predict_eps, 
                t)
            images = self.vae.decode(latent_images)
            loss = F.mse_loss(batch, images)
            return RunStats(loss=loss, 
                            generated_images=images)


    def save_path(self) -> Path:
        return Path("diffusion.pth")


def main():
    runner = ModelRunner()
    vae_controller = VAEController(runner)
    diffusion_controller = DiffusionModelController(vae_controller.vae)

    runner.train(diffusion_controller)
    runner.train(vae_controller)


if __name__ == "__main__":
    main()
