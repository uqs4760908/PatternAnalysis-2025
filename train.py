from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from config import DiffusionConfig, EncoderDecoderConfig, ImageInfo, VAEConfig
from dataset import AD_LABEL, NC_LABEL, NUM_CLASS, ANDIDataset, ImageListDataset
from pathlib import Path
from modules import VAE, DiffusionModel, DiffusionSampler
from torch import multiprocessing as mp
from torch import distributed as dist
from torch import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel
from torchvision import utils as vutils
from tempfile import NamedTemporaryFile
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from dataclasses import dataclass
from io import StringIO
import sys
import time
import typing
import abc
import contextlib
import torch
import os


# From https://github.com/CompVis/latent-diffusion/blob/main/models/ldm/celeba256/config.yaml
# and https://github.com/Stability-AI/stablediffusion/blob/main/configs/stable-diffusion/v2-inference-v.yaml
VAE_CONFIG = VAEConfig(
    learn_rate=1e-4,
    encoder_decoder_config=EncoderDecoderConfig(
        num_channels=(
            1 * 64,
            2 * 64,
            4 * 64,
            4 * 64
        ),
        should_downsample=(
            True,
            True,
            True,
            False
        ),
        num_resnet_blocks=2,
        num_attention_heads=2,
        layer_norm_num_groups=32,
        embedding_dim=None,
        use_attention=(False,) * 4
    ),
    latent_dim=4,
    weight_decay=1e-5,
)

DIFFUSION_CONFIG = DiffusionConfig(
    learn_rate=1e-4,

    noise_start=0.00085,
    noise_end=0.0120,
    denoise_steps=1000,
    weight_decay=1e-6,
    unet_config=EncoderDecoderConfig(
        num_channels=(
            1 * 128,
            2 * 128,
            4 * 128,
            4 * 128
        ),
        should_downsample=(
            True,
            True,
            False,
            False
        ),
        num_resnet_blocks=2,
        num_attention_heads=2,
        layer_norm_num_groups=32,
        embedding_dim=320,
        use_attention=(True,) * 4
    )
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
        else:
            loss.backward()


    def update(self, optimiser: Optimizer):
        if self.scaler is not None:
            self.scaler.step(optimiser)
            self.scaler.update()
        else:
            optimiser.step()


@dataclass(frozen=True)
class DistributedParams:
    rank: int


RunModelFn = typing.Callable[[nn.Module, Tensor], typing.Any]


@dataclass(frozen=True)
class DeviceStats:
    temperature: int
    usage: int
    memory_allocated: int

    @staticmethod
    def capture(device: torch.device) -> typing.Optional["DeviceStats"]:
        if device.type == "cuda":
            return DeviceStats(
                    temperature=torch.cuda.temperature(device),
                    usage=torch.cuda.utilization(device),
                    memory_allocated=torch.cuda.memory_allocated(device)
            )
        return None


@dataclass(frozen=True)
class TrainBatchStats:
    loss: Tensor
    device_stats: typing.Optional[DeviceStats]


@dataclass(frozen=True)
class EvalBatchStats:
    loss: Tensor
    images: dict[str, Tensor]
    device_stats: typing.Optional[DeviceStats]


@dataclass(frozen=True)
class EvalStats:
    loss: Tensor
    generated_images: dict[str, Tensor]

    
class ModelController(abc.ABC):
    """
    Controls how a model should be executed
    """
    @abc.abstractmethod
    def model(self) -> nn.Module: ...


    def dependent_models(self) -> typing.Sequence[nn.Module]:
        return ()


    def prepare_train(self):
        pass


    def prepare_eval(self):
        pass


    @abc.abstractmethod
    def name(self) -> str: ...


    @abc.abstractmethod
    def num_epochs(self) -> int: ...


    @abc.abstractmethod
    def step(self, loss: Tensor) -> None: ...


    @abc.abstractmethod
    def get_lr(self) -> float: ...

    
    @abc.abstractmethod
    def train_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> TrainBatchStats: ...


    @abc.abstractmethod
    def eval_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> EvalBatchStats: ...

    
    @abc.abstractmethod
    def generate(self, n: int, batch: Tensor, label: Tensor) -> dict[str, Tensor]: ...

    
    @abc.abstractmethod
    def save_model(self): ...


    @abc.abstractmethod
    def load_model(self): ...


@dataclass(frozen=True)
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
        self.batch_size_cache: dict[torch.device, int] = {}


    def log(self, *args):
        if dist.is_initialized():
            rank = f" Rank [{dist.get_rank()}] "
        else:
            rank = ""

        if sys.stdout.isatty():
            color = "\033[42m" #] <-- fix neovim indentation bug
            clear = "\033[0m" #]
        else:
            color = ""
            clear = ""

        io = StringIO()
        print(f"[{color}INFO{clear}]{rank}", *args, file=io)
        print(io.getvalue(), end="")


    @contextlib.contextmanager
    def setup(self, rank: int, world_size: int, file: str):
        device = torch.accelerator.current_accelerator()
        torch.set_float32_matmul_precision("high")

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cudnn.benchmark = True

        if device is None:
            return

        store = dist.FileStore(file, world_size) # type:ignore
        torch.accelerator.set_device_index(rank)
        backend = dist.get_default_backend_for_device(device)
        dist.init_process_group(backend, store=store, rank=rank, world_size=world_size, device_id=rank)
        try:
            self.log(f"Worker {os.getpid()} initialised: rank={rank} backend={backend}")
            yield None
        finally:
            dist.destroy_process_group()


    @staticmethod
    def sync(device: torch.device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        
    
    def batch_size(self, controller: ModelController, model: nn.Module, device: torch.device) -> int:
        if device in self.batch_size_cache:
            return self.batch_size_cache[device]
        image = self.dataset.train_dataset[0][0]

        batch_size = 1

        for size in (2, 4, 8, 16):
            try:
                # It is possible that size just fit on device, and once we/some other process allocates 
                # some more memory allocation will fail
                # Therefore use 'size' only if the device will have some memory left
                data = torch.zeros((size + min(size // 2, 4), *image.shape), device=device)
                label = torch.arange(0, 2, device=device).repeat((size + min(size // 2, 4), 1))
                self.sync(device)
                controller.train_batch(model, data, label)
                self.sync(device)
                controller.eval_batch(model, data, label)
                batch_size = size
            except torch.OutOfMemoryError:
                break

        self.batch_size_cache[device] = batch_size
        return batch_size


    def optimise_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        if device.type == "cuda":
            version = torch.cuda.get_device_capability(device)
            properties = torch.cuda.get_device_properties(device)

            # torch.compile requires CUDA 7.0+
            # Also, compile() crashes on my RTX 3060
            if version >= (7, 0) and properties.name != "NVIDIA GeForce RTX 3060":
                model.compile()
        else:
          model.compile()
        return model


    def send_model_to_device(self, controller: ModelController, device: torch.device):
        for model in [controller.model(), *controller.dependent_models()]:
            model.to(device)
        return controller.model()


    def make_dataloader(self, params: RunModelParams, epoch: int):
        if params.dist_params is not None:
            sampler=DistributedSampler(params.dataset, shuffle=True)
            sampler.set_epoch(epoch)
        else:
            sampler = None

        loader = DataLoader(params.dataset, 
                            batch_size=self.batch_size(params.controller ,params.model, params.device), 
                            shuffle=None if sampler is not None else True, 
                            sampler=sampler,
                            num_workers=min(12, os.cpu_count() or 0))
        return loader


    def summary_writer(self, params: RunModelParams):
        if params.is_master():
            return SummaryWriter()
        else:
            return contextlib.nullcontext()


    def train_loop(self, params: RunModelParams):
        batch_size = self.batch_size(params.controller, params.model, params.device)
        self.log(f"Using batch size {batch_size}")

        self.optimise_model(params.model, params.device)

        with self.summary_writer(params) as summary:
            last_loss = float("inf")

            train_start = time.time()
            for epoch in range(1, params.controller.num_epochs() + 1):
                params.controller.prepare_train()
                params.model.train()

                epoch_start = time.time()

                avg_loss = torch.zeros(1, device=params.device)

                loader = self.make_dataloader(params, epoch)
                for batch_idx, (batch, label) in enumerate(loader, start=1):
                    batch: Tensor = batch.to(params.device)
                    one_hot_label = F.one_hot(label, NUM_CLASS).to(params.device)

                    stats = params.controller.train_batch(params.model, batch, one_hot_label)
                    with torch.no_grad():
                        avg_loss += stats.loss / len(loader)

                    if (batch_idx % 50) != 0:
                        continue

                    self.log(f"Training: epoch [{epoch}/{params.controller.num_epochs()}] batch [{batch_idx}/{len(loader)}]")
                    self.log(f"\tLoss: {stats.loss.item()}")

                    if stats.loss.isnan().item():
                        raise Exception("NaN loss detected")

                    if stats.device_stats is None:
                        continue
                    
                    self.log(f"\tGPU usage: {stats.device_stats.usage}%")
                    self.log(f"\tGPU temperature: {stats.device_stats.temperature}C")
                    self.log(f"\tGPU memory allocated: {(stats.device_stats.memory_allocated / (2 ** 30)):.2f}GB")

                    if summary is not None:
                        step = (epoch - 1) * len(loader) + batch_idx
                        summary.add_scalar("Train/GPU usage(%)", stats.device_stats.usage, step)
                        summary.add_scalar("Train/GPU temperature(C)", stats.device_stats.temperature, step)
                        summary.add_scalar("Train/GPU memory allocated(GB)", stats.device_stats.memory_allocated / (2 ** 30), step)

                self.log(f"Validating...")
                eval_stats = self.train_eval(params, summary, 
                                   self.dataset.validation_dataset, 
                                   tag="Validation",
                                   step=epoch)

                if params.is_master() and last_loss > eval_stats.loss.item():
                    # Note: do not use params.model here since it might be DistributedDataParallel
                    # When loading saved model we are loading params.controller.model(),
                    # so be consistent
                    params.controller.save_model()
                    last_loss = float(eval_stats.loss.item())

                if summary is not None:
                    summary.add_scalar("Train/loss", avg_loss, global_step=epoch)
                    summary.add_scalar("Train/learn rate", params.controller.get_lr(), global_step=epoch)

                params.controller.step(eval_stats.loss)
                epoch_end = time.time()
                self.log(f"Epoch {epoch} done, took {epoch_end - epoch_start:2} seconds")

            if dist.is_initialized():
                dist.barrier()

            train_end = time.time()
            self.log(f"Training done, took {train_end - train_start:.2} seconds")

            # Load best model
            params.controller.load_model()
            self.log(f"Testing...")
            self.train_eval(
                train_params=params,
                summary=summary,
                dataset=self.dataset.test_dataset,
                tag="Test",
                step=0
            )


    def train_eval(self, 
                  train_params: RunModelParams, 
                  summary: typing.Optional[SummaryWriter], 
                  dataset: ImageListDataset,
                  tag: str, 
                  step: int):
        params = RunModelParams(
            model=train_params.model,
            device=train_params.device,
            dataset=dataset,
            dist_params=train_params.dist_params,
            controller=train_params.controller)
        stats = self.eval_loop(params, tag=tag)
        self.log(f"{tag} loss: {stats.loss.item()}")

        if summary is None:
            return stats

        loader = self.make_dataloader(params, epoch=0)
        batch, labels = next(iter(loader))
        batch = batch.to(params.device)
        one_hot_label = F.one_hot(labels, NUM_CLASS).to(params.device)
        images = train_params.controller.generate(16, batch, one_hot_label)
        images.update(stats.generated_images)

        summary.add_scalar(f"{tag}/loss", stats.loss, global_step=step)
        for name, images in images.items():
            summary.add_image(f"{tag}/{name}", vutils.make_grid(images.detach().cpu()), global_step=step)
        return stats


    @torch.inference_mode()
    def eval_loop(self, params: RunModelParams, tag: str) -> EvalStats:
        params.model.eval()
        params.controller.prepare_eval()

        start = time.time()

        loader = self.make_dataloader(params, epoch=0)

        generated_images = {}
        avg_loss = torch.zeros(1, device=params.device, requires_grad=False)

        assert loader.batch_size

        for batch_idx, (batch, label) in enumerate(loader, start=1):
            with torch.autocast(device_type=params.device.type):
                batch: Tensor = batch.to(params.device)
                one_hot_label = F.one_hot(label, NUM_CLASS).to(params.device)

                stats = params.controller.eval_batch(params.model, batch, one_hot_label)
            avg_loss += stats.loss / len(loader)

            if len(generated_images) == 0:
                generated_images = stats.images

            if (batch_idx % 50) == 0:
                self.log(f"{tag}: batch [{batch_idx}/{len(loader)}]")
                self.log(f"\tLoss {stats.loss.item()}")

        end = time.time()

        self.log(f"{tag} done, took {end - start:2} seconds")
        self.log(f"\tAverage loss: {avg_loss.item()}")
        return EvalStats(avg_loss, generated_images)


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
            model = self.send_model_to_device(controller, device)
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

        self.log(f"Using accelerator kind {device.type}")

        if device.type in ("cuda", "xpu"):
            self.log(f"Found {nprocs} devices:")

            for i in range(nprocs):
                if device.type == "cuda":
                    name = torch.cuda.get_device_name(i)
                else:
                    name = torch.xpu.get_device_name(i)

                self.log(f"\t[{i + 1}]: {name}")
        else:
            self.log(f"Found {nprocs} devices")

        # mps does not support DistributedDataParallel
        if nprocs == 1 or (device and device.type == "mps"):
            fn(RunModelParams(
                model=self.send_model_to_device(controller, device),
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
        
        try:
            controller.load_model()
            self.log(f"Loaded {model_name}")
            return
        except:
            pass

        self.log(f"Training {model_name}")
        self.run_model(self.train_loop, 
                       self.dataset.train_dataset,
                       controller)
        self.log(f"Train {model_name} completed")


@dataclass(frozen=True)
class VAEStats:
    loss: Tensor


class VAEController(ModelController):
    MODEL_PATH = "vae.pth"

    def __init__(self, dataset: ANDIDataset):
        super().__init__()
        self.vae = VAE(VAE_CONFIG, dataset.image_info)
        self.optimiser = AdamW(self.vae.parameters(), 
                              lr=VAE_CONFIG.learn_rate, fused=True, 
                              weight_decay=VAE_CONFIG.weight_decay)
        self.scaler = PortableGradScaler()
        self.scheduler = CosineAnnealingLR(self.optimiser, self.num_epochs())


    def num_epochs(self) -> int:
        return 30


    @staticmethod
    def loss_fn(image: Tensor, generated: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        reconstruction: Tensor = F.mse_loss(image, generated, reduction="none").sum(dim=[1, 2, 3]).mean()
        kld: Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]).mean()

        return reconstruction + 1e-6 * kld

    
    def step(self, loss: Tensor) -> None:
        self.scheduler.step()


    def name(self) -> str:
        return "VAE"


    def get_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]


    def train_batch(self, model: nn.Module, batch: Tensor, label: torch.Tensor) -> TrainBatchStats:
        self.optimiser.zero_grad(set_to_none=True)

        with torch.autocast(device_type=batch.device.type):
            generated, mu, logvar = model(batch)

            loss = self.loss_fn(batch, generated, mu, logvar)
            device_stats = DeviceStats.capture(batch.device)

        self.scaler.scale(loss)
        nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1)
        self.scaler.update(self.optimiser)

        return TrainBatchStats(loss=loss, device_stats=device_stats)

    
    @torch.inference_mode()
    def eval_batch(self, model: nn.Module, batch: Tensor, label: torch.Tensor) -> EvalBatchStats:
        with torch.autocast(device_type=batch.device.type):
            generated, mu, logvar = model(batch)

            loss = self.loss_fn(batch, generated, mu, logvar)
            images = {
                    "Ground truth": batch,
                    "Generated": generated
            }
            return EvalBatchStats(loss=loss, images=images, device_stats=DeviceStats.capture(batch.device))


    def generate(self, n: int, batch: Tensor, label: Tensor) -> dict[str, Tensor]:
        return {}


    def model(self) -> nn.Module:
        return self.vae


    def load_model(self):
        self.vae.load_state_dict(torch.load(self.MODEL_PATH, weights_only=True))


    def save_model(self):
        torch.save(self.vae.state_dict(), self.MODEL_PATH)


def compute_ema(avg_param: Tensor, model_param: Tensor, *_):
    return 0.999 * avg_param + (1 - 0.999) * model_param


class DiffusionModelController(ModelController):
    EMA_MODEL_PATH = "diffusion_model.pt"

    def __init__(self, vae: VAE, image_info: ImageInfo):
        super().__init__()

        self.diffusion_model = DiffusionModel(DIFFUSION_CONFIG, 
                                              VAE_CONFIG.latent_dim, NUM_CLASS)
        self.ema_model = AveragedModel(self.diffusion_model, avg_fn=compute_ema)
        self.sampler = DiffusionSampler(DIFFUSION_CONFIG)
        self.image_info = image_info
        self.vae = vae
        self.optimiser = AdamW(self.diffusion_model.parameters(), 
                              DIFFUSION_CONFIG.learn_rate, fused=True, 
                              weight_decay=DIFFUSION_CONFIG.weight_decay)
        self.scaler = PortableGradScaler()


    def prepare_train(self):
        self.ema_model.train()
        self.sampler.eval()
        self.vae.eval()


    def prepare_eval(self):
        self.ema_model.eval()
        self.sampler.eval()
        self.vae.eval()
        

    def model(self) -> nn.Module:
        return self.diffusion_model


    def num_epochs(self) -> int:
        return 200


    def name(self) -> str:
        return "Diffusion"


    def dependent_models(self) -> typing.Sequence[nn.Module]:
        return (self.vae, self.ema_model, self.sampler)


    def step(self, loss: Tensor) -> None:
        pass


    def get_lr(self) -> float:
        return DIFFUSION_CONFIG.learn_rate


    def loss(self, noise: Tensor, predict_eps: Tensor) -> Tensor:
        loss = F.mse_loss(predict_eps, noise)
        return loss


    def train_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> TrainBatchStats:
        label = label.float()
        self.optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device_type=batch.device.type):
            with torch.no_grad():
                latent = self.vae.encode(batch)

            t = torch.randint(0, DIFFUSION_CONFIG.denoise_steps, 
                              size=(batch.size(0),), device=batch.device, dtype=torch.int32)
            x_t, noise = self.sampler.add_noise(latent, t)
            predict_eps = model(x_t, t, label)
            loss = self.loss(noise, predict_eps)

            device_stats = DeviceStats.capture(batch.device)

        self.scaler.scale(loss)
        self.scaler.update(self.optimiser)
        self.ema_model.update_parameters(model)

        return TrainBatchStats(loss=loss, device_stats=device_stats)


    @torch.inference_mode()
    def eval_batch(self, model: nn.Module, batch: Tensor, label: Tensor) -> EvalBatchStats:
        label = label.float()
        with torch.autocast(device_type=batch.device.type):
            latent = self.vae.encode(batch)
            t = torch.full(batch.shape[:1], DIFFUSION_CONFIG.denoise_steps - 1,
                           device=batch.device, dtype=torch.int32)
            x_t, noise = self.sampler.add_noise(latent, t)
            predict_eps = self.ema_model(x_t, t, label)
            loss = self.loss(noise, predict_eps)

            images = {
                "Ground truth": batch,
            }
            return EvalBatchStats(loss=loss, 
                            images=images,
                            device_stats=DeviceStats.capture(batch.device))


    @torch.inference_mode()
    def generate(self, n: int, batch: Tensor, label: Tensor) -> dict[str, Tensor]:
        label = label.float()
        with torch.autocast(device_type=batch.device.type):
            ad_label = F.one_hot(torch.tensor([AD_LABEL]), NUM_CLASS).to(batch.device).float()
            nc_label = F.one_hot(torch.tensor([NC_LABEL]), NUM_CLASS).to(batch.device).float()
            height: int = self.image_info.size[0] // (2 ** sum(VAE_CONFIG.encoder_decoder_config.should_downsample))
            width: int = self.image_info.size[1] // (2 ** sum(VAE_CONFIG.encoder_decoder_config.should_downsample))
            size = height, width

            ad_latent = self.sampler.generate(n, size, 
                                              VAE_CONFIG.latent_dim, ad_label, self.ema_model)
            nc_latent = self.sampler.generate(n, size, 
                                              VAE_CONFIG.latent_dim, nc_label, self.ema_model)

            return {
                "Generate AD": self.vae.decode(ad_latent),
                "Generate CN": self.vae.decode(nc_latent)
            }

        
    def load_model(self):
        self.ema_model.load_state_dict(torch.load(self.EMA_MODEL_PATH, weights_only=True))


    def save_model(self):
        torch.save(self.ema_model.state_dict(), self.EMA_MODEL_PATH)


def main():
    runner = ModelRunner()
    vae_controller = VAEController(runner.dataset)
    diffusion_controller = DiffusionModelController(vae_controller.vae, runner.dataset.image_info)

    runner.train(vae_controller)
    runner.train(diffusion_controller)


if __name__ == "__main__":
    main()
