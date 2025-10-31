import math
from os import times
from torch import nn
from torch.nn import functional as F
from config import EncoderDecoderConfig, VAEConfig, ImageInfo, DiffusionConfig
from enum import Enum
from torchvision.models import inception_v3, Inception_V3_Weights
from dataclasses import dataclass
import dataclasses
import torch
import typing
import abc


try:
    import flash_attn
except ImportError:
    flash_attn = None


class SequentialWithEmbedding(nn.ModuleList):
    def __init__(self, *layers: nn.Module):
        super().__init__(layers)


    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None):
        for layer in self:
            batch = layer(batch, embedding)
        return batch


@dataclass(frozen=True)
class ResNetBlockInfo:
    in_channels: int
    out_channels: int
    layer_norm_num_groups: int
    embedding_dim: typing.Optional[int]


class ResNetBlock(nn.Module):
    def __init__(self, info: ResNetBlockInfo):
        super().__init__()

        self.input_net = nn.Sequential(
                nn.Conv2d(
                        in_channels=info.in_channels, 
                        out_channels=info.out_channels, 
                        kernel_size=3, 
                        padding=1, 
                        bias=False),
                nn.GroupNorm(
                    num_groups=info.layer_norm_num_groups, 
                    num_channels=info.out_channels),
                nn.SiLU(inplace=True)
        )
        self.output_net = nn.Sequential(
                nn.Conv2d(
                    in_channels=info.out_channels, 
                    out_channels=info.out_channels, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False),
                nn.GroupNorm(
                    num_groups=info.layer_norm_num_groups, 
                    num_channels=info.out_channels)
        )

        if info.embedding_dim is not None:
            self.embedding_projection_net = nn.Sequential(
                nn.Linear(
            info.embedding_dim,
            info.out_channels
                ),
                nn.SiLU(inplace=True),
                nn.Linear(info.out_channels, info.out_channels)
            )
        else:
            self.embedding_projection_net = None

        if info.in_channels != info.out_channels:
            self.skip_connection: nn.Module = nn.Conv2d(
                    in_channels=info.in_channels, 
                    out_channels=info.out_channels, 
                    kernel_size=1)
        else:
            self.skip_connection: nn.Module = nn.Identity()

    
    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        output: torch.Tensor = self.input_net(batch)

        if embedding is not None and self.embedding_projection_net:
            embedding_projection: torch.Tensor = self.embedding_projection_net(embedding)
            embedding_projection = embedding_projection.view(*embedding_projection.shape[:2], 1, 1)
            output = output + embedding_projection
        elif self.embedding_projection_net is not None and embedding is None:
            raise Exception()

        skipped: torch.Tensor = self.skip_connection(batch)
        output = self.output_net(output)
        return (output + skipped).relu()


class ResNetBlockList(SequentialWithEmbedding):
    def __init__(self, num_blocks: int, info: ResNetBlockInfo):
        inner_info = dataclasses.replace(info, in_channels=info.out_channels)
        super().__init__(
            ResNetBlock(info),
            *(ResNetBlock(inner_info) for _ in range(num_blocks - 1))
        )


class PixelTransformer(nn.Module):
    def __init__(self, num_channels: int, num_heads: int):
        super().__init__()

        if num_channels % num_heads != 0:
            raise ValueError(f"num_channels({num_channels}) must be divisable by num_heads({num_heads})")
        self.qkv_projection = nn.Conv2d(num_channels, num_channels * 3, kernel_size=1)
        self.num_heads = num_heads

        if flash_attn is None:
            self.net = nn.MultiheadAttention(num_channels, num_heads)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        images = self.qkv_projection(images) # shape: (batch, channels * 3, height, width)
        images = images.view(batch, 3, self.num_heads, channels // self.num_heads, height * width).permute(0, 4, 1, 2, 3)

        if flash_attn is not None:
            patches = typing.cast(torch.Tensor, flash_attn.flash_attn_qkvpacked_func(images))
            patches = patches.view(batch, height * width, channels).transpose(1, 2)

            return patches.view(batch, channels, height, width)


        q, k, v = (t.view(batch, channels, height * width).transpose(1, 2) 
                   for t in images.chunk(3, 1))
        patches: torch.Tensor = self.net(q, k, v, need_weights=False)[0]
        return patches.transpose(1, 2).view(batch, channels, height, width)


class EncoderStage(nn.Module):
    def __init__(self, stage_index: int, config: EncoderDecoderConfig):
        super().__init__()

        channels = (config.num_channels[0], *config.num_channels)
        info = ResNetBlockInfo(
            in_channels=channels[stage_index],
            out_channels=channels[stage_index + 1],
            layer_norm_num_groups=config.layer_norm_num_groups,
            embedding_dim=config.embedding_dim
        )
        self.resnet_list = ResNetBlockList(
        config.num_resnet_blocks,
            info
        )

        if config.use_attention_in_up_down_sampling:
            self.transformer = PixelTransformer(
                    info.out_channels, 
                    config.num_attention_heads)
        else:
            self.transformer = nn.Identity()


    def conv(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.resnet_list(batch, embedding)
        output = self.transformer(output)
        return output


    def downsample(self, batch: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(batch, kernel_size=2)


    def forward(self, batch: torch.Tensor, 
                embedding: typing.Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.conv(batch, embedding)
        downsampled = self.downsample(output)
        return output, downsampled


class Autoencoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor: ...


class VAE(nn.Module, Autoencoder):
    def __init__(self, config: VAEConfig, image_info: ImageInfo):
        super().__init__()
        
        self.project_channels = nn.Conv2d(
                        in_channels=image_info.depth,
                        out_channels=config.encoder_decoder_config.num_channels[0],
                        kernel_size=1
                    )
        self.encoder = nn.ModuleList(
            [
                        EncoderStage(i, config.encoder_decoder_config)
                            for i in range(len(config.encoder_decoder_config.num_channels))
                    ]
        )

        self.should_downsample = config.encoder_decoder_config.should_downsample
        self.feature_to_mean_logvar = nn.Conv2d(
                in_channels=config.encoder_decoder_config.num_channels[-1],
                out_channels=config.latent_dim * 2,
                kernel_size=3,
                padding=1
        )
        self.mean_logvar_to_feature = nn.Conv2d(
                in_channels=config.latent_dim,
                out_channels=config.encoder_decoder_config.num_channels[-1],
                kernel_size=3,
                padding=1
        )

        channels = (config.encoder_decoder_config.num_channels[0], 
                    *config.encoder_decoder_config.num_channels)

        decoder_layers: list[nn.Module] = []
        for i in reversed(range(len(config.encoder_decoder_config.num_channels))):
            resblocks = ResNetBlockList(
                num_blocks=config.encoder_decoder_config.num_resnet_blocks,
                info=ResNetBlockInfo(
                    in_channels=channels[i + 1],
                    out_channels=channels[i],
                    layer_norm_num_groups=config.encoder_decoder_config.layer_norm_num_groups,
                    embedding_dim=config.encoder_decoder_config.embedding_dim
                )
            )
            decoder_layers.append(resblocks)
            if config.encoder_decoder_config.should_downsample[i]:
                upsampler = nn.ConvTranspose2d(
                        in_channels=channels[i],
                        out_channels=channels[i],
                        kernel_size=3,
                        padding=1,
                        output_padding=1,
                        stride=2
                    )
                decoder_layers.append(upsampler)


        self.decoder = nn.Sequential(
                *decoder_layers,
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=image_info.depth,
                    kernel_size=1
                ),
            nn.Sigmoid()
        )


    def encode_vars(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.project_channels(image)

        for layer, downsample in zip(self.encoder, self.should_downsample):
            output, downsampled = layer(output)
            if downsample:
                output = downsampled

        project: torch.Tensor = self.feature_to_mean_logvar(output)
        mean, logvar = project.chunk(2, dim=1)

        return mean, logvar
    

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode_vars(image)
        return self.sample(mean, logvar)


    def sample(self, mean: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.rand_like(mean)
        else:
            return mean


    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        feature = self.mean_logvar_to_feature(latent_vector)
        decoded = self.decoder(feature)
        return decoded


    def forward(self, image: torch.Tensor):
        mean, logvar = self.encode_vars(image)
        latent_vector = self.sample(mean, logvar)
        return self.decode(latent_vector), mean, logvar


class UNetUpsampler(nn.Module):
    def __init__(self, stage: int, config: EncoderDecoderConfig):
        super().__init__()

        channels = (config.num_channels[0], *config.num_channels)

        if config.should_downsample[stage]:
            self.upsampler = nn.ConvTranspose2d(
                    in_channels=channels[stage + 1],
                    out_channels=channels[stage],
                    kernel_size=3,
                    padding=1,
                    output_padding=1,
                    stride=2
            )
        else:
            self.upsampler = nn.Conv2d(
                in_channels=channels[stage + 1],
                out_channels=channels[stage],
                kernel_size=1,
            )

        self.conv = ResNetBlockList(
                num_blocks=config.num_resnet_blocks,
                info=ResNetBlockInfo(
                    in_channels=channels[stage] + channels[stage + 1],
                    out_channels=channels[stage],
                    layer_norm_num_groups=config.layer_norm_num_groups,
                    embedding_dim=config.embedding_dim
                )
        )


class UNet(nn.Module):
    def __init__(self, 
                 config: DiffusionConfig, 
                 latent_channels: int):
        super().__init__()

        self.downsample_projection = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=config.unet_config.num_channels[0],
            kernel_size=1,
        )
        self.downsample_passes = nn.ModuleList([
            EncoderStage(i, config.unet_config) 
                for i in range(len(config.unet_config.num_channels))
        ])
        self.should_downsample = config.unet_config.should_downsample

        info = ResNetBlockInfo(
            in_channels=config.unet_config.num_channels[-1],
            out_channels=config.unet_config.num_channels[-1],
            layer_norm_num_groups=config.unet_config.layer_norm_num_groups,
            embedding_dim=config.unet_config.embedding_dim
        )
        self.mid_resnet1 = ResNetBlock(info)
        self.mid_pixel_transformer = PixelTransformer(
            num_channels=info.out_channels,
            num_heads=config.unet_config.num_attention_heads
        )
        self.mid_resnet2 = ResNetBlock(info)

        self.upsample_passes = nn.ModuleList([
            UNetUpsampler(i, config=config.unet_config)
                for i in reversed(range(len(config.unet_config.num_channels) - 1))
        ])
        self.upsample_projection = nn.Conv2d(
                in_channels=config.unet_config.num_channels[0],
                out_channels=latent_channels,
                kernel_size=1
        )


    def forward(self, batch: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        downsample_outputs: list[torch.Tensor] = []

        output: torch.Tensor = self.downsample_projection(batch)
        for module, downsample in zip(self.downsample_passes, self.should_downsample):
            module = typing.cast(EncoderStage, module)
            output = module.conv(output, embedding)
            downsample_outputs.append(output)

            if downsample:
                output = module.downsample(output)

        output = self.mid_resnet1(output, embedding)
        output = self.mid_pixel_transformer(output)
        output = self.mid_resnet2(output, embedding)

        for module, downsample_output in zip(
                self.upsample_passes, 
                reversed(downsample_outputs[:-1])):
            module = typing.cast(UNetUpsampler, module)
            output = module.upsampler(output)

            if output.shape[2:] != downsample_output.shape[2:]:
                output = F.interpolate(output, size=downsample_output.shape[2:])

            output = torch.cat([output, downsample_output], dim=1)
            output = module.conv(output, embedding)

        output = self.upsample_projection(output)
        return output


class DiffusionModelForwardMode(Enum):
    """
    Since we might wrap DiffusionModel in DistributedDataParallel,
    the only entry point to DiffusionModel is forward(aka __call__)
    During training, forward() samples a random t, where in inference
    we want to run it over all steps
    """
    TRAIN = 0
    EVAL = 1


class DiffusionModel(nn.Module):
    @staticmethod
    def beta_schedules(config: DiffusionConfig):
        return torch.linspace(config.noise_start, config.noise_end, config.denoise_steps)


    @staticmethod
    @torch.no_grad
    def generate_timesteps_embeddings(timesteps: torch.Tensor, embedding_size: int):
        # Positional embedding:
        # PE(pos, 2i) = sin(pos / 10000*(2i/d))
        # PE(pos, 2i + 1) = cos(pos / 10000*(2i/d))
        # Compute it in log space since 10000*(2i/d) is too large
        out = torch.empty((timesteps.size(0), embedding_size), device=timesteps.device)
        # The bitand -2 rounds odd elements down to the closest even
        # This extracts 'i'(in the formula above) from 0..N
        denom = torch.exp(-math.log(10000) * 2 * (torch.arange(0, embedding_size) & -2) / embedding_size)
        numerator = timesteps.reshape(timesteps.size(0), 1)
        angles = numerator * denom

        odds = angles[1::2]
        evens = angles[::2]
        
        out[::2] = torch.sin(evens)
        out[1::2] = torch.cos(odds)
        return out


    def __init__(self, config: DiffusionConfig, latent_channels: int, num_classes: int):
        super().__init__()

        assert config.unet_config.embedding_dim
        self.timesteps = torch.arange(0, config.denoise_steps)
        self.timesteps_embeddings = self.generate_timesteps_embeddings(
                self.timesteps, 
                config.unet_config.embedding_dim)
        self.betas = self.beta_schedules(config)
        self.unet = UNet(config, latent_channels)
        self.denoise_steps = config.denoise_steps

        self.alphas: torch.Tensor = 1 - self.betas
        self.alpha_bars: torch.Tensor = self.alphas.cumprod(0)

        self.timesteps_projection = nn.Sequential(
                nn.Linear(config.unet_config.embedding_dim, config.unet_config.embedding_dim),
                nn.SiLU(inplace=True),
                nn.Linear(config.unet_config.embedding_dim, config.unet_config.embedding_dim)
        )
        self.label_projection = nn.Sequential(
                nn.Linear(num_classes, config.unet_config.embedding_dim),
                nn.SiLU(inplace=True),
                nn.Linear(config.unet_config.embedding_dim, config.unet_config.embedding_dim)
        )

        # Hack: IDE will not work without self.buffer = ... assignment,
        # but torch will complain if a buffer with name already exists
        def register(name: str, buffer: torch.Tensor):
            delattr(self, name)
            self.register_buffer(name, buffer)

        register("timesteps", self.timesteps)
        register("timesteps_embeddings", self.timesteps_embeddings)
        register("betas", self.betas)
        register("alphas", self.alphas)
        register("alpha_bars", self.alpha_bars)


    @torch.inference_mode()
    def predicted_noise_to_image(self, noise: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t].view(noise.size(0), 1, 1, 1)

        eps = (1 - alpha_bar_t).sqrt() * eps 
        return (1 / alpha_bar_t.sqrt()) * (noise - eps)


    def embed_from_label(self, label: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        timesteps_embedding = self.timesteps_embeddings[t]
        timesteps_embedding = self.timesteps_projection(timesteps_embedding)
        label_embedding = self.label_projection(label)
        embedding = timesteps_embedding + label_embedding
        return embedding


    @torch.inference_mode()
    def denoise_step(self, x_t: torch.Tensor, eps_t: torch.Tensor, t: int) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        alpha_t = 1 - beta_t

        coeff_x_t = beta_t * (1 - alpha_bar_t).rsqrt()
        mu_t = alpha_t.rsqrt() * (x_t - coeff_x_t * eps_t)

        x_t = mu_t

        if t > 0:
            alpha_bar_t_prev = self.alpha_bars[t - 1]
            sigma_t = (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t
            eps_t_prev = torch.normal(0, 1, size=x_t.shape, device=x_t.device)
            x_t = x_t + sigma_t.sqrt() * eps_t_prev

        return x_t


    @torch.inference_mode()
    def denoise(self, x_t: torch.Tensor, label: torch.Tensor):
        for t in reversed(range(self.timesteps.size(0))):
            timestep = torch.full((x_t.size(0), ), t, device=label.device)
            embedding = self.embed_from_label(label, timestep)
            eps_t = self.unet(x_t, embedding)
            x_t = self.denoise_step(x_t, eps_t, t)

        return x_t


    @torch.inference_mode()
    def generate(self, num_images: int, size: tuple[int, int], latent_dim: int, label: torch.Tensor) -> torch.Tensor:
        x_t = torch.normal(0, 1, 
                             size=(num_images, latent_dim, *size), 
                           device=label.device)
        return self.denoise(x_t, label)


    def add_noise_step(self, x_t: torch.Tensor, t: int):
        """
        Given x_t, compute x_t+1
        """
        beta_t = self.betas[t]
        eps = torch.normal(0, 1, size=x_t.shape)

        x_t_next: torch.Tensor = (1 - beta_t).sqrt() * x_t + beta_t.sqrt() * eps

        return x_t_next, eps


    def add_noise(self, batch: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.normal(0, 1, size=batch.shape, device=batch.device)
        alpha_bar = self.alpha_bars[t].view(batch.size(0), 1, 1, 1)
        x_t = alpha_bar.sqrt() * batch + (1 - alpha_bar).sqrt() * eps
        return x_t, eps


    def reconstruct(self, batch: torch.Tensor, label: torch.Tensor, t: torch.Tensor):
        x_t, eps = self.add_noise(batch, t)
        embedding = self.embed_from_label(label, t)

        prediction: torch.Tensor = self.unet(x_t, embedding)

        return t, eps, prediction, x_t


    def forward(self, batch: torch.Tensor, label: torch.Tensor, mode: DiffusionModelForwardMode):
        if mode == DiffusionModelForwardMode.TRAIN:
            t = torch.randint(0, self.denoise_steps, size=(batch.size(0),), dtype=torch.int32)
            return self.reconstruct(batch, label, t)
        else:
            t = torch.full((batch.size(0),), 
                           self.denoise_steps - 1, device=batch.device)
            return self.reconstruct(batch, label, t)


class FIDInception(nn.Module):
    FEATURE_MAP_SIZE = 2048

    def __init__(self):
        super().__init__()
        self.inception = inception_v3(Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Identity() # type:ignore


    @staticmethod
    def fid_score(mu1: torch.Tensor, mu2: torch.Tensor, sigma1: torch.Tensor, sigma2: torch.Tensor):
        mu_dist = F.mse_loss(mu1, mu2, reduction="sum")
        sigma12 = sigma1.mm(sigma2)
        eigvec, eigval = torch.linalg.eig(sigma12)
        eigval = eigval.sqrt()
        sigma12_sqrt = eigvec.mm(eigval).mm(eigvec.T)

        sigma_trace = torch.trace(sigma1 + sigma2 - 2 * sigma12_sqrt)
        return mu_dist + sigma_trace


    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.inception(batch)
