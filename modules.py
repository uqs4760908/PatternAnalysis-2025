from torch import nn
from torch.nn import functional as F
from config import VAEConfig, ImageInfo, DiffusionConfig
from enum import Enum
from torchvision.models import inception_v3, Inception_V3_Weights
import torch
import typing
import abc


class FCBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.SiLU(inplace=True)
        )


    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3, 
                 num_groups=32,
                 padding: typing.Optional[int] = None):
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                          kernel_size, padding=padding, bias=False),
                nn.GroupNorm(num_groups, out_channels),
                nn.SiLU(inplace=True)
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class SequentialWithEmbedding(nn.ModuleList):
    def __init__(self, *layers: nn.Module):
        super().__init__(layers)


    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None):
        for layer in self:
            batch = layer(batch, embedding)
        return batch


class ResNetBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 num_groups: int,
                 embedding_dim: typing.Optional[int]):
        super().__init__()

        self.input_net = ConvBlock(in_channels, out_channels, num_groups=num_groups)
        self.output_net = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                          padding=1, bias=False),
                nn.GroupNorm(num_groups, out_channels)
        )

        if embedding_dim is not None:
            if embedding_dim != out_channels:
                self.embedding_projection_net = nn.Linear(
                        embedding_dim,
                        out_channels
                )
            else:
                self.embedding_projection_net = nn.Identity()
        else:
            self.embedding_projection_net = None

        if in_channels != out_channels:
            self.skip_connection: nn.Module = nn.Conv2d(in_channels, out_channels, 
                                                        kernel_size=3, padding=1)
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
    def __init__(self, 
                 num_blocks: int, 
                 in_channels: int, 
                 out_channels: int, 
                 num_groups: int,
                 embedding_dim: typing.Optional[int]):
        super().__init__(
            ResNetBlock(in_channels, out_channels, 
                        num_groups, embedding_dim),
            *(ResNetBlock(out_channels, 
                          out_channels, 
                          num_groups,
                          embedding_dim) for _ in range(num_blocks - 1))
        )


class PixelTransformer(nn.Module):
    def __init__(self, num_channels: int, num_heads: int):
        super().__init__()

        self.qkv_projection = nn.Conv2d(num_channels, num_channels * 3, kernel_size=1)
        self.net = nn.MultiheadAttention(num_channels, num_heads)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        images = self.qkv_projection(images)

        q, k, v = (t.view(batch, channels, height * width).transpose(1, 2) 
                   for t in images.chunk(3, 1))
        patches: torch.Tensor = self.net(q, k, v, need_weights=False)[0]
        return patches.transpose(1, 2).view(batch, channels, height, width)


class EncoderStage(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_resnet_blocks: int,
                 num_groups: int,
                 num_heads: int,
                 use_attention: bool,
                 embedding_dim: typing.Optional[int],
                 should_downsample: tuple[bool, bool]):
        super().__init__()

        self.resnet_list = ResNetBlockList(
                    num_resnet_blocks,
                    in_channels,
                    out_channels,
                    num_groups,
                    embedding_dim)

        if use_attention:
            self.transformer = PixelTransformer(out_channels, num_heads)
        else:
            self.transformer = nn.Identity()

        self.avgpool = nn.AvgPool2d(kernel_size=(should_downsample[0] + 1, should_downsample[1] + 1))


    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        output: torch.Tensor = self.resnet_list(batch, embedding)
        output = self.transformer(output)
        output = self.avgpool(output)
        return output


class DecoderStage(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 num_resnet_blocks: int,
                 num_groups: int,
                 num_heads: int,
                 use_attention: int,
                 embedding_dim: typing.Optional[int],
                 should_upsample: tuple[bool, bool],
                 upsample_with_activation: bool = True):
        super().__init__()

        self.resnet_list = ResNetBlockList(num_resnet_blocks, 
                            in_channels, 
                            out_channels,
                            num_groups,
                            embedding_dim)

        if use_attention:
            self.transformer = PixelTransformer(out_channels, num_heads)
        else:
            self.transformer = nn.Identity()

        upsample_layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=(should_upsample[0] + 1, should_upsample[1] + 1),
                padding=1,
                output_padding=(int(should_upsample[0]), int(should_upsample[1])),
                bias=False
            ),
        ]

        # ugly hack for VAE, since its last layer does not need normalisation and activation
        if upsample_with_activation:
            upsample_layers.extend([
                nn.GroupNorm(num_groups, out_channels),
                nn.SiLU(inplace=True)
            ])


        self.upsampler = nn.Sequential(*upsample_layers)


    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        output: torch.Tensor = self.resnet_list(batch, embedding)
        output = self.transformer(output)
        output = self.upsampler(output)
        return output


class Autoencoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor: ...


class DownsamplePass(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 num_channels: typing.Sequence[int],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int,
                 use_attention_in_block: typing.Sequence[bool],
                 embedding_dim: typing.Optional[int],
                 should_downsample_in_block: typing.Sequence[tuple[bool, bool]]):
        super().__init__()

        if in_channels != num_channels[0]:
            self.project_channels = nn.Conv2d(
                    in_channels,
                    num_channels[0],
                    kernel_size=1,
                    padding=1)
        else:
            self.project_channels = nn.Identity()

        layers: list[nn.Module] = []

        out_channels_list = (*num_channels[1:], num_channels[-1])
        for in_channels, out_channels, should_downsample, use_attention in zip(
                                                                                num_channels, 
                                                                                out_channels_list, 
                                                                                should_downsample_in_block, 
                                                                                use_attention_in_block):
            layers.append(EncoderStage(in_channels, 
                                       out_channels, 
                                       num_resnet_blocks, 
                                       layer_norm_num_groups,
                                       num_attention_heads, 
                                       use_attention,
                                       embedding_dim,
                                       should_downsample))

        self.net = SequentialWithEmbedding(*layers)


    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.project_channels(batch)
        output = self.net(output, embedding)
        return output


class AutoEncoder(nn.Sequential):
    def __init__(self, 
                 in_channels: int, 
                 num_channels: typing.Sequence[int],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int,
                 should_downsample_in_block: typing.Sequence[tuple[bool, bool]]):
        deep_num_channels = num_channels[-1]

        layers = (
            DownsamplePass(in_channels, 
                           num_channels, 
                           num_resnet_blocks,
                           layer_norm_num_groups,
                           num_attention_heads,
                           (False,) * len(num_channels),
                           None,
                           should_downsample_in_block),
            ResNetBlock(deep_num_channels, deep_num_channels, 
                        layer_norm_num_groups, None),
            PixelTransformer(deep_num_channels, num_attention_heads),
            ResNetBlock(deep_num_channels, deep_num_channels, 
                        num_resnet_blocks, None)
        )
        super().__init__(*layers)


class UpsamplePass(nn.Module):
    def __init__(self, 
                 out_channels: int, 
                 num_channels: typing.Sequence[int],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int,
                 use_attention_for_block: typing.Sequence[bool],
                 embedding_dim: typing.Optional[int],
                 should_upsample_in_block: typing.Sequence[tuple[bool, bool]]):
        super().__init__()

        layers: list[nn.Module] = []

        in_channels_list = (*num_channels[1:], num_channels[-1])

        if out_channels != num_channels[0]:
            self.project_channels = nn.Conv2d(
                    num_channels[0],
                    out_channels,
                    kernel_size=1,
                    padding=1
            )
        else:
            self.project_channels = nn.Identity()

        for in_channels, out_channels, use_attention, should_upsample in reversed((*zip(
                                                                                in_channels_list, 
                                                                                num_channels, 
                                                                                use_attention_for_block, 
                                                                                should_upsample_in_block),)):
            layers.append(DecoderStage(in_channels, 
                                       out_channels, 
                                       num_resnet_blocks,
                                       layer_norm_num_groups,
                                       num_attention_heads,
                                       use_attention,
                                       embedding_dim,
                                       should_upsample))


        self.net = SequentialWithEmbedding(*layers)


    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self.net(batch, embedding)
        output = self.project_channels(output)
        return output


class AutoDecoder(nn.Module):
    def __init__(self, 
                 out_channels: int, 
                 num_channels: typing.Sequence[int],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int,
                 should_upsample_in_block: typing.Sequence[tuple[bool, bool]]):
        super().__init__()

        deep_num_channels = num_channels[-1]

        # VAE's last upsampling layer should not have normalisation and SiLU activation
        # Construct UpsamplePass for n-1 passes and construct DecoderStage manually to disable 
        # normalisation and SiLU
        self.decoder = nn.Sequential(
                ResNetBlock(deep_num_channels, deep_num_channels, 
                            layer_norm_num_groups, None),
                PixelTransformer(deep_num_channels, num_attention_heads),
                ResNetBlock(deep_num_channels, deep_num_channels, 
                            num_resnet_blocks, None),
                UpsamplePass(num_channels[1],
                             num_channels[1:],
                             num_resnet_blocks,
                             layer_norm_num_groups,
                             num_attention_heads,
                             (False,) * (len(num_channels) - 1),
                             None,
                             should_upsample_in_block[1:]),
                DecoderStage(
                    num_channels[1],
                    num_channels[0],
                    num_resnet_blocks,
                    layer_norm_num_groups,
                    num_attention_heads,
                    False,
                    None,
                    should_upsample_in_block[0],
                    False
                ),
                nn.Conv2d(
                    num_channels[0],
                    out_channels,
                    kernel_size=1,
                ),
                nn.Sigmoid()
        )


    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vector)


class VAE(nn.Module, Autoencoder):
    def __init__(self, 
                 config: VAEConfig,
                 image_info: ImageInfo):
        super().__init__()
        
        self.encoder = AutoEncoder(
                image_info.depth,
                config.num_channels,
                config.num_resnet_blocks,
                config.layer_norm_num_groups,
                num_attention_heads=config.num_attention_heads,
                should_downsample_in_block=config.should_downsample_in_block
        )
        self.feature_to_mean_logvar = nn.Conv2d(
                in_channels=config.num_channels[-1],
                out_channels=config.latent_dim * 2,
                kernel_size=3,
                padding=1
        )
        self.mean_logvar_to_feature = nn.Conv2d(
                in_channels=config.latent_dim,
                out_channels=config.num_channels[-1],
                kernel_size=3,
                padding=1
        )

        self.decoder = AutoDecoder(
                image_info.depth,
                config.num_channels,
                config.num_resnet_blocks,
                config.layer_norm_num_groups,
                config.num_attention_heads,
                config.should_downsample_in_block
        )


    def encode_vars(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.encoder(image)
        project: torch.Tensor = self.feature_to_mean_logvar(feature_maps)
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
        return self.decoder(feature)


    def forward(self, image: torch.Tensor):
        mean, logvar = self.encode_vars(image)
        latent_vector = self.sample(mean, logvar)
        return self.decode(latent_vector), mean, logvar


class UNet(nn.Module):
    def __init__(self, 
                 config: DiffusionConfig, 
                 latent_channels: int,
                 num_classes: int):
        super().__init__()

        self.downsample = DownsamplePass(
                latent_channels,
                config.num_channels,
                config.num_resnet_blocks,
                config.layer_norm_num_groups,
                config.num_attention_heads,
                (True,) * len(config.num_channels),
                num_classes,
                should_downsample_in_block=config.should_downsample_in_block
        )

        self.mid_resnet1 = ResNetBlock(
            config.num_channels[-1],
            config.num_channels[-1],
            num_groups=config.layer_norm_num_groups,
            embedding_dim=num_classes
        )
        self.mid_pixel_transformer = PixelTransformer(
            config.num_channels[-1],
            num_heads=config.num_attention_heads
        )
        self.mid_resnet2 = ResNetBlock(
            config.num_channels[-1],
            config.num_channels[-1],
            num_groups=config.layer_norm_num_groups,
            embedding_dim=num_classes
        )

        self.upsample = UpsamplePass(
                latent_channels,
                config.num_channels,
                config.num_resnet_blocks,
                config.layer_norm_num_groups,
                config.num_attention_heads,
                (True,) * len(config.num_channels),
                num_classes,
                config.should_downsample_in_block
        )


    def forward(self, batch: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.downsample(batch, embedding)
        output = self.mid_resnet1(output, embedding)
        output = self.mid_pixel_transformer(output)
        output = self.mid_resnet2(output, embedding)
        output = self.upsample(output, embedding)
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
        out = torch.empty((timesteps.size(0), embedding_size), device=timesteps.device)

        denomator = torch.pow(timesteps.size(0), 2 * (torch.arange(0, embedding_size) & -2))
        numerator = timesteps.reshape(timesteps.size(0), 1)
        angles = numerator / denomator

        odds = angles[1::2]
        evens = angles[::2]
        
        out[::2] = torch.sin(evens)
        out[1::2] = torch.cos(odds)
        return out


    def __init__(self, config: DiffusionConfig, latent_channels: int, num_classes: int):
        super().__init__()

        self.timesteps = torch.arange(0, config.denoise_steps)
        self.timesteps_embeddings = self.generate_timesteps_embeddings(self.timesteps, num_classes)
        self.betas = self.beta_schedules(config)
        self.unet = UNet(config, latent_channels, num_classes)
        self.denoise_steps = config.denoise_steps

        self.alphas: torch.Tensor = 1 - self.betas
        self.alpha_bars: torch.Tensor = self.alphas.cumprod(0)

        # Hack: IDE will not work without self.buffer = ... assignments,
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
        beta_t = self.betas[t].view(noise.size(0), 1, 1, 1)
        alpha_t = 1 - beta_t
        alpha_bar_t = self.alpha_bars[t].view(noise.size(0), 1, 1, 1)

        x_0 = (beta_t / (1 - alpha_bar_t).sqrt()) * eps
        return (1 / alpha_t.sqrt()) * (noise - x_0)


    def embed_from_label(self, label: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        timesteps_embedding = self.timesteps_embeddings[t]
        embedding = timesteps_embedding + label
        return embedding


    @torch.inference_mode()
    def generate(self, image_info: ImageInfo, label: torch.Tensor) -> torch.Tensor:
        noise = torch.normal(0, 1, 
                             size=(image_info.depth, *image_info.size), device=self.timesteps.device)

        for t in reversed(range(self.timesteps.size(0))):
            t = torch.full((1,), 1, device=label.device)
            embedding = self.embed_from_label(label, t)
            prediction = self.unet(noise, embedding)
            noise = self.predicted_noise_to_image(noise, prediction, t)

        return noise


    def reconstruct(self, batch: torch.Tensor, label: torch.Tensor, t: torch.Tensor):
        eps = torch.normal(0, 1, size=batch.shape, device=batch.device)
        embedding = self.embed_from_label(label, t)

        alpha_bar = self.alpha_bars[t].view(t.size(0), 1, 1, 1)
        x_t = alpha_bar.sqrt() * batch + (1 - alpha_bar).sqrt() * eps

        prediction: torch.Tensor = self.unet(x_t, embedding)
        prediction = F.interpolate(prediction, batch.shape[2:])

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
