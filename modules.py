from torch import nn
import torch
import typing
import abc

from config import VAEConfig, ImageInfo, DiffusionConfig


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


class ConvTransposeBlock(nn.Module):
  """
  A ConvTranspose2d that doubles its inputs size
  ConvTransposeBlock(height, width) -> (height * 2, width * 2)
  """

  def __init__(self, in_channels: int, 
               out_channels: int, 
               kernel_size: int = 3,
               stride: int = 2,
               padding: int = 1,
               num_groups: int = 32,
               output_padding: int = 1):
      super().__init__()

      self.net = nn.Sequential(
              nn.ConvTranspose2d(in_channels, 
                                 out_channels, 
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 output_padding=output_padding,
                                 stride=stride,
                                 bias=False),
              nn.GroupNorm(num_groups, out_channels),
              nn.SiLU(inplace=True),
      )

  def forward(self, batch: torch.Tensor) -> torch.Tensor:
      return self.net(batch)


class Sequential2(nn.ModuleList):
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
            if in_channels != out_channels:
                self.embedding_projection_net = nn.Linear(
                        out_channels,
                        embedding_dim  
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

        if embedding and self.embedding_projection_net:
            embedding_projection = self.embedding_projection_net(embedding)
            output = output + embedding_projection

        skipped: torch.Tensor = self.skip_connection(batch)
        output = self.output_net(output)
        return (output + skipped).relu()


class ResNetBlockList(Sequential2):
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
        return patches.view(batch, channels, height, width)


class EncoderStage(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_resnet_blocks: int,
                 num_groups: int,
                 num_heads: int,
                 use_attention: bool,
                 embedding_dim: typing.Optional[int]):
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

        self.avgpool = nn.AvgPool2d(kernel_size=2)


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
                 embedding_dim: typing.Optional[int]):
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

        self.upsampler = ConvTransposeBlock(out_channels, 
                                                 out_channels, 
                                                 num_groups=num_groups)
    

    def forward(self, images: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        output: torch.Tensor = self.resnet_list(images, embedding)
        output = self.transformer(output)
        return self.upsampler(output)


class Autoencoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor: ...


class DownsamplePass(Sequential2):
    def __init__(self, 
                 in_channels: int, 
                 num_channels: tuple[int, ...],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int,
                 use_attention_for_block: typing.Sequence[bool],
                 embedding_dim: typing.Optional[int]):
        layers: list[nn.Module] = []

        for i in range(len(num_channels) - 1):
            in_channels = num_channels[i]
            out_channels = num_channels[i + 1]
            layers.append(EncoderStage(in_channels, 
                                       out_channels, 
                                       num_resnet_blocks, 
                                       layer_norm_num_groups,
                                       num_attention_heads, 
                                       use_attention_for_block[i],
                                       embedding_dim))

        super().__init__(*layers)


class AutoEncoder(nn.Sequential):
    def __init__(self, 
                 in_channels: int, 
                 num_channels: tuple[int, ...],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int):
        deep_num_channels = num_channels[-1]

        if in_channels != num_channels[0]:
            project_channels = nn.Conv2d(in_channels,
                  num_channels[0],
                  kernel_size=3,
                  padding=1)
        else:
            project_channels = nn.Identity()

        layers = (
                project_channels,
            DownsamplePass(in_channels, 
                           num_channels, 
                           num_resnet_blocks,
                           layer_norm_num_groups,
                           num_attention_heads,
                           (False,) * len(num_channels),
                           None),
            ResNetBlock(deep_num_channels, deep_num_channels, 
                        layer_norm_num_groups, None),
            PixelTransformer(deep_num_channels, num_attention_heads),
            ResNetBlock(deep_num_channels, deep_num_channels, 
                        num_resnet_blocks, None)
        )
        super().__init__(*layers)


class UpsamplePass(Sequential2):
    def __init__(self, 
                 out_channels: int, 
                 num_channels: tuple[int, ...],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int,
                 use_attention_for_block: typing.Sequence[bool],
                 embedding_dim: typing.Optional[int]):
        layers: list[nn.Module] = []

        for i in reversed(range(len(num_channels) - 1)):
            in_channels = num_channels[i + 1]
            out_channels = num_channels[i]
            layers.append(DecoderStage(in_channels, 
                                       out_channels, 
                                       num_resnet_blocks,
                                       layer_norm_num_groups,
                                       num_attention_heads,
                                       use_attention_for_block[i],
                                       embedding_dim))
        super().__init__(*layers)


class AutoDecoder(nn.Module):
    def __init__(self, 
                 out_channels: int, 
                 num_channels: tuple[int, ...],
                 num_resnet_blocks: int,
                 layer_norm_num_groups: int,
                 num_attention_heads: int):
        super().__init__()

        deep_num_channels = num_channels[-1]
        self.decoder = nn.Sequential(
                ResNetBlock(deep_num_channels, deep_num_channels, 
                            layer_norm_num_groups, None),
                PixelTransformer(deep_num_channels, num_attention_heads),
                ResNetBlock(deep_num_channels, deep_num_channels, 
                            num_resnet_blocks, None),
                UpsamplePass(out_channels,
                             num_channels,
                             num_resnet_blocks,
                             layer_norm_num_groups,
                             num_attention_heads,
                             (False,) * len(num_channels),
                             None),
                nn.Conv2d(
                        in_channels=num_channels[0],
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1
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
                num_attention_heads=config.num_attention_heads
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
                config.num_attention_heads
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
                num_classes
        )
        self.middle = nn.Sequential(
                ResNetBlock(
                    config.num_channels[-1],
                    config.num_channels[-1],
                    num_groups=config.layer_norm_num_groups,
                    embedding_dim=num_classes
                ),
                PixelTransformer(
                    config.num_channels[-1],
                    num_heads=config.num_attention_heads
                ),
                ResNetBlock(
                    config.num_channels[-1],
                    config.num_channels[-1],
                    num_groups=config.layer_norm_num_groups,
                    embedding_dim=num_classes
                ),
        )
        self.upsample = UpsamplePass(
                latent_channels,
                config.num_channels,
                config.num_resnet_blocks,
                config.layer_norm_num_groups,
                config.num_attention_heads,
                (True,) * len(config.num_channels),
                num_classes
        )


    def forward(self, batch: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.downsample(batch, embedding)
        output = self.middle(output)
        output = self.upsample(output)
        return output


class DiffusionModel(nn.Module):
    @staticmethod
    @torch.no_grad
    def beta_schedules(config: DiffusionConfig):
        timesteps = torch.arange(0, config.denoise_steps)
        factor = (config.noise_start - config.noise_end) / config.denoise_steps
        return timesteps, config.noise_start - factor * timesteps


    @staticmethod
    def generate_timesteps_embeddings(timesteps: torch.Tensor, embedding_size: int):
        out = torch.empty_like(timesteps)

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

        self.timesteps, self.betas = self.beta_schedules(config)
        self.timesteps_embeddings = self.generate_timesteps_embeddings(self.timesteps, num_classes)
        self.unet = UNet(config, latent_channels, num_classes)
        self.denoise_steps = config.denoise_steps

        self.alphas: torch.Tensor = 1 - self.betas
        self.alpha_bars: torch.Tensor = self.alphas.cumprod(0)

        self.register_buffer("timesteps", self.timesteps)
        self.register_buffer("timesteps_embeddings", self.timesteps_embeddings)
        self.register_buffer("betas", self.betas)
        self.register_buffer("alphas", self.alphas)
        self.register_buffer("alpha_bars", self.alpha_bars)


    def forward(self, batch: torch.Tensor, label: torch.Tensor):
        t = int(torch.randint(0, self.denoise_steps, size=(batch.size(0),)).int().item())
        eps = torch.normal(0, 1, size=batch.shape[1:], device=batch.device)
        timesteps_embedding = self.timesteps_embeddings[t]
        embedding = timesteps_embedding + label

        alpha_bar = self.alpha_bars[t]
        x_t = alpha_bar.sqrt() * batch + alpha_bar * eps

        prediction = self.unet(x_t, embedding)

        return eps, prediction
