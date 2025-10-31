from torch import nn
import torch
import typing
import abc

from torch._prims_common import device_or_default

from config import VAEConfig, ImageInfo


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


class ResNetBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 num_groups: int):
        super().__init__()

        self.net = nn.Sequential(
                ConvBlock(in_channels, out_channels, num_groups=num_groups),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups, out_channels)
        )

        if in_channels != out_channels:
            self.skip_connection: nn.Module = nn.Conv2d(in_channels, out_channels, 
                                                        kernel_size=3, padding=1)
        else:
            self.skip_connection: nn.Module = nn.Identity()

    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.net(image)
        skipped: torch.Tensor = self.skip_connection(image)

        return (output + skipped).relu_()


class ResNetBlockList(nn.Module):
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()

        self.net = nn.Sequential(
            ResNetBlock(in_channels, out_channels, num_groups),
            *(ResNetBlock(out_channels, out_channels, num_groups) for _ in range(num_blocks - 1))
        )


    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.net(batch)


class AttentionBlock(nn.Module):
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
                 use_attention: bool):
        super().__init__()

        layers: list[nn.Module] = [
            ResNetBlockList(
                    num_resnet_blocks,
                    in_channels,
                    out_channels,
                    num_groups)
        ]
        if use_attention:
            layers.append(AttentionBlock(out_channels, num_heads))

        self.net = nn.Sequential(*layers)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class DecoderStage(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 num_resnet_blocks: int,
                 num_groups: int,
                 num_heads: int,
                 use_attention: int):
        super().__init__()

        layers: list[nn.Module] = [
            ResNetBlockList(num_resnet_blocks, 
                            in_channels, 
                            out_channels,
                            num_groups)
        ]
        if use_attention:
            layers.append(AttentionBlock(out_channels, num_heads))
        
        self.net = nn.Sequential(*layers)


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Autoencoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor: ...



class VAEEncoder(nn.Module):
    def __init__(self, config: VAEConfig, image_info: ImageInfo):
        super().__init__()
        encoder_layers: list[nn.Module] = [
            nn.Conv2d(image_info.depth, 
                      config.num_channels[0],
                      kernel_size=3,
                      padding=1),
        ]

        for i in range(len(config.num_channels) - 1):
            in_channels = config.num_channels[i]
            out_channels = config.num_channels[i + 1]
            encoder_layers.append(EncoderStage(in_channels, 
                                               out_channels, 
                                               config.num_resnet_blocks, 
                                               config.layer_norm_num_groups,
                                               0, 
                                               False))
            encoder_layers.append(nn.AvgPool2d(kernel_size=2))

        encoder_layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder_layers)


    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.encoder(batch)


class VAEDecoder(nn.Module):
    def __init__(self, config: VAEConfig, image_info: ImageInfo):
        super().__init__()

        deep_num_channels = config.num_channels[-1]
        decoder_layers: list[nn.Module] = [
                ResNetBlock(deep_num_channels, deep_num_channels, config.layer_norm_num_groups),
                AttentionBlock(deep_num_channels, config.attention_heads),
                ResNetBlock(deep_num_channels, deep_num_channels, config.num_resnet_blocks)
        ]
        for i in reversed(range(len(config.num_channels) - 1)):
            in_channels = config.num_channels[i + 1]
            out_channels = config.num_channels[i]
            decoder_layers.append(DecoderStage(in_channels, in_channels, 
                                               config.num_resnet_blocks,
                                               config.layer_norm_num_groups,
                                               0,
                                               False))
            decoder_layers.append(ConvTransposeBlock(in_channels, 
                                                     out_channels, num_groups=config.layer_norm_num_groups))
            
        decoder_layers.append(nn.Conv2d(
            in_channels=config.num_channels[0],
            out_channels=image_info.depth,
            kernel_size=3,
            padding=1
        ))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_vector)


class VAE(nn.Module, Autoencoder):
    def __init__(self, 
                 config: VAEConfig,
                 image_info: ImageInfo):
        super().__init__()
        
        self.encoder = VAEEncoder(config, image_info)
        
        feature_map_height: int = image_info.size[0] // (2 ** (len(config.num_channels) - 1))
        feature_map_width: int = image_info.size[1] // (2 ** (len(config.num_channels) - 1))
        feature_map_depth: int = config.num_channels[-1]
        feature_map_numel: int = feature_map_height * feature_map_width * feature_map_depth

        self.mean_net = nn.Linear(feature_map_numel,config.latent_dim)
        self.logvar_net = nn.Linear(feature_map_numel,config.latent_dim)
        self.decoder = nn.Sequential(
            FCBlock(config.latent_dim, feature_map_numel),
            nn.Unflatten(dim=1, unflattened_size=(feature_map_depth, feature_map_height, feature_map_width)),
            VAEDecoder(config, image_info)
        )


    def encode_vars(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.encoder(image)
        mean = self.mean_net(feature_maps)
        logvar = self.logvar_net(feature_maps)
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
        return self.decoder(latent_vector)


    def forward(self, image: torch.Tensor):
        mean, logvar = self.encode_vars(image)
        latent_vector = self.sample(mean, logvar)
        return self.decode(latent_vector), mean, logvar


class DiffusionModel:
    def __init__(self):
        pass
