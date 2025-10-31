"""
Defines VAE and DiffusionModel, as well as other components used

Each component receive output of a Linear/Conv2d layer.
Therefore each component should start by a GroupNorm and SiLU layer
"""
import math
from torch import nn
from torch.nn import functional as F
from config import EncoderDecoderConfig, VAEConfig, ImageInfo, DiffusionConfig
from dataclasses import dataclass
import dataclasses
import torch
import typing


class SequentialWithEmbedding(nn.ModuleList):
    """
    Similar to nn.Sequential, but passes embedding to sublayers
    """
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
    num_attention_heads: typing.Optional[int]


class PixelTransformer(nn.Module):
    """
    Self attention on each pixel
    """
    def __init__(self, 
                 num_channels: int, 
                 num_heads: int, 
                 layer_norm_num_groups: int):
        super().__init__()

        if num_channels % num_heads != 0:
            raise ValueError(f"num_channels({num_channels}) must be divisable by num_heads({num_heads})")
        self.qkv_projection = nn.Conv2d(num_channels, num_channels * 3, kernel_size=1)
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=layer_norm_num_groups, num_channels=num_channels)
        self.net = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        self.project_out = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        norm = self.norm(images)
        batch, channels, height, width = images.shape
        qkv = self.qkv_projection(norm) # shape: (batch, channels * 3, height, width)

        q, k, v = (t.view(batch, channels, height * width).transpose(1, 2) 
                   for t in qkv.chunk(3, 1))
        patches: torch.Tensor = self.net(q, k, v, need_weights=False)[0] #(batch, height*width, channels)
        patches = patches.transpose(1, 2).view(batch, channels, height, width)
        patches = self.project_out(patches)
        return images + patches


class ResNetBlock(nn.Module):
    """
    Consists of 2 Conv2d + GroupNorm + SiLU block
    A skipped connection connects input to output
    Embedding is injected to the network here
    """
    def __init__(self, info: ResNetBlockInfo):
        super().__init__()

        self.input_net = nn.Sequential(
                nn.GroupNorm(
                    num_groups=info.layer_norm_num_groups, 
                    num_channels=info.in_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                        in_channels=info.in_channels, 
                        out_channels=info.out_channels, 
                        kernel_size=3, 
                        padding=1, 
                        bias=False),
        )
        self.output_net = nn.Sequential(
                nn.GroupNorm(
                    num_groups=info.layer_norm_num_groups, 
                    num_channels=info.out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    in_channels=info.out_channels, 
                    out_channels=info.out_channels, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False),
        )

        if info.embedding_dim is not None:
            self.embedding_projection_net = nn.Sequential(
                nn.SiLU(),
                nn.Linear(info.embedding_dim, info.out_channels),
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

        if info.num_attention_heads:
            self.attention = PixelTransformer(
                    info.out_channels, info.num_attention_heads, 
                    info.layer_norm_num_groups)
        else:
            self.attention = nn.Identity()

    
    def forward(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        output: torch.Tensor = self.input_net(batch)

        if embedding is not None and self.embedding_projection_net:
            embedding_projection: torch.Tensor = self.embedding_projection_net(embedding)
            embedding_projection = embedding_projection.view(*embedding_projection.shape[:2], 1, 1)
            output = output + embedding_projection

        skipped: torch.Tensor = self.skip_connection(batch)
        output = self.output_net(output) + skipped
        return self.attention(output)


class ResNetBlockList(SequentialWithEmbedding):
    """
    Usually more than 1 ResNetBlock is used per resolution. This construct all of them
    """
    def __init__(self, num_blocks: int, info: ResNetBlockInfo):
        inner_info = dataclasses.replace(info, in_channels=info.out_channels)
        super().__init__(
            ResNetBlock(info),
            *(ResNetBlock(inner_info) for _ in range(num_blocks - 1))
        )


class EncoderStage(nn.Module):
    """
    Pass input to a ResNetBlockList, then optionally downsample it

    Side note: there is no DecoderStage because UNet has to handle skipped connection where VAE does not
    """
    def __init__(self, stage_index: int, config: EncoderDecoderConfig):
        super().__init__()

        channels = (config.num_channels[0], *config.num_channels)
        info = ResNetBlockInfo(
            in_channels=channels[stage_index],
            out_channels=channels[stage_index + 1],
            layer_norm_num_groups=config.layer_norm_num_groups,
            embedding_dim=config.embedding_dim,
            num_attention_heads=config.num_attention_heads_for_block(stage_index)
        )
        self.resnet_list = ResNetBlockList(
        config.num_resnet_blocks,
            info
        )


    def conv(self, batch: torch.Tensor, embedding: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Feed batch through ResNetBlockList
        """
        output = self.resnet_list(batch, embedding)
        return output


    def downsample(self, batch: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(batch, kernel_size=2)


    def forward(self, batch: torch.Tensor, 
                embedding: typing.Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.conv(batch, embedding)
        downsampled = self.downsample(output)
        return output, downsampled


class VAE(nn.Module):
    def __init__(self, config: VAEConfig, image_info: ImageInfo):
        super().__init__()
        
        self.project_channels = nn.Conv2d(
                        in_channels=image_info.depth,
                        out_channels=config.encoder_decoder_config.num_channels[0],
                        kernel_size=3,
                        padding=1
                    )
        self.encoder = nn.ModuleList(
            [
                        EncoderStage(i, config.encoder_decoder_config)
                            for i in range(len(config.encoder_decoder_config.num_channels))
                    ]
        )

        self.should_downsample = config.encoder_decoder_config.should_downsample
        self.feature_to_mean_logvar = nn.Sequential(
                nn.GroupNorm(config.encoder_decoder_config.layer_norm_num_groups, 
                             config.encoder_decoder_config.num_channels[-1]),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                        in_channels=config.encoder_decoder_config.num_channels[-1],
                        out_channels=config.latent_dim * 2,
                        kernel_size=3,
                        padding=1
                    ),
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
                    embedding_dim=config.encoder_decoder_config.embedding_dim,
                    num_attention_heads=config.encoder_decoder_config.num_attention_heads_for_block(i)
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
                nn.GroupNorm(config.encoder_decoder_config.layer_norm_num_groups,
                             channels[0]),
                nn.SiLU(inplace=True),
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
        """
        Encodes image into a latent vector
        """
        mean, logvar = self.encode_vars(image)
        return self.sample(mean, logvar)


    def sample(self, mean: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.rand_like(mean)
        else:
            return mean


    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Decode latent_vector into an image
        """
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
                    embedding_dim=config.embedding_dim,
                    num_attention_heads=config.num_attention_heads_for_block(stage)
                )
        )


class UNet(nn.Module):
    """
    The UNet model, used to predict noise added in time t.
    This can be used for latent diffusion or pixel diffusion.
    """
    def __init__(self, 
                 config: DiffusionConfig, 
                 latent_channels: int):
        super().__init__()

        self.downsample_projection = nn.Conv2d(
            in_channels=latent_channels,
            out_channels=config.unet_config.num_channels[0],
            kernel_size=3,
            padding=1
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
            embedding_dim=config.unet_config.embedding_dim,
            num_attention_heads=None
        )
        self.mid_resnet1 = ResNetBlock(info)
        self.mid_pixel_transformer = PixelTransformer(
            num_channels=info.out_channels,
            num_heads=config.unet_config.num_attention_heads,
            layer_norm_num_groups=config.unet_config.layer_norm_num_groups
        )
        self.mid_resnet2 = ResNetBlock(info)

        self.upsample_passes = nn.ModuleList([
            UNetUpsampler(i, config=config.unet_config)
                for i in reversed(range(len(config.unet_config.num_channels) - 1))
        ])
        self.upsample_projection = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.GroupNorm(config.unet_config.layer_norm_num_groups, config.unet_config.num_channels[0]),
            nn.Conv2d(
                    in_channels=config.unet_config.num_channels[0],
                    out_channels=latent_channels,
                    kernel_size=3,
                    padding=1
            )
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


class DiffusionModel(nn.Module):
    """
    Encodes time and class embedding and pass it to UNet
    """
    @staticmethod
    @torch.no_grad
    def generate_timesteps_embeddings(T: int, embedding_size: int):
        timesteps = torch.arange(0, T)
        # Positional embedding:
        # PE(pos, 2i) = sin(pos / 10000*(2i/d))
        # PE(pos, 2i + 1) = cos(pos / 10000*(2i/d))
        # Compute it in log space since 10000*(2i/d) is too large
        out = torch.empty((timesteps.size(0), embedding_size), device=timesteps.device)
        # The bitand -2 rounds odd elements down to the closest even
        # This extracts 'i'(in the formula above) from 0..N
        i = torch.arange(0, embedding_size) & -2
        denom = torch.exp(-math.log(10000) * i / embedding_size)
        numerator = timesteps.reshape(timesteps.size(0), 1)
        angles = numerator * denom

        odds = angles[..., 1::2]
        evens = angles[..., ::2]
        
        out[..., ::2] = torch.sin(evens)
        out[..., 1::2] = torch.cos(odds)
        return out


    def __init__(self, config: DiffusionConfig, latent_channels: int, num_classes: int):
        super().__init__()

        assert config.unet_config.embedding_dim
        self.unet = UNet(config, latent_channels)
        self.timesteps_embedding = nn.Embedding.from_pretrained(
                self.generate_timesteps_embeddings(config.denoise_steps, 
                                                   config.unet_config.embedding_dim))
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


    def embed_from_label(self, label: typing.Optional[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        timesteps_embedding = self.timesteps_embedding(t)
        timesteps_embedding = self.timesteps_projection(timesteps_embedding)

        embedding = timesteps_embedding
        if label is not None:
            label_embedding = self.label_projection(label)
            embedding = embedding + label_embedding
        return embedding


    def forward(self, noise: torch.Tensor, t: torch.Tensor, label: typing.Optional[torch.Tensor]):
        embedding = self.embed_from_label(label, t)
        return self.unet(noise, embedding)


class DiffusionSampler(nn.Module):
    """
    Implements equation 4 and 7 in the DDPM paper to handles noise addition/removal
    This module contains no learnable parameter.

    Some methods takes a model parameter. This should be a DiffusionModel(or a wrapper around it)
    """
    def __init__(self, config: DiffusionConfig):
        super().__init__()

        assert config.unet_config.embedding_dim
        betas = torch.linspace(config.noise_start, config.noise_end, config.denoise_steps)
        self.betas = nn.Embedding.from_pretrained(betas.unsqueeze(1))
        self.denoise_steps = config.denoise_steps

        alphas = 1 - betas
        self.alphas = nn.Embedding.from_pretrained(alphas.unsqueeze(1))
        alpha_bars = alphas.cumprod(0)
        self.alpha_bars = nn.Embedding.from_pretrained(alpha_bars.unsqueeze(1))

        # Coefficients for x_0 = 1/sqrt(alpha_bar_t) * (x_t - sqrt(1 - alpha_bar_t) * eps)
        alpha_bars_sqrt_recp = alpha_bars.rsqrt()
        self.alpha_bars_sqrt_recp = nn.Embedding.from_pretrained(alpha_bars_sqrt_recp.unsqueeze(1))
        one_minus_alpha_bar_sqrt = (1 - alpha_bars).sqrt()
        self.one_minus_alpha_bar_sqrt = nn.Embedding.from_pretrained(one_minus_alpha_bar_sqrt.unsqueeze(1))

        # Coefficients for equation 7
        alpha_t_minus_one = torch.cat([torch.ones(1), alpha_bars])[:alpha_bars.size(0)]
        self.beta_bars = self.betas
        x_0_coeff = alpha_t_minus_one.sqrt() * betas / (1 - alpha_bars)
        self.x_0_coeff = nn.Embedding.from_pretrained(x_0_coeff.unsqueeze(1))
        x_t_coeff = alphas.sqrt() * (1 - alpha_t_minus_one) / (1 - alpha_bars)
        self.x_t_coeff = nn.Embedding.from_pretrained(x_t_coeff.unsqueeze(1))


    @torch.inference_mode()
    def denoise_step(self, x_t: torch.Tensor, eps_t: torch.Tensor, t: int) -> torch.Tensor:
        timestep = torch.full((1,), t, device=x_t.device)

        x_0 = self.alpha_bars_sqrt_recp(timestep) * (x_t - self.one_minus_alpha_bar_sqrt(timestep) * eps_t)
        
        mu_t = self.x_0_coeff(timestep) * x_0 + self.x_t_coeff(timestep) * x_t

        x_t = mu_t

        if t > 0:
            sigma_t = self.beta_bars(timestep)
            eps_t_prev = torch.randn_like(x_t)
            x_t = mu_t + sigma_t.sqrt() * eps_t_prev

        return x_t


    @torch.inference_mode()
    def denoise_with_steps(self, x_t: torch.Tensor, label: torch.Tensor, model: nn.Module):
        """
        Given x_t, compute x_0 while returning all intermediate x_t-1, x_t-2...
        """
        for t in reversed(range(self.denoise_steps)):
            timestep = torch.full((x_t.size(0), ), t, device=label.device)
            eps_t = model(x_t, timestep, label)
            x_t = self.denoise_step(x_t, eps_t, t)
            yield x_t
        return x_t


    @torch.inference_mode()
    def denoise(self, x_t: torch.Tensor, label: torch.Tensor, model: nn.Module):
        """
        Given x_t, compute x_0
        """
        for x_t in self.denoise_with_steps(x_t, label, model):
            pass

        return x_t


    @torch.inference_mode()
    def generate_with_steps(self, 
                            num_images: int, 
                            size: tuple[int, int], 
                            latent_dim: int, 
                            label: torch.Tensor,
                            model: nn.Module):
        """
        Generate an image while returning all intermediate noisy images
        """
        x_t = torch.randn((num_images, latent_dim, *size), 
                          device=label.device)
        yield from self.denoise_with_steps(x_t, label, model)


    @torch.inference_mode()
    def generate(self, 
                 num_images: int, 
                 size: tuple[int, int], 
                 latent_dim: int, 
                 label: torch.Tensor,
                 model: nn.Module) -> torch.Tensor:
        x_t = torch.randn((num_images, latent_dim, *size), 
                          device=label.device)
        return self.denoise(x_t, label, model)


    def add_noise_step(self, x_t: torch.Tensor, t: int):
        """
        Given x_t, compute x_t+1
        """
        beta_t = self.betas(t)
        eps = torch.normal(0, 1, size=x_t.shape)

        x_t_next: torch.Tensor = (1 - beta_t).sqrt() * x_t + beta_t.sqrt() * eps

        return x_t_next, eps


    def add_noise(self, batch: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given x_0, compute x_t
        """
        eps = torch.normal(0, 1, size=batch.shape, device=batch.device)
        alpha_bar = self.alpha_bars(t).view(batch.size(0), 1, 1, 1)
        x_t = alpha_bar.sqrt() * batch + (1 - alpha_bar).sqrt() * eps
        return x_t, eps
