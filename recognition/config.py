"""
Configurations for models
Definitions are in train.py
"""
from dataclasses import dataclass
import typing


@dataclass(frozen=True)
class EncoderDecoderConfig:
    """
    Both VAE and UNet are encoder-decoder networks
    They are structured as follow

    UpsampleBlock/DownsampleBlock
        ResNetList
            ResNetBlock 1
                Conv2d
                GroupNorm
                SiLU
                Conv2d
                GroupNorm
                SiLU
                (Optional)SelfAttention
            ResNetBlock 2
                ...
    UpsampleBlock/DownsampleBlock
        ...
    """

    # The following 3 sequences should have the same length
    # num_channels[i] defines how many channels UpsampleBlock i outputs
    # should_downsample[i] controls if DownsampleBlock i should up sample
    # use_attention controls if all ResNetBlock in DownsampleBlock in i should have a SelfAttention layer
    # UpsampleBlock is define in the same way, but in reverse order
    num_channels: typing.Sequence[int]
    should_downsample: typing.Sequence[bool]
    use_attention: typing.Sequence[bool]

    # number of ResNetBlock in each ResNetList
    num_resnet_blocks: int
    # number of attention heads in each SelfAttention block
    num_attention_heads: int
    # number of groups in GroupNorm
    layer_norm_num_groups: int
    # Embedding dimension, only used in DiffusionModel
    embedding_dim: typing.Optional[int]


    def num_attention_heads_for_block(self, block: int) -> typing.Optional[int]:
        return self.num_attention_heads if self.use_attention[block] else None


@dataclass(frozen=True)
class VAEConfig:
    learn_rate: float
    # Number of channels in the latent image
    # Unlike conventional VAE that outputs a 1D latent vector, 
    # our VAE outputs a 3D latent image
    latent_dim: int
    weight_decay: float
    encoder_decoder_config: EncoderDecoderConfig

    def output_size(self, input_size: tuple[int, int]):
        scale = 2 ** sum(self.encoder_decoder_config.should_downsample)
        return input_size[0] // scale, input_size[1] // scale


@dataclass(frozen=True)
class ImageInfo:
    # height, width
    size: tuple[int, int]
    depth: int


@dataclass(frozen=True)
class DiffusionConfig:
    learn_rate: float
    weight_decay: float
    noise_start: float
    noise_end: float
    denoise_steps: int
    latent_scale_factor: float

    unet_config: EncoderDecoderConfig
