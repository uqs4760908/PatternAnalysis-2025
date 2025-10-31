from dataclasses import dataclass
import typing


T = typing.TypeVar("T")


@dataclass(frozen=True)
class EncoderDecoderConfig:
    num_channels: typing.Sequence[int]
    should_downsample: typing.Sequence[bool]
    num_resnet_blocks: int
    num_attention_heads: int
    layer_norm_num_groups: int
    use_attention_in_up_down_sampling: bool
    embedding_dim: typing.Optional[int]


@dataclass(frozen=True)
class VAEConfig:
    learn_rate: float
    latent_dim: int
    weight_decay: float
    encoder_decoder_config: EncoderDecoderConfig


@dataclass(frozen=True)
class ImageInfo:
    size: tuple[int, int]
    depth: int


@dataclass(frozen=True)
class DiffusionConfig:
    learn_rate: float
    weight_decay: float
    noise_start: float
    noise_end: float
    denoise_steps: int

    unet_config: EncoderDecoderConfig
