from dataclasses import dataclass
import typing


@dataclass(frozen=True, slots=True)
class VAEConfig:
    learn_rate: float
    num_channels: tuple[int, ...]
    latent_dim: int
    num_resnet_blocks: int
    num_attention_heads: int
    layer_norm_num_groups: int
    should_downsample_in_block: typing.Sequence[tuple[bool, bool]]
    weight_decay: float


@dataclass(frozen=True, slots=True)
class ImageInfo:
    size: tuple[int, int]
    depth: int


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    learn_rate: float
    noise_start: float
    noise_end: float
    denoise_steps: int
    num_channels: tuple[int, ...]
    num_resnet_blocks: int
    num_attention_heads: int
    layer_norm_num_groups: int
    should_downsample_in_block: typing.Sequence[tuple[bool, bool]]
    weight_decay: float
