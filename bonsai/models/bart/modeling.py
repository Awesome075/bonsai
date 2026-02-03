import dataclasses
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P


# --- 1. SHARDING CONFIGURATION ---
@dataclasses.dataclass(frozen=True)
class ShardingCfg:
    """Controls how the model weights are split."""

    qkv_kernel: P = P(None)
    qkv_bias: P = P(None)
    o_kernel: P = P(None)
    o_bias: P = P(None)

    # MLP Layers
    mlp_kernel: P = P(None)
    mlp_bias: P = P(None)

    @classmethod
    def no_sharding(cls):
        """Helper for default behavior (everything replicated)."""
        return cls()


# --- 2. MODEL CONFIGURATION ---
@dataclasses.dataclass(frozen=True)
class BartConfig:
    """
    Configuration for BART (Bidirectional and Auto-Regressive Transformers).
    """

    # Model Architecture
    vocab_size: int = 50265
    d_model: int = 1024
    encoder_layers: int = 12
    decoder_layers: int = 12
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    decoder_ffn_dim: int = 4096
    activation_function: str = "gelu"

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    classifier_dropout: float = 0.0

    # Initialization
    init_std: float = 0.02

    # Position & Embeddings
    max_position_embeddings: int = 1024
    scale_embedding: bool = False
    tie_word_embeddings: bool = True

    # Token IDs
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2

    # Metadata
    is_encoder_decoder: bool = True
    num_labels: int = 3
    use_cache: bool = True

    # JAX Sharding
    shd_cfg: ShardingCfg = dataclasses.field(default_factory=ShardingCfg.no_sharding)

    @classmethod
    def bart_large(cls):
        """Standard BART-Large configuration."""
        return cls(
            d_model=1024,
            encoder_layers=12,
            decoder_layers=12,
            encoder_attention_heads=16,
            decoder_attention_heads=16,
            encoder_ffn_dim=4096,
            decoder_ffn_dim=4096,
            scale_embedding=False,
        )


# --- 3. HELPER LAYERS ---
class ShardedLinear(nnx.Module):
    """
    Linear Layer with sharding support
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        kernel_sharding: P,
        bias_sharding: P = P(None),
        use_bias: bool = True,
        dtype: Optional[Any] = None,
        kernel_init: nnx.Initializers = nnx.initializers.normal(stddev=0.02),
        rngs: nnx.Rngs,
    ):
        # 1. Initialize Kernel (Weight)
        self.kernel = nnx.Param(
            kernel_init(rngs.params(), (in_dim, out_dim), dtype=dtype, out_sharding=kernel_sharding)
        )

        # 2. Initialize Bias
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_dim,), dtype=dtype, out_sharding=bias_sharding))
        else:
            self.bias = None

    def __call__(self, x: jax.Array, out_sharding: P = None) -> jax.Array:
        # Standard Matrix Multiplication
        output = jnp.dot(x, self.kernel)

        # Enforce output location (Standard JAX way)
        if out_sharding is not None:
            output = jax.lax.with_sharding_constraint(
                output, jax.sharding.NamedSharding(jax.sharding.Mesh(), out_sharding)
            )

        if self.bias is not None:
            output = output + self.bias

        return output
