import dataclasses

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
    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0

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
        dtype=None,
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
        output = jnp.dot(x, self.kernel)

        if out_sharding is not None:
            output = jax.lax.with_sharding_constraint(
                output, jax.sharding.NamedSharding(jax.sharding.Mesh(), out_sharding)
            )

        if self.bias is not None:
            output = output + self.bias

        return output


# --- 4. ATTENTION MECHANISM ---
class BartAttention(nnx.Module):
    """
    Multi-head attention layer.
    Handles Self-Attention, Cross-Attention, and KV-Caching.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        is_cross_attention: bool = False,
        config: BartConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention

        shd = config.shd_cfg if config else ShardingCfg.no_sharding()

        self.q_proj = ShardedLinear(
            embed_dim, embed_dim, kernel_sharding=shd.qkv_kernel, bias_sharding=shd.qkv_bias, rngs=rngs
        )

        self.k_proj = ShardedLinear(
            embed_dim, embed_dim, kernel_sharding=shd.qkv_kernel, bias_sharding=shd.qkv_bias, rngs=rngs
        )

        self.v_proj = ShardedLinear(
            embed_dim, embed_dim, kernel_sharding=shd.qkv_kernel, bias_sharding=shd.qkv_bias, rngs=rngs
        )

        self.out_proj = ShardedLinear(
            embed_dim, embed_dim, kernel_sharding=shd.o_kernel, bias_sharding=shd.o_bias, rngs=rngs
        )

    def _reshape(self, tensor: jax.Array, seq_len: int, bsz: int) -> jax.Array:
        tensor = tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return jnp.transpose(tensor, (0, 2, 1, 3))

    def __call__(
        self,
        hidden_states: jax.Array,
        key_value_states: jax.Array | None = None,
        past_key_value: tuple[jax.Array, jax.Array] | None = None,
        attention_mask: jax.Array | None = None,
        layer_head_mask: jax.Array | None = None,
        output_attentions: bool = False,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array] | None, jax.Array | None]:
        bsz, tgt_len, _ = hidden_states.shape

        # If Cross-Attention (inside Decoder) use encoder outputs (key_value_states)
        is_cross = self.is_cross_attention and (key_value_states is not None)
        kv_input = key_value_states if is_cross else hidden_states

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(kv_input)
        value_states = self.v_proj(kv_input)

        query_states = self._reshape(query_states, tgt_len, bsz)

        kv_seq_len = key_states.shape[1]
        key_states = self._reshape(key_states, kv_seq_len, bsz)
        value_states = self._reshape(value_states, kv_seq_len, bsz)

        if past_key_value is not None:
            key_states = jnp.concatenate([past_key_value[0], key_states], axis=2)
            value_states = jnp.concatenate([past_key_value[1], value_states], axis=2)

        current_key_value = (key_states, value_states) if self.is_decoder else None

        attn_weights = jnp.matmul(query_states, jnp.swapaxes(key_states, -1, -2))
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, current_key_value)
        if output_attentions:
            outputs += (attn_weights,)
            return outputs

        return outputs + (None,)


class BartEncoderLayer(nnx.Module):
    """
    Single Encoder Block: Self Attention -> LayerNorm -> MLP -> LayerNorm
    """

    def __init__(self, config: BartConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            is_cross_attention=False,
            config=config,
            rngs=rngs,
        )
        self.self_attn_layer_norm = nnx.LayerNorm(self.embed_dim, rngs=rngs)

        self.fc1 = ShardedLinear(
            self.embed_dim,
            config.encoder_ffn_dim,
            kernel_sharding=config.shd_cfg.mlp_kernel,
            bias_sharding=config.shd_cfg.mlp_bias,
            rngs=rngs,
        )

        self.fc2 = ShardedLinear(
            self.encoder_ffn_dim,
            config.embed_dim,
            kernel_sharding=config.shd_cfg.mlp_kernel,
            bias_sharding=config.shd_cfg.mlp_bias,
            rngs=rngs,
        )
        self.final_layer_norm = nnx.LayerNorm(self.embed_dim, rngs=rngs)

        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.activation_dropout = nnx.Dropout(config.activation_dropout, rngs=rngs)
        self.activation_fn = nnx.gelu

    def __call__(
        self, hidden_states: jax.Array, attention_mask: jax.Array | None = None, output_attentions: bool = False
    ):
        residual = hidden_states

        # Self Attention Block
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        attn_output = self.dropout(attn_output)
        hidden_states = self.self_attn_layer_norm(residual + attn_output)

        # MLP
        residual = hidden_states

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        if output_attentions:
            return (hidden_states, attn_outputs[1])

        return (hidden_states,)


class BartLearnedPositionalEmbedding(nnx.Module):
    """
    Bart uses Learned positional embeddings upto a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *, rngs: nnx.Rngs):
        # Bart is set up so that if padding_idx is specified then offset embedding ids by 2
        self.offset = 2
        self.embedding = nnx.Embed(
            num_embeddings + self.offset, embedding_dim, embedding_init=nnx.initializers.normal(0.02), rngs=rngs
        )

    def __call__(self, input_ids: jax.Array, past_key_values_length: int = 0, position_ids: jax.Array | None = None):
        if position_ids is None:
            bsz, seq_len = input_ids.shape[:2]
            position_ids = jnp.arange(past_key_values_length, past_key_values_length + seq_len, dtype=jnp.int32)

            position_ids = jnp.broadcast_to(jnp.expand_dims(position_ids, 0), (bsz, seq_len))

        return self.embedding(position_ids + self.offset)


class BartEncoder(nnx.Module):
    """
    BART Encoder consists of multiple BartEncoderLayers.
    """

    def __init__(self, config: BartConfig, embed_tokens: nnx.Embed | None = None, *, rngs: nnx.Rngs):
        self.config = config

        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)
        self.layerdrop = config.encoder_layerdrop

        self.embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings

        self.embed_scale = jnp.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nnx.Embed(
                config.vocab_size, self.embed_dim, embedding_init=nnx.initializers.normal(config.init_std), rngs=rngs
            )

        self.embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, self.embed_dim, rngs=rngs)

        self.layers = [BartEncoderLayer(config, rngs=rngs) for _ in range(config.encoder_layers)]
        self.layernorm_embedding = nnx.LayerNorm(self.embed_dim, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        input_embeds: jax.Array | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        if input_ids is not None and input_embeds is not None:
            raise ValueError("You cannot specify both input_ids and input_embeds at the same time")
        elif input_ids is not None:
            input_node = input_ids
        elif input_embeds is not None:
            input_node = input_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or input_embeds")

        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_node)

        hidden_states = input_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return (hidden_states, all_hidden_states, all_attentions)
